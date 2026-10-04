"""Google Play Books (audiobooks) support for Music Assistant."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from aiohttp import ClientError
from music_assistant_models.enums import (
    ContentType,
    ImageType,
    MediaType,
    ProviderFeature,
    StreamType,
)
from music_assistant_models.errors import (
    LoginFailed,
    MediaNotFoundError,
    RateLimited,
    ResourceTemporarilyUnavailable,
    UnplayableMediaError,
)
from music_assistant_models.media_items import (
    Audiobook,
    AudioFormat,
    MediaItemChapter,
    MediaItemImage,
    ProviderMapping,
    UniqueList,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.constants import CONF_ENTRY_UNOFFICIAL_PROVIDER
from music_assistant.helpers.throttle_retry import (
    ThrottlerManager,
    parse_retry_after,
    throttle_with_retries,
)
from music_assistant.helpers.util import html_to_markdown
from music_assistant.models.music_provider import MusicProvider

from .helpers import (
    ListenPage,
    ResumePoint,
    get,
    get_authorization,
    parse_library_response,
    parse_listen_page,
    sapisid_from_cookie,
    to_int,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry, ProviderConfig
    from music_assistant_models.media_items import MediaItemType
    from music_assistant_models.provider import ProviderManifest

    from music_assistant import MusicAssistant
    from music_assistant.models import ProviderInstanceType


CONF_COOKIE = "cookie"

GPB_DOMAIN = "https://play.google.com"
GPB_LISTEN_URL = f"{GPB_DOMAIN}/books/listen"
GPB_STORE_URL = f"{GPB_DOMAIN}/store/audiobooks/details"
GPB_API_URL = "https://playbooks-pa.clients6.google.com/$rpc/google.internal.play.books"
# public api key of the Google Play Books web client (see _OC_apiInfo in its pages)
GPB_API_KEY = "AIzaSyCWq1--9JnN9QM7k57Rc_qmt9c0OVy0rME"
GPB_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36"
)
# the web client syncs its listening position as client type 2
GPB_WEB_CLIENT_TYPE = 2
LIBRARY_PAGE_SIZE = 400
# how long a fetched listen page is reused, so a playback start fetches it only once
LISTEN_PAGE_MAX_AGE = 60
# stream urls are signed for a few minutes only, refresh them a bit before they expire
STREAM_URL_EXPIRY_MARGIN = 30
DEFAULT_STREAM_URL_EXPIRATION = 240
# a book counts as finished once the position is this close to the end
FULLY_PLAYED_MARGIN_MS = 30_000

SUPPORTED_FEATURES = {
    ProviderFeature.BROWSE,
    ProviderFeature.LIBRARY_AUDIOBOOKS,
}


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return GooglePlayBooksProvider(mass, manifest, config, SUPPORTED_FEATURES)


class GooglePlayBooksProvider(MusicProvider):
    """Provider for audiobooks in a Google Play Books library."""

    throttler: ThrottlerManager = ThrottlerManager(rate_limit=1, period=1)
    _cookie: str
    _sapisid: str
    _device_id: str
    _listen_pages: dict[str, tuple[float, ListenPage]]
    _position_parents: dict[str, str | None]

    @property
    def is_streaming_provider(self) -> bool:
        """Return True if the provider is a streaming provider."""
        return True

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return Config entries to configure this provider."""
        return (CONF_ENTRY_UNOFFICIAL_PROVIDER,)

    async def handle_async_init(self) -> None:
        """Set up the Google Play Books provider."""
        self._cookie = str(self.get_setup_value(CONF_COOKIE) or "")
        self._sapisid = sapisid_from_cookie(self._cookie)
        # a stable id per instance, so Google sees every position update coming from one device
        self._device_id = str(uuid.uuid5(uuid.NAMESPACE_URL, self.instance_id)).upper()
        self._listen_pages = {}
        self._position_parents = {}
        # cheap authenticated call to validate the cookie
        await self._rpc("library.v1.LibraryService/ListTags", [])

    async def sync_library(self, media_type: MediaType) -> None:
        """Run library sync for this provider."""
        try:
            await super().sync_library(media_type)
        except LoginFailed as err:
            # Every following sync fails the same way until the cookie is replaced,
            # so hand the provider back to the user for re-authentication.
            if self.available:
                self.unload_with_error(err)
            raise

    async def get_library_audiobooks(self) -> AsyncGenerator[Audiobook]:
        """Retrieve the audiobooks in the Google Play Books library."""
        response = await self._rpc(
            "library.v1.LibraryService/SyncUserLibrary",
            [
                None,
                None,
                None,
                None,
                [LIBRARY_PAGE_SIZE],
                [[[1, 2]], None, [[1, 2, 3]], 1],
                None,
                None,
                [None, 1],
            ],
        )
        entries = parse_library_response(response)
        total = to_int(get(response, 3, 0))
        if total > len(get(response, 0) or []):
            self.logger.warning(
                "Google Play Books reported %s library items but returned only %s",
                total,
                len(get(response, 0) or []),
            )
        for entry, resume_point in entries:
            yield self._parse_audiobook(entry[0], entry[1], resume_point)

    async def get_audiobook(self, prov_audiobook_id: str) -> Audiobook:
        """Get full audiobook details by id."""
        page = await self._get_listen_page(prov_audiobook_id)
        if not page.volume:
            raise MediaNotFoundError(f"Audiobook {prov_audiobook_id} not found")
        audiobook = self._parse_audiobook(prov_audiobook_id, page.volume, page.resume_point)
        if page.duration_ms:
            audiobook.duration = page.duration_ms // 1000
        audiobook.metadata.chapters = self._parse_chapters(page)
        return audiobook

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Return the content details for the given audiobook when it will be streamed."""
        page = await self._get_listen_page(item_id)
        if not page.representations:
            raise UnplayableMediaError(f"No audio stream available for audiobook {item_id}")
        representation = page.representations[0]
        expiration = DEFAULT_STREAM_URL_EXPIRATION
        if expire_ts := parse_qs(urlparse(representation.url).query).get("expire", [None])[0]:
            expiration = max(int(expire_ts) - int(time.time()) - STREAM_URL_EXPIRY_MARGIN, 0)
        self.logger.debug(
            "Using stream format %s (%s) for audiobook %s",
            representation.itag,
            representation.codecs,
            item_id,
        )
        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            audio_format=AudioFormat(
                content_type=ContentType.M4A,
                codec_type=ContentType.AAC,
                sample_rate=representation.sample_rate,
                channels=representation.channels,
                bit_rate=representation.bandwidth,
            ),
            media_type=MediaType.AUDIOBOOK,
            stream_type=StreamType.HTTP,
            path=representation.url,
            duration=page.duration_ms // 1000 or None,
            size=representation.content_length,
            can_seek=True,
            allow_seek=True,
            expiration=expiration,
        )

    async def get_resume_position(
        self, item_id: str, media_type: MediaType
    ) -> tuple[bool, int, datetime | None]:
        """
        Return the resume position Google Play Books holds for the given audiobook.

        :param item_id: The volume id of the audiobook.
        :param media_type: The media type of the item.
        """
        if media_type != MediaType.AUDIOBOOK:
            raise NotImplementedError
        page = await self._get_listen_page(item_id, max_age=0)
        if not (resume_point := page.resume_point):
            return False, 0, None
        return (
            _is_fully_played(resume_point.position_ms, page.duration_ms),
            resume_point.position_ms,
            datetime.fromtimestamp(resume_point.timestamp_ms / 1000, tz=UTC),
        )

    async def on_played(
        self,
        media_type: MediaType,
        prov_item_id: str,
        fully_played: bool,
        position: int,
        media_item: MediaItemType,
        is_playing: bool = False,
    ) -> None:
        """
        Sync the listening position of an audiobook back to Google Play Books.

        :param media_type: The media type of the played item.
        :param prov_item_id: The volume id of the audiobook.
        :param fully_played: Whether the audiobook has been played to the end.
        :param position: The last known position in seconds.
        :param media_item: The full media item that was played.
        :param is_playing: Whether the audiobook is currently playing.
        """
        if media_type != MediaType.AUDIOBOOK:
            return
        position_ms = position * 1000
        if fully_played and isinstance(media_item, Audiobook) and media_item.duration:
            position_ms = max(position_ms, media_item.duration * 1000)
        if prov_item_id not in self._position_parents:
            await self._get_listen_page(prov_item_id)
        parent_id = self._position_parents.get(prov_item_id)
        await self._rpc(
            "library.v1.LibraryService/SyncDocumentPosition",
            [
                prov_item_id,
                [
                    str(uuid.uuid4()).upper(),
                    [parent_id] if parent_id else [],
                    int(time.time() * 1000),
                    [self._device_id, None, [GPB_WEB_CLIENT_TYPE]],
                    [position_ms],
                ],
            ],
        )

    async def _get_listen_page(self, volume_id: str, max_age: int | None = None) -> ListenPage:
        """
        Return the parsed listen page of an audiobook, reusing a recently fetched one.

        :param volume_id: The volume id of the audiobook.
        :param max_age: Maximum age in seconds of a reused page, None for the default.
        """
        max_age = LISTEN_PAGE_MAX_AGE if max_age is None else max_age
        if cached := self._listen_pages.get(volume_id):
            fetched_at, page = cached
            if time.monotonic() - fetched_at < max_age:
                return page
        page = parse_listen_page(volume_id, await self._fetch_listen_page(volume_id))
        # like the web client, position updates refer to the position known when the book opened
        self._position_parents[volume_id] = (
            page.resume_point.annotation_id if page.resume_point else None
        )
        # drop pages that went stale so the cache only holds recently played books
        now = time.monotonic()
        self._listen_pages = {
            key: value
            for key, value in self._listen_pages.items()
            if now - value[0] < LISTEN_PAGE_MAX_AGE
        }
        self._listen_pages[volume_id] = (now, page)
        return page

    @throttle_with_retries
    async def _fetch_listen_page(self, volume_id: str) -> str:
        """
        Fetch the html of the listen page of an audiobook.

        :param volume_id: The volume id of the audiobook.
        """
        headers = {
            "User-Agent": GPB_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": self._cookie,
        }
        try:
            async with self.mass.http_session.get(
                GPB_LISTEN_URL, params={"id": volume_id}, headers=headers
            ) as response:
                self._raise_for_status(response.status, response.headers.get("Retry-After"))
                html = await response.text()
                if "accounts.google.com" in str(response.url):
                    raise LoginFailed(
                        "Your Google Play Books session is no longer valid. "
                        "Please reconfigure this provider with a fresh cookie."
                    )
                return html
        except (ClientError, TimeoutError) as err:
            raise ResourceTemporarilyUnavailable(f"Failed to load audiobook: {err}") from err

    @throttle_with_retries
    async def _rpc(self, method: str, body: list[Any]) -> Any:
        """
        Call a method of the Google Play Books API.

        :param method: The service and method, relative to google.internal.play.books.
        :param body: The JSON+protobuf request message.
        """
        headers = {
            "User-Agent": GPB_USER_AGENT,
            "Accept": "*/*",
            "Content-Type": "application/json+protobuf",
            "Origin": GPB_DOMAIN,
            "Referer": f"{GPB_DOMAIN}/",
            "Authorization": get_authorization(self._sapisid, GPB_DOMAIN),
            "X-Goog-Api-Key": GPB_API_KEY,
            "X-Goog-AuthUser": "0",
            "X-User-Agent": "grpc-web-javascript/0.1",
            "Cookie": self._cookie,
        }
        try:
            async with self.mass.http_session.post(
                f"{GPB_API_URL}.{method}", json=body, headers=headers
            ) as response:
                self._raise_for_status(response.status, response.headers.get("Retry-After"))
                return await response.json(content_type=None)
        except (ClientError, TimeoutError) as err:
            raise ResourceTemporarilyUnavailable(
                f"Google Play Books request failed: {err}"
            ) from err

    def _raise_for_status(self, status: int, retry_after: str | None) -> None:
        """Translate an unsuccessful HTTP status into the matching Music Assistant error."""
        if status < 400:
            return
        if status in (401, 403):
            raise LoginFailed(
                "Your Google Play Books session is no longer valid. "
                "Please reconfigure this provider with a fresh cookie."
            )
        if status == 404:
            raise MediaNotFoundError("Item not found on Google Play Books")
        if status == 429:
            raise RateLimited(
                "Google Play Books is rate limiting requests",
                backoff_time=parse_retry_after(retry_after),
            )
        if status >= 500:
            raise ResourceTemporarilyUnavailable(
                f"Google Play Books responded with status {status}",
                backoff_time=parse_retry_after(retry_after),
            )
        raise MediaNotFoundError(f"Google Play Books responded with status {status}")

    def _parse_audiobook(
        self, volume_id: str, volume: list[Any], resume_point: ResumePoint | None
    ) -> Audiobook:
        """Parse a Google Play Books volume into an Audiobook."""
        duration_ms = to_int(get(volume, 16, 1))
        audiobook = Audiobook(
            item_id=volume_id,
            provider=self.instance_id,
            name=get(volume, 0) or volume_id,
            duration=duration_ms // 1000,
            publisher=get(volume, 2),
            authors=UniqueList(author for author in get(volume, 1) or [] if author),
            narrators=UniqueList(narrator for narrator in get(volume, 16, 3) or [] if narrator),
            provider_mappings={
                ProviderMapping(
                    item_id=volume_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    url=get(volume, 9) or f"{GPB_STORE_URL}?id={volume_id}",
                    audio_format=AudioFormat(
                        content_type=ContentType.M4A,
                        codec_type=ContentType.AAC,
                    ),
                )
            },
        )
        if resume_point:
            audiobook.resume_position_ms = resume_point.position_ms
            audiobook.fully_played = _is_fully_played(resume_point.position_ms, duration_ms)
        else:
            audiobook.resume_position_ms = 0
            audiobook.fully_played = False
        if description := get(volume, 4):
            audiobook.metadata.description = html_to_markdown(description)
        if language := get(volume, 8):
            audiobook.metadata.languages = UniqueList([language])
        if release_date := _parse_release_date(get(volume, 3)):
            audiobook.metadata.release_date = release_date
        audiobook.metadata.genres = {
            genre for categories in get(volume, 20) or [] if (genre := get(categories, 0, 0))
        }
        audiobook.metadata.images = UniqueList(
            MediaItemImage(
                type=ImageType.THUMB,
                path=image_url,
                provider=self.instance_id,
                remotely_accessible=True,
            )
            for image in get(volume, 7) or []
            if (image_url := get(image, 0))
        )
        return audiobook

    def _parse_chapters(self, page: ListenPage) -> list[MediaItemChapter]:
        """Parse the chapters of a listen page, the end of a chapter is the start of the next."""
        chapters: list[MediaItemChapter] = []
        for index, (name, start_ms) in enumerate(page.chapters):
            next_start_ms = (
                page.chapters[index + 1][1] if index + 1 < len(page.chapters) else page.duration_ms
            )
            chapters.append(
                MediaItemChapter(
                    position=index + 1,
                    name=name or f"Chapter {index + 1}",
                    start=start_ms / 1000,
                    end=next_start_ms / 1000 if next_start_ms else None,
                )
            )
        return chapters


def _is_fully_played(position_ms: int, duration_ms: int) -> bool:
    """Return whether a position marks an audiobook of the given duration as finished."""
    return bool(duration_ms) and position_ms >= duration_ms - FULLY_PLAYED_MARGIN_MS


def _parse_release_date(value: str | None) -> datetime | None:
    """Parse a release date, which Google gives as a full date or only a year (and month)."""
    if not value:
        return None
    for date_format in ("%Y-%m-%d", "%Y-%m", "%Y"):
        with suppress(ValueError):
            return datetime.strptime(value, date_format).replace(tzinfo=UTC)
    return None
