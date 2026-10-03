"""
Helpers for parsing the Google Play Books web API.

Google Play Books talks "JSON+protobuf" to its backend: every message is a plain JSON
array in which the position of a value is its protobuf field number minus one. These
helpers translate those positional arrays (and the variables embedded in the listen page)
into simple structures the provider can work with, so the index juggling lives in one place.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from http.cookies import CookieError, SimpleCookie
from typing import Any
from xml.etree.ElementTree import ParseError

from defusedxml import ElementTree
from music_assistant_models.errors import LoginFailed

# the listen page embeds its data as javascript string literals, e.g. _OC_contentInfo = '...';
_PAGE_VAR_PATTERN = re.compile(r"(_OC_\w+) = '((?:[^'\\]|\\.)*)'")
_JS_ESCAPE_PATTERN = re.compile(r"\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)")
_JS_SIMPLE_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f", "v": "\v"}
_MPD_NAMESPACES = {
    "mpd": "urn:mpeg:DASH:schema:MPD:2011",
    "yt": "http://youtube.com/yt/2012/10/10",
}
# cookies that carry the SAPISID used to sign requests, in order of preference
_SAPISID_COOKIES = ("SAPISID", "__Secure-3PAPISID", "__Secure-1PAPISID")


@dataclass
class AudioRepresentation:
    """A single audio rendition of an audiobook as offered in the listen page manifest."""

    itag: str
    url: str
    codecs: str
    sample_rate: int
    bandwidth: int
    channels: int = 2
    content_length: int | None = None


@dataclass
class ResumePoint:
    """The most recent listening position Google Play Books holds for an audiobook."""

    annotation_id: str
    timestamp_ms: int
    position_ms: int


@dataclass
class ListenPage:
    """The parts of the listen page of an audiobook that the provider uses."""

    volume_id: str
    volume: list[Any] | None
    duration_ms: int
    chapters: list[tuple[str, int]] = field(default_factory=list)
    representations: list[AudioRepresentation] = field(default_factory=list)
    resume_point: ResumePoint | None = None


def get(obj: Any, *path: int) -> Any:
    """
    Return the value at the given index path of a nested JSON+protobuf array.

    Trailing empty fields are omitted by the backend, so a missing index is simply None.

    :param obj: The (nested) list to read from.
    :param path: The indexes to follow.
    """
    for index in path:
        if not isinstance(obj, list) or index >= len(obj):
            return None
        obj = obj[index]
    return obj


def to_int(value: Any, default: int = 0) -> int:
    """
    Convert a JSON+protobuf number to an int, int64 values arrive as strings.

    :param value: The raw value.
    :param default: The value to return when the raw value is missing or not a number.
    """
    try:
        return int(value)
    except TypeError, ValueError:
        return default


def sapisid_from_cookie(raw_cookie: str) -> str:
    """
    Return the SAPISID value used to sign API requests from the raw cookie string.

    :param raw_cookie: The cookie header as copied from the browser.
    """
    cookie: SimpleCookie = SimpleCookie()
    try:
        cookie.load(raw_cookie)
    except CookieError as err:
        raise LoginFailed("The provided cookie could not be parsed.") from err
    for key in _SAPISID_COOKIES:
        if key in cookie:
            return cookie[key].value
    raise LoginFailed(
        "Invalid Cookie detected. Cookie is missing the SAPISID field. "
        "Please ensure you copy the full cookie of a request to play.google.com."
    )


def get_authorization(sapisid: str, origin: str, timestamp: int | None = None) -> str:
    """
    Return the Authorization header value Google expects from its web clients.

    :param sapisid: The SAPISID value from the cookie.
    :param origin: The origin the requests are made from.
    :param timestamp: Unix timestamp to sign with, defaults to now.
    """
    timestamp = timestamp or int(time.time())
    digest = hashlib.sha1(f"{timestamp} {sapisid} {origin}".encode()).hexdigest()
    token = f"{timestamp}_{digest}"
    return f"SAPISIDHASH {token} SAPISID1PHASH {token} SAPISID3PHASH {token}"


def is_audiobook(volume: list[Any] | None) -> bool:
    """
    Return whether the given library volume is an audiobook (rather than an ebook).

    :param volume: The volume info array of a library entry.
    """
    return bool(get(volume, 16))


def latest_resume_point(annotations: list[Any] | None) -> ResumePoint | None:
    """
    Return the most recent listening position from a list of position annotations.

    :param annotations: Position annotations as returned by the API.
    """
    best: ResumePoint | None = None
    for annotation in annotations or []:
        annotation_id = get(annotation, 0)
        timestamp_ms = to_int(get(annotation, 2))
        position_ms = to_int(get(annotation, 4, 0), -1)
        if not annotation_id or position_ms < 0:
            continue
        if best is None or timestamp_ms > best.timestamp_ms:
            best = ResumePoint(annotation_id, timestamp_ms, position_ms)
    return best


def parse_library_response(response: list[Any]) -> list[tuple[list[Any], ResumePoint | None]]:
    """
    Return the audiobooks of a SyncUserLibrary response with their resume points.

    :param response: The decoded SyncUserLibrary response.
    """
    result: list[tuple[list[Any], ResumePoint | None]] = []
    for entry in get(response, 0) or []:
        volume = get(entry, 1)
        if not get(entry, 0) or not is_audiobook(volume):
            continue
        result.append((entry, latest_resume_point(get(entry, 2, 16, 0))))
    return result


def extract_page_vars(html: str) -> dict[str, Any]:
    """
    Return the (JSON decoded) _OC_ variables embedded in a Google Play Books page.

    :param html: The html of the page.
    """
    result: dict[str, Any] = {}
    for match in _PAGE_VAR_PATTERN.finditer(html):
        try:
            result[match.group(1)] = json.loads(_unescape_js(match.group(2)))
        except ValueError:
            continue
    return result


def parse_listen_page(volume_id: str, html: str) -> ListenPage:
    """
    Parse the listen page of an audiobook.

    :param volume_id: The volume id of the audiobook.
    :param html: The html of the listen page.
    """
    page_vars = extract_page_vars(html)
    content_info = page_vars.get("_OC_contentInfo")
    resource = page_vars.get("_OC_resourceResponse")
    overview = page_vars.get("_OC_volumeOverview")
    volume = get(overview, 1)
    duration_ms = to_int(get(content_info, 1)) or to_int(get(volume, 16, 1))
    chapters = [
        (str(get(chapter, 0) or ""), to_int(get(chapter, 1)))
        for chapter in get(content_info, 0, 0) or []
    ]
    representations: list[AudioRepresentation] = []
    if manifest := get(resource, 0, 1):
        representations = parse_dash_manifest(manifest)
    return ListenPage(
        volume_id=volume_id,
        volume=volume,
        duration_ms=duration_ms,
        chapters=chapters,
        representations=representations,
        resume_point=latest_resume_point(get(resource, 1, 0)),
    )


def parse_dash_manifest(manifest: str) -> list[AudioRepresentation]:
    """
    Return the audio representations of a DASH manifest, best quality first.

    :param manifest: The DASH manifest (xml).
    """
    try:
        root = ElementTree.fromstring(manifest)
    except ParseError:
        return []
    result: list[AudioRepresentation] = []
    for representation in root.iterfind(".//mpd:Representation", _MPD_NAMESPACES):
        base_url = representation.find("mpd:BaseURL", _MPD_NAMESPACES)
        if base_url is None or not base_url.text:
            continue
        channels = representation.find("mpd:AudioChannelConfiguration", _MPD_NAMESPACES)
        content_length = base_url.get(f"{{{_MPD_NAMESPACES['yt']}}}contentLength")
        result.append(
            AudioRepresentation(
                itag=representation.get("id", ""),
                url=base_url.text.strip(),
                codecs=representation.get("codecs", ""),
                sample_rate=to_int(representation.get("audioSamplingRate"), 44100),
                bandwidth=to_int(representation.get("bandwidth")),
                channels=to_int(channels.get("value") if channels is not None else None, 2),
                content_length=to_int(content_length) or None,
            )
        )
    return sorted(result, key=lambda rep: rep.bandwidth, reverse=True)


def _unescape_js(value: str) -> str:
    """Unescape the contents of a single quoted javascript string literal."""

    def _replace(match: re.Match[str]) -> str:
        escape = match.group(1)
        if escape[0] in "xu" and len(escape) > 1:
            return chr(int(escape[1:], 16))
        return _JS_SIMPLE_ESCAPES.get(escape, escape)

    return _JS_ESCAPE_PATTERN.sub(_replace, value)
