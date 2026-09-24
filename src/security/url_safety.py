"""
Guards against server-side request forgery (SSRF) when fetching a
user-supplied document URL, and enforces a hard cap on download size.
"""

import ipaddress
import logging
import socket
from urllib.parse import urljoin, urlparse

import requests

logger = logging.getLogger("url_safety")

ALLOWED_SCHEMES = {"http", "https"}
MAX_REDIRECTS = 5


def _is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _assert_resolves_to_public_ips(hostname: str) -> None:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve host: {hostname}") from e
    if not infos:
        raise ValueError(f"Could not resolve host: {hostname}")
    for info in infos:
        ip_str = info[4][0]
        if not _is_public_ip(ip_str):
            raise ValueError(f"Refusing to fetch a non-public address ({ip_str}) for host {hostname}")


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError("URL has no hostname")
    # Note: this re-resolves right before the request but there is an
    # unavoidable small TOCTOU window before the TCP connect (e.g. DNS
    # rebinding). That's an accepted, documented limitation here rather
    # than pulling in a connection-pinning HTTP adapter for this project's
    # scope.
    _assert_resolves_to_public_ips(parsed.hostname)


def safe_fetch(url: str, max_bytes: int, timeout: float = 10.0) -> bytes:
    """
    Download `url` while blocking SSRF targets (private/loopback/link-local/
    reserved addresses), re-validating every redirect hop, and aborting once
    more than `max_bytes` have been read.

    Raises ValueError for anything blocked/invalid, requests.RequestException
    for ordinary network failures.
    """
    current_url = url
    for _ in range(MAX_REDIRECTS + 1):
        _validate_url(current_url)
        response = requests.get(current_url, timeout=timeout, stream=True, allow_redirects=False)
        try:
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Redirect response had no Location header")
                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()

            content = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                content.extend(chunk)
                if len(content) > max_bytes:
                    raise ValueError(f"Download exceeded the maximum allowed size of {max_bytes} bytes")
            return bytes(content)
        finally:
            response.close()

    raise ValueError("Too many redirects")
