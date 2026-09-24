"""
SSRF guard tests. IP-literal URLs resolve locally without a real network
call (getaddrinfo on a literal IP is a parse, not a lookup), so these run
fully offline and deterministically.
"""

import pytest

from docuqueryai.security.url_safety import _is_public_ip, _validate_url


class TestIsPublicIp:
    @pytest.mark.parametrize("ip", [
        "127.0.0.1",        # loopback
        "169.254.169.254",  # link-local / cloud metadata endpoint
        "10.0.0.1",         # private
        "172.16.0.1",       # private
        "192.168.1.1",      # private
        "0.0.0.0",          # unspecified
        "::1",              # IPv6 loopback
        "fc00::1",          # IPv6 unique local (private)
    ])
    def test_rejects_non_public_addresses(self, ip):
        assert _is_public_ip(ip) is False

    @pytest.mark.parametrize("ip", [
        "8.8.8.8",
        "1.1.1.1",
        "93.184.216.34",
    ])
    def test_accepts_public_addresses(self, ip):
        assert _is_public_ip(ip) is True


class TestValidateUrl:
    def test_rejects_loopback_ip_literal(self):
        with pytest.raises(ValueError):
            _validate_url("http://127.0.0.1/secret")

    def test_rejects_cloud_metadata_endpoint(self):
        with pytest.raises(ValueError):
            _validate_url("http://169.254.169.254/latest/meta-data/")

    def test_rejects_private_network_ip(self):
        with pytest.raises(ValueError):
            _validate_url("http://10.1.2.3/internal")

    def test_rejects_non_http_scheme(self):
        with pytest.raises(ValueError):
            _validate_url("file:///etc/passwd")

    def test_rejects_ftp_scheme(self):
        with pytest.raises(ValueError):
            _validate_url("ftp://8.8.8.8/somefile")

    def test_rejects_url_with_no_hostname(self):
        with pytest.raises(ValueError):
            _validate_url("http://")

    def test_accepts_public_ip_literal(self):
        # Should not raise.
        _validate_url("http://8.8.8.8/document.pdf")
