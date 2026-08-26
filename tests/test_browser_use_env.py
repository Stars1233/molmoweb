from urllib.parse import parse_qs, urlparse

import pytest

from utils.envs import browser_env


class _FakeChromium:
    def __init__(self, browser):
        self.browser = browser
        self.cdp_url = None

    def connect_over_cdp(self, cdp_url):
        self.cdp_url = cdp_url
        return self.browser


class _FakePlaywright:
    def __init__(self, browser):
        self.chromium = _FakeChromium(browser)


def test_browser_use_env_connects_to_provisioned_context(monkeypatch):
    page = object()
    context = type("Context", (), {"pages": [page]})()
    browser = type("Browser", (), {"contexts": [context]})()
    playwright = _FakePlaywright(browser)
    monkeypatch.setattr(browser_env, "_start_playwright", lambda: playwright)

    env = browser_env.BrowserUseEnv(
        api_key="key+with/slashes",
        proxy_country_code="de",
        session_timeout_minutes=30,
        viewport_width=1440,
        viewport_height=900,
    )
    env._launch()

    parsed = urlparse(playwright.chromium.cdp_url)
    assert parsed.scheme == "wss"
    assert parsed.netloc == "connect.browser-use.com"
    assert parse_qs(parsed.query) == {
        "apiKey": ["key+with/slashes"],
        "proxyCountryCode": ["de"],
        "timeout": ["30"],
        "browserScreenWidth": ["1440"],
        "browserScreenHeight": ["900"],
    }
    assert env.context is context
    assert env.page is page


def test_browser_use_env_requires_key():
    env = browser_env.BrowserUseEnv(api_key="")
    env.api_key = ""

    with pytest.raises(ValueError, match="BROWSER_USE_API_KEY required"):
        env._launch()


@pytest.mark.parametrize("timeout", [0, 241])
def test_browser_use_env_rejects_invalid_timeout(timeout):
    env = browser_env.BrowserUseEnv(api_key="test", session_timeout_minutes=timeout)

    with pytest.raises(ValueError, match="between 1 and 240"):
        env._launch()
