import pytest

from utils.envs import browser_env

SESSION = {
    "id": "bu-session-123",
    "cdpUrl": "https://bu-session-123.cdp.browser-use.com",
    "liveUrl": "https://live.browser-use.com?wss=bu-session-123",
}


class _FakePage:
    def __init__(self):
        self.viewport = None

    def set_viewport_size(self, size):
        self.viewport = size


class _FakeContext:
    def __init__(self):
        self.pages = [_FakePage()]


class _FakeBrowser:
    def __init__(self):
        self.contexts = [_FakeContext()]
        self.closed = False

    def close(self):
        self.closed = True


class _FakePlaywright:
    def __init__(self):
        self.browser = _FakeBrowser()
        self.cdp_url = None
        self.stopped = False
        self.chromium = self

    def connect_over_cdp(self, cdp_url):
        self.cdp_url = cdp_url
        return self.browser

    def stop(self):
        self.stopped = True


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


@pytest.fixture
def api(monkeypatch):
    """Record every Browser Use REST call; answer each one with SESSION."""
    import requests

    calls = []

    def request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return _FakeResponse(SESSION)

    monkeypatch.setattr(requests, "request", request)
    return calls


@pytest.fixture
def playwright(monkeypatch):
    fake = _FakePlaywright()
    monkeypatch.setattr(browser_env, "_start_playwright", lambda: fake)
    return fake


def test_launch_provisions_a_session_and_connects_to_its_cdp_url(api, playwright):
    env = browser_env.BrowserUseEnv(
        api_key="key+with/slashes",
        proxy_country_code="de",
        session_timeout_minutes=30,
        viewport_width=1440,
        viewport_height=900,
    )
    env._launch()

    assert len(api) == 1
    create = api[0]
    assert create["method"] == "POST"
    assert create["url"] == "https://api.browser-use.com/api/v4/browsers"
    # The key travels in a header, never in a URL that could land in an error log.
    assert create["headers"] == {"X-Browser-Use-API-Key": "key+with/slashes"}
    assert create["json"] == {
        "proxyCountryCode": "de",
        "timeout": 30,
        "browserScreenWidth": 1440,
        "browserScreenHeight": 900,
        # Without this Browser Use ignores set_viewport_size entirely.
        "allowResizing": True,
    }

    assert playwright.cdp_url == SESSION["cdpUrl"]
    assert env.page is env.context.pages[0]


def test_close_stops_the_cloud_session(api, playwright):
    env = browser_env.BrowserUseEnv(api_key="test")
    env._launch()

    env.close()

    stop = api[-1]
    assert stop["method"] == "PATCH"
    assert stop["url"] == "https://api.browser-use.com/api/v4/browsers/bu-session-123"
    assert stop["json"] == {"action": "stop"}
    assert playwright.browser.closed and playwright.stopped

    # The session is forgotten, so a second close() cannot double-stop it.
    env.close()
    assert api[-1] is stop


def test_close_survives_a_failing_stop_call(api, playwright, monkeypatch):
    import requests

    env = browser_env.BrowserUseEnv(api_key="test")
    env._launch()

    def boom(*a, **k):
        raise RuntimeError("browser use is down")

    monkeypatch.setattr(requests, "request", boom)
    env.close()

    assert playwright.browser.closed and env.bu_session is None


def test_close_before_launch_makes_no_api_call(api):
    browser_env.BrowserUseEnv(api_key="test").close()
    assert api == []


def test_a_failed_attach_still_stops_the_session(api, playwright):
    def boom(cdp_url):
        raise RuntimeError("cdp refused")

    playwright.connect_over_cdp = boom
    env = browser_env.BrowserUseEnv(api_key="test")

    with pytest.raises(RuntimeError, match="cdp refused"):
        env._launch()

    assert [c["method"] for c in api] == ["POST", "PATCH"]
    assert playwright.stopped and env.bu_session is None


def test_get_info_exposes_the_session(api, playwright):
    env = browser_env.BrowserUseEnv(api_key="test")
    assert env._get_info() == {"browser_provider": "browser_use"}

    env._launch()
    assert env._get_info() == {
        "browser_provider": "browser_use",
        "bu_session_id": SESSION["id"],
        "live_view_url": SESSION["liveUrl"],
    }


def test_every_observation_refits_the_active_page(api, playwright, monkeypatch):
    """New tabs open at Browser Use's default size, so each obs must re-fit first."""
    monkeypatch.setattr(browser_env.BrowserEnv, "_get_obs", lambda self: {})

    env = browser_env.BrowserUseEnv(api_key="test", viewport_width=1280, viewport_height=720)
    env._launch()
    env._get_obs()
    assert env.page.viewport == {"width": 1280, "height": 720}

    # Following the agent onto a fresh tab re-fits that tab, not the old one.
    env.page = _FakePage()
    env._get_obs()
    assert env.page.viewport == {"width": 1280, "height": 720}


def test_explicit_empty_api_key_does_not_fall_back_to_the_environment(api, monkeypatch):
    monkeypatch.setenv("BROWSER_USE_API_KEY", "key-from-env")
    assert browser_env.BrowserUseEnv().api_key == "key-from-env"

    env = browser_env.BrowserUseEnv(api_key="")
    with pytest.raises(ValueError, match="BROWSER_USE_API_KEY required"):
        env._launch()
    assert api == []


@pytest.mark.parametrize("timeout", [0, 241])
def test_invalid_timeout_is_rejected_before_provisioning(api, timeout):
    env = browser_env.BrowserUseEnv(api_key="test", session_timeout_minutes=timeout)

    with pytest.raises(ValueError, match="between 1 and 240"):
        env._launch()
    assert api == []
