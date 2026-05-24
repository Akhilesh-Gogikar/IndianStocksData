from __future__ import annotations

import os

import pytest
import requests
from playwright.sync_api import expect, sync_playwright

LIVE_URL = os.getenv("CEREBRAL_LIVE_URL", "http://157.173.109.247/")
API_HEALTH = f"{LIVE_URL.rstrip('/')}/api/health"

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_SITE_TESTS") != "1",
    reason="set RUN_LIVE_SITE_TESTS=1 to run live deployment checks",
)


def test_api():
    response = requests.get(API_HEALTH, timeout=10)

    assert response.status_code == 200
    assert response.text.strip()


def test_frontend():
    with sync_playwright() as pl:
        browser = pl.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        errors = []
        page.on("pageerror", lambda err: errors.append(err))

        page.goto(LIVE_URL, wait_until="load")
        page.wait_for_selector("#landingView", state="visible", timeout=10000)

        expect(page.locator("#landingView")).to_be_visible()
        auth_class = page.locator("#authView").get_attribute("class") or ""
        assert "is-hidden" in auth_class

        page.click("button:has-text('Sign in')")
        expect(page.locator("#loginCard")).to_be_visible(timeout=5000)
        landing_class = page.locator("#landingView").get_attribute("class") or ""
        assert "is-hidden" in landing_class
        landing_display = page.locator("#landingView").evaluate(
            "el => window.getComputedStyle(el).display"
        )
        assert landing_display == "none"

        page.fill("#loginEmail", "test@example.com")
        page.fill("#loginPassword", "wrongpassword")
        page.click("#loginForm button[type='submit']")
        expect(page.locator("#loginErrorMsg")).not_to_be_empty(timeout=10000)

        assert errors == []

        browser.close()
