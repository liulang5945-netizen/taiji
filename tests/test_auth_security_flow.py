"""Focused regression tests for auth/bootstrap and privileged update routes."""

from __future__ import annotations

import importlib
import io
import os
import shutil
from pathlib import Path
from uuid import uuid4
import zipfile

import pytest


def _build_ui_zip() -> io.BytesIO:
    """Create a minimal in-memory frontend bundle zip."""
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("index.html", "<!doctype html><title>Taiji</title>")
    payload.seek(0)
    return payload


@pytest.fixture
def isolated_client(monkeypatch):
    """Create a TestClient with isolated settings and auth state."""
    base_dir = Path("test_artifacts") / f"auth_security_{uuid4().hex}"
    os.makedirs(base_dir, exist_ok=True)
    monkeypatch.setenv("TAIJI_BASE_DIR", str(base_dir.resolve()))

    from taiji.core.security import AuthManager

    AuthManager._instance = None

    import api.app as app_module

    app_module = importlib.reload(app_module)

    from fastapi.testclient import TestClient

    with TestClient(app_module.create_app(startup_tasks=False)) as client:
        yield client

    AuthManager._instance = None
    shutil.rmtree(base_dir, ignore_errors=True)


def test_runtime_bootstrap_stays_public_after_auth_enable(isolated_client):
    enable_resp = isolated_client.post(
        "/api/auth/enable",
        json={"username": "admin", "password": "secret"},
    )
    assert enable_resp.status_code == 200

    bootstrap_resp = isolated_client.get("/api/runtime/bootstrap")
    assert bootstrap_resp.status_code == 200
    payload = bootstrap_resp.json()
    assert payload["auth_enabled"] is True
    assert payload["need_login"] is True


def test_auth_enable_requires_token_after_bootstrap(isolated_client):
    first_enable = isolated_client.post(
        "/api/auth/enable",
        json={"username": "admin", "password": "secret"},
    )
    assert first_enable.status_code == 200

    second_enable = isolated_client.post(
        "/api/auth/enable",
        json={"username": "attacker", "password": "pwnd"},
    )
    assert second_enable.status_code == 401

    login_resp = isolated_client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "secret"},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["token"]

    admin_enable = isolated_client.post(
        "/api/auth/enable",
        json={"username": "owner", "password": "rotated"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert admin_enable.status_code == 200


def test_upload_ui_requires_auth_enabled(isolated_client):
    response = isolated_client.post(
        "/api/system/upload_ui",
        files={"file": ("frontend.zip", _build_ui_zip(), "application/zip")},
    )
    assert response.status_code == 403


def test_upload_ui_requires_admin_token_when_auth_enabled(isolated_client):
    enable_resp = isolated_client.post(
        "/api/auth/enable",
        json={"username": "admin", "password": "secret"},
    )
    assert enable_resp.status_code == 200

    unauthenticated_resp = isolated_client.post(
        "/api/system/upload_ui",
        files={"file": ("frontend.zip", _build_ui_zip(), "application/zip")},
    )
    assert unauthenticated_resp.status_code == 401

    login_resp = isolated_client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "secret"},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["token"]

    authenticated_resp = isolated_client.post(
        "/api/system/upload_ui",
        files={"file": ("frontend.zip", _build_ui_zip(), "application/zip")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert authenticated_resp.status_code not in (401, 403)
