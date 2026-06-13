"""Auth service — unified authentication interface.

Wraps AuthManager so routes don't directly depend on core.security.
Used by routes_auth.py and runtime_service.py.
"""
from taiji.core.security import AuthManager


def _auth() -> AuthManager:
    """Get a fresh AuthManager instance."""
    return AuthManager()


def get_status() -> dict:
    """Return auth status without requiring a token."""
    return _auth().get_status()


def get_authenticated_status(authorization_header: str = "") -> dict:
    """Return auth status including token validation.

    Used by runtime_service to populate the auth field.
    """
    auth = _auth()
    status = auth.get_status()
    token = ""
    if authorization_header.startswith("Bearer "):
        token = authorization_header[7:]

    authenticated = not status.get("enabled", False)
    token_valid = False
    if token:
        token_valid = bool(auth.verify_token(token))
        authenticated = token_valid

    return {
        "enabled": bool(status.get("enabled", False)),
        "authenticated": authenticated,
        "token_valid": token_valid,
        "username": status.get("username") or "",
        "has_password": bool(status.get("has_password", False)),
    }


def login(username: str, password: str) -> str | None:
    """Attempt login. Returns JWT token on success, None on failure."""
    return _auth().login(username, password)


def change_password(old_password: str, new_password: str) -> bool:
    """Change password. Returns True on success."""
    auth = _auth()
    if not auth.verify_password(old_password):
        return False
    auth.set_password(new_password)
    return True


def enable_auth(username: str, password: str) -> None:
    """Enable authentication with the given credentials."""
    _auth().enable_auth(username, password)


def disable_auth() -> None:
    """Disable authentication."""
    _auth().disable_auth()


def verify_token(token: str) -> bool:
    """Verify a JWT token."""
    return bool(_auth().verify_token(token))


def refresh_token(token: str) -> str | None:
    """Refresh a JWT token. Returns new token or None."""
    return _auth().jwt.refresh_token(token)


def get_audit_logs(limit: int = 50, days: int = 7) -> list:
    """Get recent audit log events."""
    return _auth().audit.get_recent_events(limit=limit, days=days)
