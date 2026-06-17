import unittest
from datetime import datetime, timezone

from fastapi import HTTPException
from fastapi.testclient import TestClient
from jose import jwt

from backend.config import settings
from backend.main import DEFAULT_USERS, app, create_token, verify_token
from backend.models.schemas import LoginRequest, LoginResponse, UserInfo


class AuthSessionTests(unittest.TestCase):
    def test_default_poc_passwords_match_current_login_hint(self):
        defaults = {item["username"]: item["password"] for item in DEFAULT_USERS}

        self.assertEqual(defaults["admin"], "Admin123456@")
        self.assertEqual(defaults["executive"], "Exec123456@")
        self.assertEqual(defaults["procurement"], "Proc123456@")

    def test_login_schema_accepts_remember_me_without_exposing_access_token(self):
        req = LoginRequest(username="executive", password="not-used", remember_me=True)
        self.assertTrue(req.remember_me)

        fields = LoginResponse.model_fields
        self.assertNotIn("access_token", fields)
        self.assertIn("session_expires_in", fields)
        self.assertIn("remember_me", UserInfo.model_fields)

    def test_cookie_session_token_can_authenticate_user(self):
        token = create_token("executive", "executive", remember_me=False)

        user = verify_token(credentials=None, session_token=token)

        self.assertEqual(user["username"], "executive")
        self.assertEqual(user["role"], "executive")
        self.assertFalse(user["remember_me"])

    def test_remember_me_token_uses_fourteen_day_lifetime(self):
        token = create_token("executive", "executive", remember_me=True)
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )

        expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        remaining = expires_at - datetime.now(timezone.utc)

        self.assertGreaterEqual(remaining.days, 13)
        self.assertLessEqual(remaining.days, 14)
        self.assertTrue(payload["remember_me"])

    def test_missing_session_token_is_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            verify_token(credentials=None, session_token=None)

        self.assertEqual(ctx.exception.status_code, 401)

    def test_public_session_probe_does_not_401_when_logged_out(self):
        client = TestClient(app)

        res = client.get("/api/auth/session")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"authenticated": False})

    def test_public_session_probe_restores_cookie_session(self):
        client = TestClient(app)
        token = create_token("executive", "executive", remember_me=True)

        res = client.get(
            "/api/auth/session",
            headers={"cookie": f"{settings.session_cookie_name}={token}"},
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["authenticated"], True)
        self.assertEqual(res.json()["username"], "executive")
        self.assertTrue(res.json()["remember_me"])


if __name__ == "__main__":
    unittest.main()
