"""Password hashing and authentication helpers.

Uses PBKDF2-HMAC-SHA256 with a random salt. This is suitable for a research
prototype; production systems should use a memory-hard algorithm such as Argon2
and enforce rate limiting, MFA, and TLS.
"""

import hashlib
import hmac
import os

import database


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    iterations = 260_000
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations_str, salt_hex, hash_hex = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


def signup(username: str, password: str, role: str) -> tuple[bool, str]:
    username = username.strip()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if role not in {"patient", "clinician"}:
        return False, "Invalid role."
    if database.get_user_by_username(username):
        return False, "Username already exists."
    database.create_user(username, hash_password(password), role)
    return True, "Account created. Please log in."


def login(username: str, password: str) -> dict | None:
    user = database.get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user