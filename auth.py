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
    if len(username) > 30:
        return False, "Username must be at most 30 characters."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if role not in {"patient", "clinician"}:
        return False, "Invalid role."
    if database.get_user_by_username(username):
        return False, "Username already exists (usernames are case-insensitive)."
    database.create_user(username, hash_password(password), role)
    return True, "Account created. Please log in."


def login(username: str, password: str) -> dict | None:
    clean_username = (username or "").strip()
    if not clean_username or not password:
        return None
    user = database.get_user_by_username(clean_username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


def change_password(user_id: int, old_password: str, new_password: str, confirm_password: str) -> tuple[bool, str]:
    user = database.get_user_by_id(user_id)
    if not user:
        return False, "User not found."
    if not verify_password(old_password, user["password_hash"]):
        return False, "Current password does not match."
    if len(new_password) < 8:
        return False, "New password must be at least 8 characters long."
    if new_password != confirm_password:
        return False, "New password and confirmation do not match."
    if old_password == new_password:
        return False, "New password must be different from current password."
    database.update_password(user_id, hash_password(new_password))
    return True, "Password updated successfully."


def delete_account(user_id: int, password: str) -> tuple[bool, str]:
    user = database.get_user_by_id(user_id)
    if not user:
        return False, "User not found."
    if not verify_password(password, user["password_hash"]):
        return False, "Incorrect password. Account deletion aborted."
    database.delete_user_account(user_id)
    return True, "Account and all associated records permanently deleted."


def update_display_name(user_id: int, display_name: str) -> tuple[bool, str]:
    clean_name = (display_name or "").strip()
    if not clean_name:
        return False, "Display name cannot be empty."
    if len(clean_name) > 60:
        return False, "Display name must be at most 60 characters."
    database.update_profile(user_id, clean_name)
    return True, "Profile display name updated successfully."