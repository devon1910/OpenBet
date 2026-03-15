"""Run this once to generate a bcrypt hash for your admin password.

Usage:
    python scripts/generate_password_hash.py

Then copy the output into your .env file:
    ADMIN_PASSWORD_HASH=<hash>
"""

import sys
sys.path.insert(0, ".")
from src.api.auth import hash_password

password = input("Enter admin password: ").strip()
if not password:
    print("Password cannot be empty.")
    sys.exit(1)

print("\nAdd this to your .env file:")
print(f'ADMIN_PASSWORD_HASH="{hash_password(password)}"')
