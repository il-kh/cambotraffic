import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def load_env_local() -> None:
    """Load .env.local if it exists; warn and continue if it is missing."""
    env_file = PROJECT_ROOT / ".env.local"
    if not env_file.exists():
        print(
            f"Warning: '{env_file}' not found.\n"
            "Copy '.env.local.example' to '.env.local' and fill in your paths."
        )
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        print("python-dotenv is not installed. Run: pip install python-dotenv")
        sys.exit(1)

    load_dotenv(env_file, override=True)


def require(var: str) -> Path:
    value = os.environ.get(var)
    if not value:
        print(
            f"Error: '{var}' is not set.\n"
            "Add it to your .env.local file (see .env.local.example)."
        )
        sys.exit(1)
    return Path(value)
