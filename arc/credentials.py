"""
Credential management for ARC data sources.

Credentials are stored in ~/.arc/config.json with restricted file permissions.
Environment variables take priority over the config file.

Supported data sources and their credentials:
    cdse:       CDSE_S3_ACCESS_KEY, CDSE_S3_SECRET_KEY (or CDSE_USERNAME, CDSE_PASSWORD)
    gee:        Managed by earthengine-api (ee.Authenticate)
    aws:        No credentials needed
    planetary:  No credentials needed (handled by planetary-computer package)
"""

import os
import json
import stat
from pathlib import Path

CONFIG_DIR = Path.home() / ".arc"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Default config template
DEFAULT_CONFIG = {
    "cdse": {
        "s3_access_key": "",
        "s3_secret_key": "",
        "username": "",
        "password": "",
    },
    "gee": {
        "note": "GEE authentication is managed by 'earthengine authenticate'. No keys needed here."
    },
    "aws": {
        "note": "AWS Earth Search requires no authentication."
    },
    "planetary": {
        "note": "Planetary Computer auth is handled by the planetary-computer package."
    },
}


def _ensure_config_dir():
    """Create ~/.arc/ directory with restricted permissions."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Restrict to owner only (rwx------)
    os.chmod(CONFIG_DIR, stat.S_IRWXU)


def save_config(config: dict):
    """Save credentials to ~/.arc/config.json with restricted permissions."""
    _ensure_config_dir()
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    # Restrict file to owner read/write only (rw-------)
    os.chmod(CONFIG_FILE, stat.S_IRUSR | stat.S_IWUSR)


def load_config() -> dict:
    """Load credentials from ~/.arc/config.json, or return defaults."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()


def create_default_config():
    """Create a default config file if one doesn't exist."""
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        print(f"Created default config at {CONFIG_FILE}")
    else:
        print(f"Config already exists at {CONFIG_FILE}")


def get_cdse_credentials() -> dict:
    """
    Get CDSE credentials. Environment variables take priority over config file.

    Returns:
        dict with keys: 's3_access_key', 's3_secret_key', 'username', 'password'
    """
    config = load_config()
    cdse = config.get("cdse", {})

    return {
        "s3_access_key": os.environ.get("CDSE_S3_ACCESS_KEY", "") or cdse.get("s3_access_key", ""),
        "s3_secret_key": os.environ.get("CDSE_S3_SECRET_KEY", "") or cdse.get("s3_secret_key", ""),
        "username": os.environ.get("CDSE_USERNAME", "") or cdse.get("username", ""),
        "password": os.environ.get("CDSE_PASSWORD", "") or cdse.get("password", ""),
    }


def print_config_status():
    """Print which credentials are configured."""
    creds = get_cdse_credentials()

    print("ARC Credential Status")
    print("=" * 50)

    # CDSE
    if creds["s3_access_key"] and creds["s3_secret_key"]:
        src = "env" if os.environ.get("CDSE_S3_ACCESS_KEY") else "config"
        print(f"  CDSE (S3 keys):    configured ({src})")
    elif creds["username"] and creds["password"]:
        src = "env" if os.environ.get("CDSE_USERNAME") else "config"
        print(f"  CDSE (user/pass):  configured ({src})")
    else:
        print("  CDSE:              not configured")

    # AWS
    print("  AWS Earth Search:  no credentials needed")

    # Planetary Computer
    try:
        import planetary_computer
        print("  Planetary Computer: package installed")
    except ImportError:
        print("  Planetary Computer: package not installed (pip install planetary-computer)")

    # GEE
    try:
        import ee
        ee.Initialize()
        print("  GEE:               authenticated")
    except Exception:
        print("  GEE:               not authenticated (run: earthengine authenticate)")

    print(f"\n  Config file: {CONFIG_FILE}")
    print(f"  Config exists: {CONFIG_FILE.exists()}")
