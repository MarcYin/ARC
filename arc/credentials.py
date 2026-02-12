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
import time
from pathlib import Path

CONFIG_DIR = Path.home() / ".arc"
CONFIG_FILE = CONFIG_DIR / "config.json"

ALL_SOURCES = ["aws", "cdse", "planetary", "gee"]

# Default config template
DEFAULT_CONFIG = {
    "data_source_preference": ["aws", "cdse", "planetary", "gee"],
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
    os.chmod(CONFIG_DIR, stat.S_IRWXU)


def save_config(config: dict):
    """Save credentials to ~/.arc/config.json with restricted permissions."""
    _ensure_config_dir()
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
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


def get_data_source_preference() -> list:
    """Return the ordered list of preferred data sources from config."""
    config = load_config()
    return config.get("data_source_preference", DEFAULT_CONFIG["data_source_preference"])


def get_available_sources() -> list:
    """
    Return list of data sources that have working credentials/dependencies.

    Checks each source without importing heavy modules where possible.
    """
    available = []

    # AWS: always available (no auth)
    available.append("aws")

    # Planetary Computer: available if package installed
    try:
        import planetary_computer  # noqa: F401
        available.append("planetary")
    except ImportError:
        pass

    # CDSE: available if credentials configured
    creds = get_cdse_credentials()
    if (creds["s3_access_key"] and creds["s3_secret_key"]) or \
       (creds["username"] and creds["password"]):
        available.append("cdse")

    # GEE: available if ee.Initialize() works
    try:
        import ee
        ee.Initialize()
        available.append("gee")
    except Exception:
        pass

    return available


def probe_download_speed(geojson_path: str, sources: list = None,
                         timeout: float = 30.0) -> dict:
    """
    Probe download speed of each data source by fetching a single band.

    Args:
        geojson_path: Path to GeoJSON field boundary.
        sources: List of sources to probe (default: all available).
        timeout: Max seconds per probe before giving up.

    Returns:
        dict: {source_name: seconds} for successful probes. Failed probes not included.
    """
    if sources is None:
        sources = get_available_sources()

    # Load geometry once for STAC searches
    with open(geojson_path) as f:
        features = json.load(f)["features"]
    from shapely.geometry import shape
    import shapely
    geom = shape(features[0]["geometry"])
    geojson_dict = json.loads(shapely.to_geojson(geom))

    results = {}

    print("Probing data source speeds...")
    print(f"{'Source':<20} {'Time (s)':<12} {'Status'}")
    print("-" * 50)

    for source in sources:
        try:
            t0 = time.time()
            elapsed = _probe_single_source(source, geojson_dict, geojson_path, timeout)
            results[source] = elapsed
            print(f"{source:<20} {elapsed:<12.1f} OK")
        except Exception as e:
            print(f"{source:<20} {'--':<12} FAILED ({e})")

    if results:
        fastest = min(results, key=results.get)
        print(f"\nFastest: {fastest} ({results[fastest]:.1f}s)")

    return results


def _probe_single_source(source: str, geojson_dict: dict,
                         geojson_path: str, timeout: float) -> float:
    """Probe a single source by reading one band. Returns elapsed seconds."""
    from osgeo import gdal
    import pystac_client
    import numpy as np

    gdal.PushErrorHandler('CPLQuietErrorHandler')

    if source == "aws":
        gdal.SetConfigOption("AWS_NO_SIGN_REQUEST", "YES")
        gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

        client = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
        search = client.search(
            collections=["sentinel-2-l2a"],
            intersects=geojson_dict,
            max_items=1,
        )
        items = list(search.items())
        if not items:
            raise RuntimeError("No items found")

        href = items[0].assets["blue"].href
        vsi_path = f"/vsicurl/{href}"

    elif source == "cdse":
        from arc.s2_cdse_reader import _configure_gdal_for_cdse, _s3_path_from_href

        vsi_prefix = _configure_gdal_for_cdse()
        client = pystac_client.Client.open("https://stac.dataspace.copernicus.eu/v1")
        search = client.search(
            collections=["sentinel-2-l2a"],
            intersects=geojson_dict,
            max_items=1,
        )
        items = list(search.items())
        if not items:
            raise RuntimeError("No items found")

        href = items[0].assets["B02_10m"].href
        if vsi_prefix == "/vsis3":
            vsi_path = f"/vsis3{_s3_path_from_href(href)}"
        else:
            from arc.s2_cdse_reader import _https_url_from_href
            vsi_path = f"/vsicurl/{_https_url_from_href(href)}"

    elif source == "planetary":
        import planetary_computer

        gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
        client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1"
        )
        search = client.search(
            collections=["sentinel-2-l2a"],
            intersects=geojson_dict,
            max_items=1,
        )
        items = list(search.items())
        if not items:
            raise RuntimeError("No items found")

        href = items[0].assets["B02"].href
        signed_href = planetary_computer.sign_url(href)
        vsi_path = f"/vsicurl/{signed_href}"

    elif source == "gee":
        # GEE uses a different mechanism (ee.Image.getDownloadURL), so we
        # time a minimal download via the GEE API instead of GDAL
        import ee
        t0 = time.time()
        ee.Initialize()
        # Get a small image to test connection speed
        img = ee.Image("COPERNICUS/S2_SR_HARMONIZED").first()
        img.getInfo()  # forces a round-trip
        return time.time() - t0

    else:
        raise ValueError(f"Unknown source: {source}")

    # Time the GDAL warp (read + crop one band)
    t0 = time.time()
    ds = gdal.Warp(
        '', vsi_path,
        format='MEM',
        cutlineDSName=geojson_path,
        cropToCutline=True,
        xRes=10, yRes=10,
        outputType=gdal.GDT_Int16,
    )
    if ds is None:
        raise IOError(f"GDAL failed to read from {source}")
    _ = ds.ReadAsArray()
    ds = None
    return time.time() - t0


def select_data_source(geojson_path: str = None, probe: bool = False) -> str:
    """
    Select the best data source.

    If probe=True and geojson_path is provided, probes download speed and
    picks the fastest. Otherwise, returns the first available source from
    the user's preference list.

    Args:
        geojson_path: Path to GeoJSON (required for speed probing).
        probe: If True, probe download speeds and pick fastest.

    Returns:
        str: Data source name ('aws', 'cdse', 'planetary', or 'gee').
    """
    available = get_available_sources()
    if not available:
        raise RuntimeError(
            "No data sources available. Run the setup_credentials notebook "
            "to configure at least one data source."
        )

    if probe and geojson_path:
        speeds = probe_download_speed(geojson_path, sources=available)
        if speeds:
            fastest = min(speeds, key=speeds.get)
            return fastest

    # Fall through preference list
    preference = get_data_source_preference()
    for source in preference:
        if source in available:
            return source

    # If nothing in preference list is available, use first available
    return available[0]


def print_config_status():
    """Print which credentials are configured."""
    creds = get_cdse_credentials()
    preference = get_data_source_preference()
    available = get_available_sources()

    print("ARC Data Source Status")
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
    if "planetary" in available:
        print("  Planetary Computer: package installed")
    else:
        print("  Planetary Computer: not installed (pip install planetary-computer)")

    # GEE
    if "gee" in available:
        print("  GEE:               authenticated")
    else:
        print("  GEE:               not authenticated (run: earthengine authenticate)")

    print(f"\n  Preference order:  {preference}")
    print(f"  Available:         {available}")
    print(f"  Config file:       {CONFIG_FILE}")
