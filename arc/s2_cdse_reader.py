"""
Sentinel-2 L2A data retrieval from the Copernicus Data Space Ecosystem (CDSE) STAC API.

This module provides the same interface as s2_data_reader.py (GEE-based) but uses the
CDSE STAC catalogue and S3 storage for data access.

Authentication:
    Option 1 (preferred): Set CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY environment variables.
        Generate these at https://eodata.dataspace.copernicus.eu
    Option 2 (fallback): Set CDSE_USERNAME and CDSE_PASSWORD environment variables.
        These are your Copernicus Data Space Ecosystem login credentials.
"""

import os
import json
import datetime
import numpy as np
from osgeo import gdal
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from shapely.geometry import shape
import shapely
import mgrs
import requests

import pystac_client

gdal.PushErrorHandler('CPLQuietErrorHandler')

# CDSE endpoints
CDSE_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
CDSE_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_COLLECTION = "sentinel-2-l2a"

# Band asset keys in the order expected by ARC: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
ARC_BAND_ASSETS = [
    'B02_10m', 'B03_10m', 'B04_10m',
    'B05_20m', 'B06_20m', 'B07_20m',
    'B08_10m',
    'B8A_20m', 'B11_20m', 'B12_20m',
]
SCL_ASSET_KEY = 'SCL_20m'

# SCL classes to mask: no data, saturated, cloud shadow, cloud med/high, cirrus, snow
SCL_MASK_VALUES = frozenset({0, 1, 3, 8, 9, 10, 11})

QUANTIFICATION_VALUE = 10000.0
BASELINE_OFFSET = -1000


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def _get_cdse_access_token(username: str, password: str) -> str:
    """Obtain an OAuth2 access token from the CDSE identity service."""
    data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    resp = requests.post(CDSE_TOKEN_URL, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def _configure_gdal_for_cdse():
    """
    Configure GDAL to read from CDSE S3 storage.

    Tries S3 keys first (CDSE_S3_ACCESS_KEY/SECRET_KEY), then falls back to
    username/password token authentication via /vsicurl/.

    Returns:
        str: The GDAL virtual filesystem prefix to use ('/vsis3' or '/vsicurl').
    """
    s3_key = os.environ.get("CDSE_S3_ACCESS_KEY", "")
    s3_secret = os.environ.get("CDSE_S3_SECRET_KEY", "")

    if s3_key and s3_secret:
        gdal.SetConfigOption("AWS_S3_ENDPOINT", CDSE_S3_ENDPOINT)
        gdal.SetConfigOption("AWS_ACCESS_KEY_ID", s3_key)
        gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", s3_secret)
        gdal.SetConfigOption("AWS_VIRTUAL_HOSTING", "FALSE")
        gdal.SetConfigOption("AWS_HTTPS", "YES")
        vsi_prefix = "/vsis3"
    else:
        username = os.environ.get("CDSE_USERNAME", "")
        password = os.environ.get("CDSE_PASSWORD", "")
        if not (username and password):
            raise EnvironmentError(
                "CDSE credentials not found. Set either:\n"
                "  CDSE_S3_ACCESS_KEY + CDSE_S3_SECRET_KEY  (preferred), or\n"
                "  CDSE_USERNAME + CDSE_PASSWORD\n"
                "Generate S3 keys at https://eodata.dataspace.copernicus.eu"
            )
        token = _get_cdse_access_token(username, password)
        gdal.SetConfigOption("GDAL_HTTP_HEADERS", f"Authorization: Bearer {token}")
        vsi_prefix = "/vsicurl"

    # Optimizations to reduce network overhead
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.jp2,.xml")
    gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", "5")
    gdal.SetConfigOption("GDAL_HTTP_RETRY_DELAY", "5")
    gdal.SetConfigOption("GDAL_CACHEMAX", "256")
    gdal.SetConfigOption("CPL_VSIL_CURL_CACHE_SIZE", "67108864")  # 64 MB curl cache

    return vsi_prefix


# ---------------------------------------------------------------------------
# GeoJSON loading (mirrors s2_data_reader.load_geojson)
# ---------------------------------------------------------------------------

def _load_geojson(file_path: str):
    """Load a GeoJSON file and return the first feature's geometry as a shapely object."""
    with open(file_path) as f:
        features = json.load(f)["features"]
    geom = shape(features[0]["geometry"])
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


# ---------------------------------------------------------------------------
# STAC search
# ---------------------------------------------------------------------------

def _search_s2_collection(geojson_geometry: dict, start_date: str, end_date: str,
                          max_cloud_cover: int = 80) -> list:
    """
    Search the CDSE STAC catalogue for Sentinel-2 L2A items intersecting the
    given geometry within the date range.
    """
    client = pystac_client.Client.open(CDSE_STAC_URL)
    search = client.search(
        collections=[CDSE_COLLECTION],
        intersects=geojson_geometry,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lte": max_cloud_cover}},
        max_items=500,
    )
    items = list(search.items())
    items.sort(key=lambda x: x.datetime)
    return items


def _filter_to_single_mgrs_tile(items: list, centroid_lon: float,
                                 centroid_lat: float) -> list:
    """Filter STAC items to the MGRS tile containing the field centroid."""
    m = mgrs.MGRS()
    target_tile = m.toMGRS(centroid_lat, centroid_lon)[:5]
    target_code = f"MGRS-{target_tile}"
    return [item for item in items
            if item.properties.get("grid:code", "") == target_code]


# ---------------------------------------------------------------------------
# Angle extraction
# ---------------------------------------------------------------------------

def _extract_angles_from_item(item) -> Tuple[float, float, float]:
    """
    Extract (SZA, VZA, RAA) in degrees from STAC item properties.

    CDSE properties:
        view:sun_elevation  → SZA = 90 - sun_elevation
        view:sun_azimuth    → SAA
        view:incidence_angle → VZA (scene-level mean across bands/detectors)
        view:azimuth        → VAA
    """
    props = item.properties
    sun_elevation = props.get("view:sun_elevation", 0.0)
    sun_azimuth = props.get("view:sun_azimuth", 0.0)
    view_zenith = props.get("view:incidence_angle", 0.0)
    view_azimuth = props.get("view:azimuth", 0.0)

    sza = 90.0 - sun_elevation
    raa = (view_azimuth - sun_azimuth) % 360.0
    return sza, view_zenith, raa


def _extract_doy_from_item(item) -> int:
    """Extract day of year from a STAC item's datetime."""
    return item.datetime.timetuple().tm_yday


# ---------------------------------------------------------------------------
# Band reading with GDAL
# ---------------------------------------------------------------------------

def _s3_path_from_href(href: str) -> str:
    """
    Convert an S3 href like 's3://eodata/Sentinel-2/...' to the bucket-relative
    path '/eodata/Sentinel-2/...'.
    """
    if href.startswith("s3://"):
        return "/" + href[5:]
    if href.startswith("/eodata"):
        return href
    return href


def _https_url_from_href(href: str) -> str:
    """
    Convert an S3 href to an HTTPS URL for /vsicurl/ access.
    """
    path = _s3_path_from_href(href)
    return f"https://{CDSE_S3_ENDPOINT}{path}"


def _read_and_crop_band(vsi_prefix: str, href: str, geojson_cutline: str,
                        target_resolution: int = 10,
                        resample_alg: str = 'bilinear') -> np.ndarray:
    """
    Read a single band from CDSE, crop to field boundary, resample to target resolution.
    Returns integer (Int16) array to minimize transfer.
    """
    if vsi_prefix == "/vsis3":
        vsi_path = f"/vsis3{_s3_path_from_href(href)}"
    else:
        vsi_path = f"/vsicurl/{_https_url_from_href(href)}"

    resample_map = {
        'bilinear': gdal.GRA_Bilinear,
        'nearest': gdal.GRA_NearestNeighbour,
    }

    ds = gdal.Warp(
        '', vsi_path,
        format='MEM',
        cutlineDSName=geojson_cutline,

        cropToCutline=True,
        xRes=target_resolution,
        yRes=target_resolution,
        resampleAlg=resample_map.get(resample_alg, gdal.GRA_Bilinear),
        dstNodata=0,
        outputType=gdal.GDT_Int16,
    )
    if ds is None:
        raise IOError(f"Failed to read band from {vsi_path}")

    data = ds.ReadAsArray()
    gt = ds.GetGeoTransform()
    crs = ds.GetProjection()
    ds = None
    return data, gt, crs


def _read_and_crop_all_bands(item, geojson_cutline: str, vsi_prefix: str,
                             target_resolution: int = 10):
    """
    Read all 10 spectral bands + SCL for a single STAC item.

    Returns:
        band_data: (10, H, W) int16 array of DN values
        scl_data: (H, W) int16 array of SCL classification
        geotransform: 6-tuple
        crs: WKT string
    """
    bands = []
    geotransform = None
    crs = None

    for asset_key in ARC_BAND_ASSETS:
        asset = item.assets.get(asset_key)
        if asset is None:
            raise KeyError(f"Asset '{asset_key}' not found in item {item.id}")
        data, gt, proj = _read_and_crop_band(
            vsi_prefix, asset.href, geojson_cutline,
            target_resolution=target_resolution,
            resample_alg='bilinear'
        )
        bands.append(data)
        if geotransform is None:
            geotransform = gt
            crs = proj

    # Read SCL with nearest-neighbour (classification data)
    scl_asset = item.assets.get(SCL_ASSET_KEY)
    if scl_asset is None:
        raise KeyError(f"Asset '{SCL_ASSET_KEY}' not found in item {item.id}")
    scl_data, _, _ = _read_and_crop_band(
        vsi_prefix, scl_asset.href, geojson_cutline,
        target_resolution=target_resolution,
        resample_alg='nearest'
    )

    band_data = np.stack(bands, axis=0)
    return band_data, scl_data, geotransform, crs


# ---------------------------------------------------------------------------
# Cloud masking and reflectance conversion
# ---------------------------------------------------------------------------

def _apply_scl_cloud_mask(reflectance: np.ndarray, scl: np.ndarray) -> np.ndarray:
    """
    Mask pixels using the Scene Classification Layer (SCL).

    Masked classes: 0=NoData, 1=Saturated, 3=CloudShadow,
    8=CloudMedium, 9=CloudHigh, 10=Cirrus, 11=Snow
    """
    mask = np.isin(scl, list(SCL_MASK_VALUES))
    reflectance[:, mask] = np.nan
    return reflectance


def _dn_to_reflectance(dn: np.ndarray, processing_baseline: str) -> np.ndarray:
    """
    Convert integer DN values to float32 reflectance [0, 1].

    For processing baseline >= 04.00, applies BOA_ADD_OFFSET of -1000.
    """
    try:
        baseline_major = int(processing_baseline.split('.')[0])
    except (ValueError, AttributeError):
        baseline_major = 5  # default to recent baseline

    refl = dn.astype(np.float32)
    if baseline_major >= 4:
        refl = (refl + BASELINE_OFFSET) / QUANTIFICATION_VALUE
    else:
        refl = refl / QUANTIFICATION_VALUE

    refl = np.clip(refl, 0.0, None)
    return refl


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _get_cache_path(item, S2_data_folder: str) -> str:
    """Return the local cache file path for a STAC item."""
    safe_id = item.id.replace('/', '_').replace('\\', '_')
    return os.path.join(S2_data_folder, f"{safe_id}.npz")


def _save_to_cache(cache_path: str, band_data: np.ndarray, scl_data: np.ndarray,
                   geotransform: tuple, crs: str):
    """Save cropped integer bands + SCL to local cache."""
    np.savez_compressed(
        cache_path,
        band_data=band_data,
        scl_data=scl_data,
        geotransform=np.array(geotransform),
        crs=np.array(crs),
    )


def _load_from_cache(cache_path: str):
    """Load cached band data."""
    f = np.load(cache_path, allow_pickle=True)
    return (
        f['band_data'],
        f['scl_data'],
        tuple(f['geotransform']),
        str(f['crs']),
    )


# ---------------------------------------------------------------------------
# Per-item processing
# ---------------------------------------------------------------------------

def _process_item(item, geojson_cutline: str, vsi_prefix: str,
                  S2_data_folder: str) -> Tuple[np.ndarray, tuple, str]:
    """
    Process a single STAC item: read bands (from cache or remote),
    convert to reflectance, apply cloud mask.

    Returns:
        reflectance: (10, H, W) float32 with NaN for masked pixels
        geotransform: 6-tuple
        crs: WKT string
    """
    cache_path = _get_cache_path(item, S2_data_folder)

    if os.path.exists(cache_path):
        band_data, scl_data, geotransform, crs = _load_from_cache(cache_path)
    else:
        band_data, scl_data, geotransform, crs = _read_and_crop_all_bands(
            item, geojson_cutline, vsi_prefix
        )
        _save_to_cache(cache_path, band_data, scl_data, geotransform, crs)

    processing_baseline = item.properties.get('processing:version', '05.00')
    reflectance = _dn_to_reflectance(band_data, processing_baseline)
    reflectance = _apply_scl_cloud_mask(reflectance, scl_data)

    return reflectance, geotransform, crs


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def get_s2_official_data(start_date: str, end_date: str, geojson_path: str,
                         S2_data_folder: str = './') -> tuple:
    """
    Get Sentinel-2 L2A data from the CDSE STAC API.

    This function provides the exact same interface as the GEE-based
    get_s2_official_data() in s2_data_reader.py.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        geojson_path: Path to GeoJSON file containing the field boundary.
        S2_data_folder: Directory for caching downloaded data.

    Returns:
        tuple: (s2_refs, s2_uncs, s2_angles, doys, mask, geotransform, crs)
            - s2_refs: (N_images, 10, H, W) float32 reflectance [0,1]
            - s2_uncs: (N_images, 10, H, W) float32 uncertainties (10%)
            - s2_angles: (3, N_images) float64 [SZA, VZA, RAA] degrees
            - doys: (N_images,) int64 day of year
            - mask: (H, W) bool, True where all-NaN
            - geotransform: 6-tuple GDAL geotransform
            - crs: WKT projection string
    """
    try:
        # Configure GDAL for CDSE access
        vsi_prefix = _configure_gdal_for_cdse()

        # Load geometry
        geometry = _load_geojson(geojson_path)
        centroid = geometry.centroid
        geojson_dict = json.loads(shapely.to_geojson(geometry))

        os.makedirs(S2_data_folder, exist_ok=True)

        # Search STAC
        items = _search_s2_collection(geojson_dict, start_date, end_date)

        # Filter to single MGRS tile
        items = _filter_to_single_mgrs_tile(items, centroid.x, centroid.y)

        if not items:
            raise RuntimeError(
                f"No Sentinel-2 L2A items found on CDSE for the given field "
                f"and date range ({start_date} to {end_date})."
            )

        # Sort by datetime
        items.sort(key=lambda x: x.datetime)

        # Extract angles and DOYs
        szas, vzas, raas, doys = [], [], [], []
        for item in items:
            sza, vza, raa = _extract_angles_from_item(item)
            szas.append(sza)
            vzas.append(vza)
            raas.append(raa)
            doys.append(_extract_doy_from_item(item))

        s2_angles = np.array([szas, vzas, raas], dtype=np.float64)
        doys = np.array(doys, dtype=np.int64)

        # Read and process bands (concurrent per-item downloads)
        # Pass the file path as the cutline — GDAL can read GeoJSON files directly
        geojson_cutline = geojson_path
        process_fn = partial(
            _process_item,
            geojson_cutline=geojson_cutline,
            vsi_prefix=vsi_prefix,
            S2_data_folder=S2_data_folder,
        )
        # Limit concurrency to avoid CDSE rate limiting (HTTP 429)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(process_fn, items))

        s2_reflectances = []
        geotransform = None
        crs = None
        for refl, gt, proj in results:
            s2_reflectances.append(refl)
            if geotransform is None:
                geotransform = gt
                crs = proj

        s2_refs = np.array(s2_reflectances, dtype=np.float32)
        s2_uncs = np.abs(s2_refs) * 0.1

        # Compute mask: True where ALL time steps and bands are NaN
        mask = np.all(np.isnan(s2_refs), axis=(0, 1))

        return s2_refs, s2_uncs, s2_angles, doys, mask, geotransform, crs

    except EnvironmentError:
        raise
    except Exception as e:
        raise RuntimeError(
            "An error occurred while retrieving Sentinel-2 data from CDSE."
        ) from e
