"""
Sentinel-2 L2A data retrieval from Element 84 Earth Search (AWS).

Cloud Optimized GeoTIFFs (COGs) served from the AWS Registry of Open Data.
No authentication required.

STAC endpoint: https://earth-search.aws.element84.com/v1
Collection: sentinel-2-l2a
"""

import os
import json
import threading
import numpy as np
from osgeo import gdal
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from shapely.geometry import shape
import shapely
import mgrs
from tqdm.auto import tqdm

import pystac_client

gdal.PushErrorHandler('CPLQuietErrorHandler')

# AWS Earth Search endpoints
AWS_STAC_URL = "https://earth-search.aws.element84.com/v1"
AWS_COLLECTION = "sentinel-2-l2a"

# Band asset keys mapping: AWS Earth Search uses descriptive names
# ARC order: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
ARC_BAND_ASSETS = [
    'blue', 'green', 'red',
    'rededge1', 'rededge2', 'rededge3',
    'nir', 'nir08', 'swir16', 'swir22',
]
SCL_ASSET_KEY = 'scl'

# SCL classes to mask: no data, saturated, cloud shadow, cloud med/high, cirrus, snow
SCL_MASK_VALUES = frozenset({0, 1, 3, 8, 9, 10, 11})

QUANTIFICATION_VALUE = 10000.0
BASELINE_OFFSET = -1000

# AWS has no strict connection limit like CDSE, but be reasonable
_AWS_MAX_CONCURRENT_READS = 8
_aws_read_semaphore = threading.Semaphore(_AWS_MAX_CONCURRENT_READS)


# ---------------------------------------------------------------------------
# GDAL configuration
# ---------------------------------------------------------------------------

def _configure_gdal_for_aws():
    """Configure GDAL for reading COGs from AWS S3 (public, no auth)."""
    gdal.SetConfigOption("AWS_NO_SIGN_REQUEST", "YES")
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.xml")
    gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", "5")
    gdal.SetConfigOption("GDAL_HTTP_RETRY_DELAY", "2")
    gdal.SetConfigOption("GDAL_CACHEMAX", "256")
    gdal.SetConfigOption("CPL_VSIL_CURL_CACHE_SIZE", "134217728")  # 128 MB
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("VSI_CACHE", "TRUE")
    gdal.SetConfigOption("VSI_CACHE_SIZE", "134217728")  # 128 MB per file


# ---------------------------------------------------------------------------
# GeoJSON loading
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
    """Search AWS Earth Search for Sentinel-2 L2A items."""
    client = pystac_client.Client.open(AWS_STAC_URL)
    search = client.search(
        collections=[AWS_COLLECTION],
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
    # AWS Earth Search uses mgrs:grid_square or grid:code
    filtered = []
    for item in items:
        props = item.properties
        grid_code = props.get("grid:code", "")
        mgrs_tile = props.get("mgrs:grid_square", "")
        s2_tile = props.get("s2:mgrs_tile", "")
        if (grid_code == f"MGRS-{target_tile}"
                or mgrs_tile == target_tile
                or s2_tile == target_tile):
            filtered.append(item)
    return filtered


# ---------------------------------------------------------------------------
# Angle extraction
# ---------------------------------------------------------------------------

def _extract_angles_from_item(item) -> Tuple[float, float, float]:
    """Extract (SZA, VZA, RAA) in degrees from STAC item properties."""
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
# Band reading with GDAL (COG - efficient windowed reads)
# ---------------------------------------------------------------------------

def _read_and_crop_band(href: str, geojson_cutline: str,
                        target_resolution: int = 10,
                        resample_alg: str = 'bilinear') -> tuple:
    """
    Read a single COG band, crop to field boundary, resample to target resolution.
    COGs support efficient partial reads via HTTP range requests.
    """
    # AWS COGs can be accessed directly via /vsicurl/
    if href.startswith("s3://"):
        vsi_path = f"/vsis3/{href[5:]}"
    elif href.startswith("http"):
        vsi_path = f"/vsicurl/{href}"
    else:
        vsi_path = href

    resample_map = {
        'bilinear': gdal.GRA_Bilinear,
        'nearest': gdal.GRA_NearestNeighbour,
    }

    with _aws_read_semaphore:
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


def _read_and_crop_all_bands(item, geojson_cutline: str,
                             target_resolution: int = 10,
                             band_executor: ThreadPoolExecutor = None):
    """
    Read all 10 spectral bands + SCL for a single STAC item.
    Band reads are submitted concurrently when band_executor is provided.
    """
    band_tasks = []
    for asset_key in ARC_BAND_ASSETS:
        asset = item.assets.get(asset_key)
        if asset is None:
            raise KeyError(f"Asset '{asset_key}' not found in item {item.id}")
        band_tasks.append((asset_key, asset.href, 'bilinear'))

    scl_asset = item.assets.get(SCL_ASSET_KEY)
    if scl_asset is None:
        raise KeyError(f"Asset '{SCL_ASSET_KEY}' not found in item {item.id}")
    band_tasks.append((SCL_ASSET_KEY, scl_asset.href, 'nearest'))

    if band_executor is not None:
        futures = {}
        for asset_key, href, resample in band_tasks:
            future = band_executor.submit(
                _read_and_crop_band,
                href, geojson_cutline,
                target_resolution=target_resolution,
                resample_alg=resample,
            )
            futures[future] = asset_key

        results = {}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    else:
        results = {}
        for asset_key, href, resample in band_tasks:
            results[asset_key] = _read_and_crop_band(
                href, geojson_cutline,
                target_resolution=target_resolution,
                resample_alg=resample,
            )

    bands = []
    geotransform = None
    crs = None
    for asset_key in ARC_BAND_ASSETS:
        data, gt, proj = results[asset_key]
        bands.append(data)
        if geotransform is None:
            geotransform = gt
            crs = proj

    scl_data, _, _ = results[SCL_ASSET_KEY]
    band_data = np.stack(bands, axis=0)
    return band_data, scl_data, geotransform, crs


# ---------------------------------------------------------------------------
# Cloud masking and reflectance conversion
# ---------------------------------------------------------------------------

def _apply_scl_cloud_mask(reflectance: np.ndarray, scl: np.ndarray) -> np.ndarray:
    """Mask pixels using the Scene Classification Layer (SCL)."""
    mask = np.isin(scl, list(SCL_MASK_VALUES))
    reflectance[:, mask] = np.nan
    return reflectance


def _dn_to_reflectance(dn: np.ndarray, processing_baseline: str) -> np.ndarray:
    """Convert integer DN values to float32 reflectance [0, 1]."""
    try:
        baseline_major = int(processing_baseline.split('.')[0])
    except (ValueError, AttributeError):
        baseline_major = 5

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
    safe_id = item.id.replace('/', '_').replace('\\', '_')
    return os.path.join(S2_data_folder, f"{safe_id}.npz")


def _save_to_cache(cache_path: str, band_data: np.ndarray, scl_data: np.ndarray,
                   geotransform: tuple, crs: str):
    np.savez_compressed(
        cache_path,
        band_data=band_data,
        scl_data=scl_data,
        geotransform=np.array(geotransform),
        crs=np.array(crs),
    )


def _load_from_cache(cache_path: str):
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

def _process_item(item, geojson_cutline: str, S2_data_folder: str,
                  band_executor: ThreadPoolExecutor = None) -> Tuple[np.ndarray, tuple, str]:
    """Process a single STAC item: read bands, convert to reflectance, apply cloud mask."""
    cache_path = _get_cache_path(item, S2_data_folder)

    if os.path.exists(cache_path):
        band_data, scl_data, geotransform, crs = _load_from_cache(cache_path)
    else:
        band_data, scl_data, geotransform, crs = _read_and_crop_all_bands(
            item, geojson_cutline,
            band_executor=band_executor,
        )
        _save_to_cache(cache_path, band_data, scl_data, geotransform, crs)

    processing_baseline = item.properties.get('s2:processing_baseline', '05.00')
    reflectance = _dn_to_reflectance(band_data, processing_baseline)
    reflectance = _apply_scl_cloud_mask(reflectance, scl_data)

    return reflectance, geotransform, crs


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def get_s2_official_data(start_date: str, end_date: str, geojson_path: str,
                         S2_data_folder: str = './') -> tuple:
    """
    Get Sentinel-2 L2A data from AWS Earth Search (Element 84).

    No authentication required. Data served as Cloud Optimized GeoTIFFs
    from the AWS Registry of Open Data.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        geojson_path: Path to GeoJSON file containing the field boundary.
        S2_data_folder: Directory for caching downloaded data.

    Returns:
        tuple: (s2_refs, s2_uncs, s2_angles, doys, mask, geotransform, crs)
    """
    try:
        _configure_gdal_for_aws()

        geometry = _load_geojson(geojson_path)
        centroid = geometry.centroid
        geojson_dict = json.loads(shapely.to_geojson(geometry))

        os.makedirs(S2_data_folder, exist_ok=True)

        # Search STAC
        items = _search_s2_collection(geojson_dict, start_date, end_date)
        print(f"STAC search (AWS): {len(items)} items found")

        # Filter to single MGRS tile
        items = _filter_to_single_mgrs_tile(items, centroid.x, centroid.y)

        if not items:
            raise RuntimeError(
                f"No Sentinel-2 L2A items found on AWS Earth Search for the given "
                f"field and date range ({start_date} to {end_date})."
            )

        tile_code = items[0].properties.get(
            "grid:code",
            items[0].properties.get("s2:mgrs_tile", "unknown")
        )
        print(f"Filtered to {len(items)} items on tile {tile_code}")

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

        # Check cache status
        geojson_cutline = geojson_path
        n_cached = sum(
            1 for item in items
            if os.path.exists(_get_cache_path(item, S2_data_folder))
        )
        n_to_download = len(items) - n_cached
        print(f"Cache: {n_cached}/{len(items)} items cached, "
              f"{n_to_download} to download")

        # Process items with concurrent band reads
        band_pool_size = _AWS_MAX_CONCURRENT_READS + 4

        with ThreadPoolExecutor(max_workers=band_pool_size) as band_executor:
            process_fn = partial(
                _process_item,
                geojson_cutline=geojson_cutline,
                S2_data_folder=S2_data_folder,
                band_executor=band_executor,
            )
            with ThreadPoolExecutor(max_workers=len(items)) as item_executor:
                future_to_idx = {
                    item_executor.submit(process_fn, item): i
                    for i, item in enumerate(items)
                }
                results = [None] * len(items)
                for future in tqdm(
                    as_completed(future_to_idx),
                    total=len(items),
                    desc="Processing S2 items (AWS)",
                    unit="item",
                ):
                    idx = future_to_idx[future]
                    results[idx] = future.result()

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
        mask = np.all(np.isnan(s2_refs), axis=(0, 1))

        return s2_refs, s2_uncs, s2_angles, doys, mask, geotransform, crs

    except Exception as e:
        raise RuntimeError(
            "An error occurred while retrieving Sentinel-2 data from AWS Earth Search."
        ) from e
