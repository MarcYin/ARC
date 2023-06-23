import os
import ee
import json
import mgrs
import shutil
import requests
import datetime
import numpy as np
from osgeo import gdal
from functools import partial
from shapely.geometry import shape
from concurrent.futures import ThreadPoolExecutor

from typing import List, Tuple, Union, Any
gdal.PushErrorHandler('CPLQuietErrorHandler')

ee.Initialize(opt_url = 'https://earthengine-highvolume.googleapis.com')

def load_geojson(file_path: str):
    """
    Load GeoJSON data from a file.

    Args:
        file_path (str): The path of the GeoJSON file.

    Returns:
        shape: The first shape in the GeoJSON file.
    """
    with open(file_path) as f:
        features = json.load(f)["features"]
    return shape(features[0]["geometry"]).buffer(0)

def download_s2_image(feature, geom, S2_data_folder: str) -> str:
    """
    Download an S2 image from a given URL.

    Args:
        feature: The feature of the image.
        geom: The geometry of the image.
        S2_data_folder (str): The folder to store the image.

    Returns:
        str: The file path of the downloaded image.
    """
    s2_data_Res = 10
    image_id = feature['id']
    S2_product_id = feature['properties']['PRODUCT_ID']
    image = ee.Image(image_id)

    # Add the cloud probability band to the image.
    cloud = ee.Image('COPERNICUS/S2_CLOUD_PROBABILITY/%s'%feature['properties']['system:index'])
    image = image.addBands(cloud)

    filename = os.path.join(S2_data_folder, S2_product_id + '.tif')
    bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12', 'probability']

    if not os.path.exists(filename):
        # Define the download options.
        download_option = {'name': S2_product_id, 'scale': s2_data_Res, 'bands': bands, 'region': geom, 'format': "GEO_TIFF", 'maxPixels': 1e16}

        # Get the download URL.
        url = image.getDownloadURL(download_option)
        
        # Download the image.
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)

    return filename

def read_s2_official_data(file_names: list[str], geojson_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads S2 official data from provided files, then uses a geojson file to crop 
    the data and handle nodata. It also assumes 10% uncertainty in the S2 data.

    Args:
        file_names (list[str]): The list of file names to read from.
        geojson_path (str): The GeoJSON file path used for cropping.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the S2 references and their uncertainties.
        
    Raises:
        IOError: An error occurred accessing the bigtable.Table object.
    """
    s2_references = []
    for file_name in file_names:
        g = gdal.Warp('', file_name, format='MEM', cutlineDSName=geojson_path, cropToCutline=True, dstNodata=np.nan, outputType=gdal.GDT_Float32)
        
        if g is None:
            raise IOError("An error occurred while reading the file: {}".format(file_name))
        
        data = g.ReadAsArray()
        cloud = data[-1]
        data = np.where(cloud > 40, np.nan, data[:-1] / 10000.0)
        s2_references.append(data)
    
    s2_references = np.array(s2_references)
    s2_uncertainties = s2_references * 0.1
    
    return s2_references, s2_uncertainties

def calculate_s2_angles(features: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the S2 angles for a list of image features.

    Args:
        features (list[dict]): The list of features for each image.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Three arrays containing the SZA, VZA, and RAA angles respectively.

    Raises:
        KeyError: If any of the expected properties are missing from the image features.
    """
    def get_average_angle(properties: dict, angle_keys: list) -> float:
        """Helper function to calculate the average angle from the feature properties."""
        return np.nanmean([properties.get(key, np.nan) for key in angle_keys])

    s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    mean_vaa_keys = [f'MEAN_INCIDENCE_AZIMUTH_ANGLE_{band}' for band in s2_bands]
    mean_vza_keys = [f'MEAN_INCIDENCE_ZENITH_ANGLE_{band}' for band in s2_bands]

    szas, vzas, raas = [], [], []
    for feature in features:
        properties = feature['properties']
        vaa = get_average_angle(properties, mean_vaa_keys)
        vza = get_average_angle(properties, mean_vza_keys)

        saa = properties['MEAN_SOLAR_AZIMUTH_ANGLE']
        sza = properties['MEAN_SOLAR_ZENITH_ANGLE']
        raa = (vaa - saa) % 360

        szas.append(sza)
        vzas.append(vza)
        raas.append(raa)

    return np.array(szas), np.array(vzas), np.array(raas)


def get_geometry_and_centroid(geometry: 'Geometry') -> Tuple['Geometry', Tuple[float, float]]:
    """
    Given a geojson geometry object, this function will return the same geometry and its centroid.
    
    Parameters:
    geometry (shapely.geometry): A geojson geometry object.

    Returns:
    tuple: A tuple containing the input geometry and its centroid (as a tuple of floats).
    """
    try:
        centroid = geometry.centroid.coords[0]
        return geometry, centroid
    except Exception as e:
        raise ValueError("Failed to calculate centroid. Ensure input is a valid geojson Geometry object.") from e

def get_geojson_geometry_and_centroid(geojson_path: str) -> Tuple['Geometry', Tuple[float, float]]:
    """
    Given a geojson file path, this function will load the geojson, 
    and then return its geometry and centroid.
    
    Parameters:
    geojson_path (str): A string path of the geojson file.

    Returns:
    tuple: A tuple containing the geojson geometry and its centroid (as a tuple of floats).
    """
    try:
        geometry = load_geojson(geojson_path)
        return get_geometry_and_centroid(geometry)
    except FileNotFoundError:
        raise ValueError(f"Geojson file not found at: {geojson_path}")
    except Exception as e:
        raise ValueError("An error occurred while processing the geojson file.") from e

def convert_geometry_to_ee_and_mgrs(centroid: Tuple[float, float], 
                                    geometry: 'Geometry') -> Tuple['ee.Geometry', str]:
    """
    Convert given geometry and centroid to Google Earth Engine (ee) geometry and 
    Military Grid Reference System (MGRS) tile string.
    
    Parameters:
    centroid (tuple): A tuple representing the centroid of the geometry in the form (longitude, latitude).
    geometry (shapely.geometry): A geojson geometry object.

    Returns:
    tuple: A tuple containing the converted Google Earth Engine geometry and MGRS tile string.
    """
    try:
        longitude, latitude = centroid

        if not (-180 <= longitude <= 180) or not (-90 <= latitude <= 90):
            raise ValueError("Invalid longitude or latitude values. They must be within (-180 to 180) for longitude and (-90 to 90) for latitude.")
        
        if geometry.is_empty:
            raise ValueError("The provided geometry is empty.")
        
        coords = list(geometry.exterior.coords)
        
        ee_geometry = ee.Geometry.Polygon(coords)
        
        m = mgrs.MGRS()
        mgrs_tile = m.toMGRS(latitude, longitude)[:5]

        return ee_geometry, mgrs_tile

    except Exception as e:
        raise ValueError("Failed to convert geometry and centroid to Google Earth Engine Geometry and MGRS.") from e


def filter_s2_collection(ee_geometry: 'ee.Geometry', start_date: str, 
                         end_date: str, mgrs_tile: str) -> List[Any]:
    """
    Filters the Sentinel-2 ImageCollection by geographical bounds, date range, 
    and MGRS tile. Then sorts the collection by start time.
    
    Parameters:
    ee_geometry (ee.Geometry): The Google Earth Engine geometry to filter images by.
    start_date (str): The start date to filter images by (inclusive), in 'YYYY-MM-DD' format.
    end_date (str): The end date to filter images by (inclusive), in 'YYYY-MM-DD' format.
    mgrs_tile (str): The MGRS tile to filter images by.

    Returns:
    list: A list of filtered feature dictionaries.
    """
    try:
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        s2 = s2.filterBounds(ee_geometry).filterDate(start_date, end_date)\
                .filterMetadata('MGRS_TILE', 'equals', mgrs_tile).sort("system:time_start")
        features = s2.getInfo()['features']
        return features
    except Exception as e:
        raise RuntimeError("Failed to filter the Sentinel-2 ImageCollection.") from e

def get_doys(features: List[Union[dict, 'ee.Feature']]) -> np.ndarray:
    """
    Given a list of features, each containing a 'PRODUCT_ID' property, this 
    function will return an array of the day of year (DOY) for each feature.

    Parameters:
    features (list): A list of feature dictionaries or ee.Feature objects. Each 
                     feature must have a 'PRODUCT_ID' property in the format 
                     'XXXX_YYYYMMDDHHMMSS_ZZZZZZ_...'

    Returns:
    numpy.ndarray: A numpy array containing the day of year (DOY) for each feature.
    """
    try:
        return np.array([
            datetime.datetime.strptime(
                feature['properties']['PRODUCT_ID'].split('_')[2][:8], '%Y%m%d'
            ).timetuple().tm_yday for feature in features
        ])
    except Exception as e:
        raise ValueError("An error occurred while processing the features. Make sure each feature has a 'PRODUCT_ID' property in the format 'XXXX_YYYYMMDDHHMMSS_ZZZZZZ_...'") from e

def download_images(ee_geometry: 'ee.Geometry', 
                    features: List[Any], 
                    S2_data_folder: str) -> List[str]:
    """
    Downloads Sentinel-2 images for the given features using concurrent threads and returns the list of file paths.
    
    Parameters:
    ee_geometry (ee.Geometry): The Google Earth Engine geometry.
    features (list): A list of feature dictionaries.
    S2_data_folder (str): The directory to save the Sentinel-2 images.

    Returns:
    list: A list of filenames representing the downloaded Sentinel-2 images.
    """
    try:
        par = partial(download_s2_image, geom=ee_geometry, S2_data_folder=S2_data_folder)
        with ThreadPoolExecutor() as executor:
            filenames = list(executor.map(par, features))
        return filenames
    except Exception as e:
        raise RuntimeError("An error occurred during image download.") from e

def get_mask_and_metadata(filenames: List[str], 
                          s2_refs: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float, float, float, float], str]:
    """
    Open the first file in filenames with GDAL, get its geotransform and projection (CRS),
    and create a mask from s2_refs where all elements along the first and second axes are NaN.
    
    Parameters:
    filenames (List[str]): A list of file paths.
    s2_refs (numpy.ndarray): A numpy array for creating the mask.

    Returns:
    Tuple[numpy.ndarray, Tuple[float, float, float, float, float, float], str]:
        A mask numpy array, the geotransform parameters as a tuple, and the coordinate reference system string.
    """
    try:
        g = gdal.Open(filenames[0])
        if g is None:
            raise IOError(f"Error opening file: {filenames[0]}")
        geotransform = g.GetGeoTransform()
        crs = g.GetProjection()
        mask = np.all(np.isnan(s2_refs), axis=(0, 1))
        return mask, geotransform, crs
    except Exception as e:
        raise RuntimeError("An error occurred while retrieving mask and metadata.") from e


def get_s2_official_data(start_date: str, end_date: str, geojson_path: str, S2_data_folder: str = './') -> tuple:
    """
    Get the S2 official data.

    Args:
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.
        geojson_path (str): The path to the GeoJSON file containing the region of interest.
        S2_data_folder (str, optional): The folder to store the data. Defaults to './'.

    Returns:
        tuple: A tuple containing the following items:
            s2_refs (np.ndarray): The S2 references.
            s2_uncs (np.ndarray): The S2 uncertainties.
            s2_angles (list): The S2 angles.
            doys (np.ndarray): The day of the year for each image.
            mask (np.ndarray): A mask indicating where data is NaN.
            geotransform (tuple): The geotransform of the images.
            crs (str): The coordinate reference system of the images.
    """
    try:
        # Load the geojson geometry and calculate its centroid
        geometry, centroid = get_geojson_geometry_and_centroid(geojson_path)

        # Convert the centroid and geometry to Earth Engine (EE) object and MGRS tile
        ee_geometry, mgrs_tile = convert_geometry_to_ee_and_mgrs(centroid, geometry)

        # Filter the Sentinel-2 (S2) collection based on the EE geometry, date range, and MGRS tile
        features = filter_s2_collection(ee_geometry, start_date, end_date, mgrs_tile)

        # Calculate S2 angles
        szas, vzas, raas = calculate_s2_angles(features)
        s2_angles = [szas, vzas, raas]

        # Get the day of the year (DOY) for each feature
        doys = get_doys(features)

        # Download the S2 images concurrently and get the filenames
        filenames = download_images(ee_geometry, features, S2_data_folder)

        # Read the S2 official data and get references and uncertainties
        s2_refs, s2_uncs = read_s2_official_data(filenames, geojson_path)
        
        # Get the mask and metadata from the first image
        mask, geotransform, crs = get_mask_and_metadata(filenames, s2_refs)

        return s2_refs, s2_uncs, s2_angles, doys, mask, geotransform, crs

    except Exception as e:
        raise RuntimeError("An error occurred while retrieving Sentinel-2 official data.") from e


def main():
    start_date = "2022-07-15"
    end_date = "2022-11-30"
    geojson_path = "test_data/SF_field.geojson"
    S2_data_folder = os.path.join(os.path.expanduser('~'), 'Downloads/' + geojson_path.split('/')[-1].split('.')[0] + '/')
    if not os.path.exists(S2_data_folder):
        os.makedirs(S2_data_folder)
    s2_refs, s2_uncs, s2_angles, doys, mask, geotransform, crs = get_s2_official_data(start_date, end_date, geojson_path, S2_data_folder=S2_data_folder)
    ndvi = (s2_refs[:, 7] - s2_refs[:, 2]) / (s2_refs[:, 7] + s2_refs[:, 2])
    
    import matplotlib.pyplot as plt
    nrows = int(len(ndvi) / 5) + int(len(ndvi) % 5 > 0)
    fig, axs = plt.subplots(ncols = 5, nrows = nrows, figsize = (20, 4 * nrows))
    axs = axs.ravel()
    for i in range(len(ndvi)):
        im = axs[i].imshow(ndvi[i], vmin = 0, vmax = 1)
        fig.colorbar(im, ax = axs[i])
        axs[i].set_title('DOY: %d'%doys[i])
    plt.show()

    mean_ndvi = np.nanmean(ndvi, axis = (1, 2))
    plt.figure(figsize = (20, 4))
    mask = np.isfinite(mean_ndvi)
    plt.plot(doys[mask], mean_ndvi[mask], '--o')
    plt.show()

if __name__ == "__main__":
    main()
