import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
# from BSM_soil import BSM
from scipy.stats import qmc

from tqdm import tqdm


from typing import List, Tuple, Union, Dict, Any

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def load_parameters(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load parameters from a numpy archive.

    Parameters:
        filepath (str): Path to the numpy archive.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing GSV, nw, and kw.
    """
    try:
        params = np.load(filepath, allow_pickle=True)
        return params['GSV'], params['nw'], params['kw']
    except IOError:
        print(f"Error: File {filepath} not found.")
        return None, None, None
    except KeyError as ke:
        print(f"Error: Key {ke} not found in file.")
        return None, None, None

def logistic_function(p: List[float], t: Union[float, np.ndarray]) -> np.ndarray:
    """
    Computes the double sigmoid logistic function.

    Parameters:
        p (List[float]): List of parameters for the logistic function.
        t (Union[float, np.ndarray]): Input values.

    Returns:
        np.ndarray: Output values computed using the logistic function.
    """
    assert len(p) == 6, 'The parameter list p should contain exactly six elements.'

    sigma1 = 1. / (1 + np.exp(p[2] * (t - p[3])))
    sigma2 = 1. / (1 + np.exp(-p[4] * (t - p[5])))

    return p[0] - p[1] * (sigma1 + sigma2 - 1)

def normalize_data(data):
    """
    Normalizes data in the range [0,1]
    """
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

def sample_logistic(sample, num_samples):
    """
    Updates the input sample and generates the logistic sample.
    """
    # Updating the sample
    sample[:, 3] += sample[:, 1]
    
    # Creating the y array
    y = np.concatenate([[np.zeros(num_samples), np.ones(num_samples)], sample.T])

    # Calculating logistics and ensuring non-negative values
    logistics = logistic_function(y, np.arange(365)[:, None])
    logistics[logistics<0] = 0
    
    # Normalizing logistics
    normalized_logistics = normalize_data(logistics)
    
    return normalized_logistics


def scale_samples(samples: np.ndarray, medians: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function scales the given samples based on the medians provided and the pre-defined limit values. 
    It ensures that all scaled values are within their respective limits.

    Parameters:
        samples (np.ndarray): The input samples to scale. It's a 2D array where each column represents a different attribute.
        medians (np.ndarray): The medians to use for scaling. It's a 1D array where each element corresponds to an attribute.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the scaled samples and the original samples.
    """

    # Define the lower and upper limits for each attribute
    limits = [
        [1, 3],    # N
        [0, 140],  # Cab
        [0, 0.02], # Cw
        [0, 0.06], # Cm
        [0, 10],   # LAI
        [20, 90],  # Leaf angle
        [0, 1.5]   # Cbrown
    ]

    # Initialize an empty list to hold the scaled samples
    scaled_samples = []

    # Scale each attribute separately
    for i in range(7):
        lower_limit, upper_limit = limits[i]
        
        # Multiply the samples by the corresponding median
        scaled = samples[:, i, None] * medians[:, i][None]

        # Ensure that the scaled samples are within the specified limits
        scaled = np.clip(scaled, lower_limit, upper_limit)

        # Transpose the scaled samples and add them to the list
        scaled_samples.append(scaled.T)

    # Convert the list of scaled samples to a numpy array
    scaled_samples = np.array(scaled_samples)

    return scaled_samples, samples


def compute_difference(p: List[float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the difference between a logistic function and provided y values.

    Parameters:
        p (List[float]): List of parameters for the logistic function.
        x (np.ndarray): Input values for the logistic function.
        y (np.ndarray): Target y values to compare with the logistic function's results.

    Returns:
        np.ndarray: Difference between the logistic function and y values. Any non-finite values are replaced with 0.
    """
    
    # Check if inputs are numpy arrays
    assert isinstance(x, np.ndarray), "Input x must be a numpy array."
    assert isinstance(y, np.ndarray), "Input y must be a numpy array."
    
    # Check if input arrays have same shape
    assert x.shape == y.shape, "Input arrays x and y must have the same shape."

    # Compute logistic function with given parameters and inputs
    result = logistic_function(p, x)

    # Calculate difference between logistic function results and target y values
    difference = result - y

    # Replace any non-finite values in the difference array with 0
    difference[~np.isfinite(difference)] = 0

    return difference


def get_mapping(reference_p: List[float], logistics: np.ndarray, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Maps the indices of a reference logistic function to the provided logistic functions.

    Parameters:
        reference_p (List[float]): Parameters of the reference logistic function.
        logistics (np.ndarray): Logistic functions to map.
        num_samples (int): Number of samples.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the mapping and ensemble indices.
    """
    # Ensure that reference_p has exactly six parameters
    assert len(reference_p) == 6, 'reference_p should have six parameters.'

    # Generate 365 days array
    days = np.arange(365)

    # Generate reference logistic function and normalize it
    y = logistic_function(reference_p, days)
    normalized_y = (y - y.min()) / (y.max() - y.min())
    normalized_y[np.argmax(normalized_y):] = normalized_y[np.argmax(normalized_y):] * -1 + 2

    # Flip the logistic functions after their maximum points
    argmaxs = np.argmax(logistics, axis=0)
    mask = (days[:, None] - argmaxs[None]) > 0
    logistics[mask] = logistics[mask] * -1 + 2

    # Create an interpolation function based on the normalized reference logistic function
    interpolation_func = interp1d(normalized_y, days, fill_value='extrapolate')

    # Generate the mapping of logistics to the reference, and ensure it falls within the range [0, 364]
    mapping = np.clip(interpolation_func(logistics).astype(int), 0, 364)

    # Generate the ensemble indices
    ensemble_indices = np.repeat(np.arange(num_samples), 365).reshape(num_samples, 365).T

    return mapping, ensemble_indices

def load_crop_model(crop_type: str) -> Dict[str, np.ndarray]:
    """
    Load crop model from a numpy archive.

        Parameters:
        crop_type (str): Type of crop.

    Returns:
        Dict[str, np.ndarray]: Loaded crop model.
    """
    # Map of crop types to file paths
    crop_models = {
        'maize': data_dir + '/US_001.npz',
        'soy': data_dir + '/US_005.npz',
        'rice': data_dir + '/China_000.npz',
        'wheat': data_dir + '/US_024.npz',
    }
    crop_type = crop_type.lower()
    if crop_type not in crop_models:
        raise ValueError(f"Invalid crop type {crop_type}. Valid types are: {list(crop_models.keys())}")

    crop_model_file = crop_models[crop_type]
    return np.load(crop_model_file)


def adjust_parameters(meds: np.ndarray, p_mins: np.ndarray, p_maxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjust parameters based on median values.

    Parameters:
        meds (np.ndarray): Median values.
        p_mins (np.ndarray): Minimum parameter values.
        p_maxs (np.ndarray): Maximum parameter values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Adjusted p_mins and p_maxs.
    """
    maxs = np.nanmax(meds, axis=0)

    for i in range(7):
        bad = ~np.isfinite(meds[:, i])
        meds[bad, i] = np.nanmin(meds[:, i])
        p_maxs[i + 4] = p_maxs[i + 4] / maxs[i]
        p_mins[i + 4] = p_mins[i + 4] / maxs[i]

    return p_mins, p_maxs

def compute_reference_parameters(medians: np.ndarray) -> Tuple[float, ...]:
    """
    Compute reference parameters for logistic functions.

    Parameters:
        medians (np.ndarray): Median values.

    Returns:
        Tuple[float, ...]: Reference parameters.
    """
    NUM_DAYS = 365
    INITIAL_PARAMETERS = 0., 2, 0.1, None, 0.1, None  # Placeholder values for the start and end days
    BOUNDS = np.array([[0, 1], [0, 10], [0, 1], [0, NUM_DAYS], [0, 1], [0, NUM_DAYS]]).T

    day_indices = np.arange(NUM_DAYS)
    mid_index = np.nanmedian(day_indices[np.argsort(medians[:, 4])][-10:])

    start_day = max([min([(mid_index + 0) / 2, NUM_DAYS]), 0])
    end_day = max([min([(mid_index + NUM_DAYS) / 2, NUM_DAYS]), 0])

    initial_parameters = list(INITIAL_PARAMETERS)
    initial_parameters[3] = start_day
    initial_parameters[5] = end_day

    result = least_squares(compute_difference, initial_parameters, loss='soft_l1', f_scale=0.001,
                           args=(np.arange(NUM_DAYS), medians[:, 4]), bounds=BOUNDS)
    
    return tuple(result.x)


def load_model(model_path: str) -> Any:
    """
    Load a model from a given path.

    Args:
        model_path (str): Path to the model.

    Returns:
        Any: Loaded model.
    """
    f = np.load(model_path, allow_pickle=True)
    model_weights = f['model_weights'].tolist()

    return model_weights


def predict_input_slices(inp_slices: List[np.ndarray], model_weights: List[np.ndarray]) -> List[np.ndarray]:
    """
    Predicts using the model for each input slice.

    Args:
        inp_slices (List[np.ndarray]): List of input slices.
        model (Any): Model used for prediction.

    Returns:
        List[np.ndarray]: List of model prediction results.
    """
    from arc.NN_predict_jax import predict
    
    predictions = []
    for inp_slice in tqdm(inp_slices, desc="Predicting S2 reflectance", unit="slice"):
        prediction = predict(inp_slice, model_weights, cal_jac=False)
        predictions.append(prediction)
    predictions = np.concatenate(predictions, axis=1).squeeze()
    return predictions


def adjust_orig_bios(orig_bios: np.ndarray, multipliers: List[int]) -> np.ndarray:
    """
    Adjusts `orig_bios` by multiplying each of its elements by corresponding multipliers.

    Args:
        orig_bios (np.ndarray): Original bios values.
        multipliers (List[int]): Multipliers.

    Returns:
        np.ndarray: Adjusted `orig_bios`.
    """
    temp = np.zeros_like(orig_bios).astype(int)
    for i, multiplier in enumerate(multipliers):
        temp[i] = (orig_bios[i] * multiplier).astype(int)
    return temp


def generate_samples(p_mins: np.ndarray, p_maxs: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Generate samples within the specified range using a Sobol sequence.

    The generated samples cover the multi-dimensional space more uniformly than uniform random numbers.

    Args:
        p_mins (np.ndarray): The minimum values for each dimension.
        p_maxs (np.ndarray): The maximum values for each dimension.
        num_samples (int): The number of samples to generate. The actual number of generated samples
                           will be the smallest power of 2 that is greater than or equal to `num_samples`.

    Returns:
        np.ndarray: The generated samples. Each row corresponds to a sample, and each column corresponds
                    to a dimension.
    """
    m = int(np.ceil(np.log2(num_samples)))

    sampler = qmc.Sobol(d=len(p_mins), scramble=False)
    sample = sampler.random_base2(m=m)
    sample = qmc.scale(sample, p_mins, p_maxs)
    
    return sample


def compute_walthall_coef(sza, vza, raa):
    """
    Compute the Walthall coefficient.
    
    Args:
        sza (float): Solar zenith angle.
        vza (float): View zenith angle.
        raa (float): Relative azimuth angle.

    Returns:
        float: Walthall coefficient.
    """
    # Implement this function based on your specific needs.
    rad_sza = np.deg2rad(sza)
    rad_vza = np.deg2rad(vza)
    rad_raa = np.deg2rad(raa)
    Walthall_coef = 1 / 16.41 * (rad_sza * rad_vza * np.cos(rad_raa) * 7.363 - 4.3 * (rad_vza**2 + rad_sza**2) + 7.702 * rad_sza**2 * rad_vza**2)
    return Walthall_coef


def prepare_geometry_params(sza, vza, raa, doys, num_samples):
    """
    Prepare the geometry parameters.

    Args:
        sza (float): Solar zenith angle.
        vza (float): View zenith angle.
        raa (float): Relative azimuth angle.
        doys (np.ndarray): Array of day of year values.
        num_samples (int): Number of samples.

    Returns:
        tuple: A tuple containing the prepared sza, vza, and raa parameters.
    """
    # Implement this function based on your specific needs.
    sza = np.repeat(sza, num_samples).reshape(len(doys), num_samples)
    vza = np.repeat(vza, num_samples).reshape(len(doys), num_samples)
    raa = np.repeat(raa, num_samples).reshape(len(doys), num_samples)
    return sza, vza, raa

def adjust_soil_params(p0, p1, p2, p3, Walthall_coef):
    """
    Adjust and normalize the soil parameters.

    Args:
        p0, p1, p2, p3 (float): Soil parameters.
        Walthall_coef (float): Walthall coefficient.

    Returns:
        tuple: A tuple containing the adjusted and normalized p0, p1, p2, and p3 parameters.
    """
    # Implement this function based on your specific needs.

    p0 = np.array([p0,] * len(Walthall_coef))
    p1 = np.array([p1,] * len(Walthall_coef))
    p2 = np.array([p2,] * len(Walthall_coef))
    p3 = np.array([p3,] * len(Walthall_coef))

    p0 = p0 / 1.5
    p1 = (p1 - 10) / 70
    p2 = (p2 - 22) / (130 - 22)
    p3 = (p3 - 2 ) / (100 - 2)
    p0 = p0 + p0 * Walthall_coef[:, None]
    return p0, p1, p2, p3

def adjust_bio_params(N, cab, cm, cw, lai, ala, cbrown, mapping, ens_inds, doys):
    """
    Adjust the biophysical parameters with a temporal dynamic and random offset.

    Args:
        N, cab, cm, cw, lai, ala, cbrown (float): Biophysical parameters.
        mapping (np.ndarray): Temporal mapping array.
        ens_inds (np.ndarray): Array of ensemble indices.
        doys (np.ndarray): Array of day of year values.

    Returns:
        tuple: A tuple containing the adjusted N, cab, cm, cw, lai, ala, and cbrown parameters.
    """
    # Implement this function based on your specific needs.
    random_date = 0
    offset = np.random.uniform(-random_date, random_date, ens_inds.shape[1]).astype(int)
    N = N[np.minimum(mapping + offset[None,], 364), ens_inds][doys-1] #* 0 + 2.2 
    # map_ind = np.arange(len(mapping[0]))
    # np.random.shuffle(map_ind)
    # mapping = mapping[:, map_ind]
    # np.random.shuffle(mapping)

    offset = np.random.uniform(-random_date, random_date, ens_inds.shape[1]).astype(int)
    cab = cab[np.minimum(mapping + offset[None,], 364), ens_inds][doys-1]
    # np.random.shuffle(mapping)
    # np.random.shuffle(map_ind)
    # mapping = mapping[:, map_ind]

    offset = np.random.uniform(-random_date, random_date, ens_inds.shape[1]).astype(int)
    cm = cm[np.minimum(mapping + offset[None,], 364), ens_inds][doys-1] 
    # np.random.shuffle(mapping)
    # np.random.shuffle(map_ind)
    # mapping = mapping[:, map_ind]

    offset = np.random.uniform(-random_date, random_date, ens_inds.shape[1]).astype(int)
    cw = cw[np.minimum(mapping + offset[None,], 364), ens_inds][doys-1] 
    # np.random.shuffle(mapping)
    # np.random.shuffle(map_ind)
    # mapping = mapping[:, map_ind]

    offset = np.random.uniform(-random_date, random_date, ens_inds.shape[1]).astype(int)
    lai= lai[np.minimum(mapping + offset[None,], 364), ens_inds][doys-1]
    # np.random.shuffle(mapping)
    # np.random.shuffle(map_ind)
    # mapping = mapping[:, map_ind]

    offset = np.random.uniform(-random_date, random_date, ens_inds.shape[1]).astype(int)
    ala = ala[np.minimum(mapping + offset[None,], 364), ens_inds][doys-1] 
    # np.random.shuffle(mapping)
    # np.random.shuffle(map_ind)
    # mapping = mapping[:, map_ind]

    offset = np.random.uniform(-random_date, random_date, ens_inds.shape[1]).astype(int)
    cbrown = cbrown[np.minimum(mapping + offset[None,], 364), ens_inds][doys-1]

    return N, cab, cm, cw, lai, ala, cbrown



def prepare_final_input(N, cab, car, cbrown, cw, cm, lai, ala, sza, vza, raa, p0, p1, p2, p3):
    """
    Prepare the final input array.

    Args:
        N, cab, car, cbrown, cw, cm, lai, ala (float): Biophysical parameters.
        sza, vza, raa, p0, p1, p2, p3 (float): Geometry and soil parameters.

    Returns:
        np.ndarray: The final input array.
    """
    sza = np.cos(np.deg2rad(sza))
    vza = np.cos(np.deg2rad(vza))
    raa = raa % 360 / 360
    inp = [
        (N - 1.) / 2.5, 
        np.exp(-cab/100.), 
        np.exp(-car/100.), 
        cbrown, 
        np.exp(-50.*cw), 
        np.exp(-50.*cm), 
        np.exp(-lai/2.), 
        np.cos(np.deg2rad(ala)), 
        sza, vza, raa, p0, p1, p2, p3
    ]
    inp = np.array(inp)
    inp = inp.reshape(15, -1)
    return inp


def create_sample(bio_paras, angs, soil_paras, doys, ens_inds, mapping, num_samples):
    """
    Creates an array of samples containing parameters related to bio and soil properties, 
    observation geometry, and temporal information.
    
    Args:
        bio_paras (np.ndarray): Biophysical parameters.
        angs (tuple): Tuple containing the angles (sza, vza, raa).
        soil_paras (np.ndarray): Soil parameters.
        doys (np.ndarray): Array of day of year values.
        ens_inds (np.ndarray): Array of ensemble indices.
        mapping (np.ndarray): Temporal mapping array.
        num_samples (int): Number of samples to generate.
        
    Returns:
        tuple: Input samples and original biophysical parameters.
    """
    # Extract bio and soil parameters, and angles
    N, cab, cm, cw, lai, ala, cbrown = bio_paras
    p0, p1, p2, p3 = soil_paras
    sza, vza, raa = angs

    # Compute Walthall coefficient
    Walthall_coef = compute_walthall_coef(sza, vza, raa)

    # Prepare geometry parameters
    sza, vza, raa = prepare_geometry_params(sza, vza, raa, doys, num_samples)

    # Adjust and normalize soil parameters
    p0, p1, p2, p3 = adjust_soil_params(p0, p1, p2, p3, Walthall_coef)

    # Adjust biophysical parameters with a temporal dynamic and random offset
    N, cab, cm, cw, lai, ala, cbrown = adjust_bio_params(N, cab, cm, cw, lai, ala, cbrown, mapping, ens_inds, doys)

    # Assume carotenoid content is a quarter of chlorophyll content
    car = cab / 4

    # Prepare final input array
    inp = prepare_final_input(N, cab, car, cbrown, cw, cm, lai, ala, sza, vza, raa, p0, p1, p2, p3)
    
    # Package original biophysical parameters
    orig_bios = np.array([N, cab, cm, cw, lai, ala, cbrown])

    return inp, orig_bios



def generate_ref_samples(p_mins, p_maxs, num_samples, angs, doys, crop_type):
    """
    Generates reference samples based on provided parameters and crop information.

    Args:
        p_mins (np.ndarray): Minimum parameter values.
        p_maxs (np.ndarray): Maximum parameter values.
        num_samples (int): Number of samples to generate.
        angs (tuple): Tuple containing the angles (vza, sza, raa).
        doys (np.ndarray): Array of day of year values.
        crop_type (str): Type of crop.

    Returns:
        tuple: A tuple containing the generated reference samples, pheo_samples, bio_samples,
        orig_bios, and soil_samples.
    """
    # Load the crop model
    crop_model = load_crop_model(crop_type)
    deltas = crop_model['deltas']
    medians = crop_model['meds']

    # Adjust the parameter ranges
    p_mins, p_maxs = adjust_parameters(medians, p_mins, p_maxs)

    # Generate the parameter samples
    sample = generate_samples(p_mins, p_maxs, num_samples)

    # Divide the samples into phenological, biophysical, and soil samples
    pheo_samples = sample[:,   :4]
    bio_samples  = sample[:, 4:-4]
    soil_samples = sample[:,  -4:]
    # number of samples will change after the Sobol sampling
    num_samples  = pheo_samples.shape[0]

    # Compute the reference parameters and create the logistic curves
    reference_p = compute_reference_parameters(medians)
    logistics = sample_logistic(pheo_samples, num_samples)

    # Get the mapping and scale the samples
    mapping, ens_inds = get_mapping(reference_p, logistics, num_samples)
    bio_paras, bio_samples = scale_samples(bio_samples, medians)

    # Create the input samples for the forward model
    inp, orig_bios = create_sample(bio_paras, angs, soil_samples.T, doys, ens_inds, mapping, num_samples)

    # Adjust the original biophysical parameters
    orig_bios = adjust_orig_bios(orig_bios, [100, 100, 10000, 10000, 100, 100, 1000])

    # Load the forward model weights
    model_weights = load_model(data_dir + '/foward_prosail_model_weights.npz')
    
    # Split the inputs for batch processing
    inp_slices = np.array_split(np.atleast_2d(inp).T, 300)

    del inp

    # Use the forward model to generate the reference spectra
    arc_refs = predict_input_slices(inp_slices, model_weights)

    # Reshape the output
    arc_refs = arc_refs.reshape(10, len(doys), num_samples)

    return arc_refs, pheo_samples, bio_samples, orig_bios, soil_samples



def generate_arc_refs(doys: List[int], start_of_season: int, growth_season_length: int, num_samples: int, angs: List[float], crop_type: str) -> Tuple:
    """
    Generates references for Sentinel-2 based on given parameters.

    Args:
        doys (List[int]): List of days of the year.
        start_of_season (int): Start of the season in days of the year.
        growth_season_length (int): Length of the growth season in days.
        num_samples (int): Number of samples to generate.
        angs (List[float]): List of angles.
        crop_type (str): Type of the crop.

    Returns:
        Tuple: Tuple containing Sentinel-2 references, phenology samples, bio samples, original bios, and soil samples.
    """
    # Define parameters
    # paras = ['growth_speed', 'start_of_season', 'senescence_speed', 'end_of_season', 'N', 'Cab', 'Cm', 'Cw', 'Lai', 'Ala', 'Cb', 'soil_brightness', 'soil_shape_p1', 'soil_shape_p2', 'soil_volume_moisture']
    # p_mins = [0.045, max([start_of_season - 10,   0]),  0.01,  growth_season_length-30, 1,  40,   0.002,  0.01, 0.5,  60 ,    0,  0.1,   10,   10,   2]
    # p_maxs = [0.325, min([start_of_season + 50, 365]),  0.37,  growth_season_length+30, 3, 100,   0.015,  0.06,  8,   80.,   1.5, 0.7,   30,   70, 100]

    parameters = {
        'growth_speed'        : {'min': 0.045,                          'max': 0.325},
        'start_of_season'     : {'min': max([start_of_season - 10, 0]), 'max': min([start_of_season + 50, 365])},
        'senescence_speed'    : {'min': 0.01,                           'max': 0.37},
        'end_of_season'       : {'min': growth_season_length-30,        'max': growth_season_length+30},
        'N'                   : {'min': 1,                              'max': 3},
        'Cab'                 : {'min': 40,                             'max': 100},
        'Cm'                  : {'min': 0.002,                          'max': 0.015},
        'Cw'                  : {'min': 0.01,                           'max': 0.06},
        'Lai'                 : {'min': 0.5,                            'max': 8},
        'Ala'                 : {'min': 60,                             'max': 80},
        'Cb'                  : {'min': 0,                              'max': 1.5},
        'soil_brightness'     : {'min': 0.1,                            'max': 0.7},
        'soil_shape_p1'       : {'min': 10,                             'max': 30},
        'soil_shape_p2'       : {'min': 10,                             'max': 70},
        'soil_volume_moisture': {'min': 2,                              'max': 100},
    }

    p_mins = [values['min'] for values in parameters.values()]
    p_maxs = [values['max'] for values in parameters.values()]

    # Generate reference samples
    arc_refs, pheo_samples, bio_samples, orig_bios, soil_samples = generate_ref_samples(p_mins, p_maxs, num_samples, angs, doys, crop_type)
    
    return arc_refs, pheo_samples, bio_samples, orig_bios, soil_samples


if __name__ == "__main__":

    doys = np.arange(1, 366, 5)
    angs = np.array([30,] * len(doys)), np.array([10,] * len(doys)), np.array([120,] * len(doys)) 
    
    num_samples = 10000
    start_of_season = 100
    growth_season_length = 45
    crop_type = 'maize'

    # Generate reference samples
    arc_refs, pheo_samples, bio_samples, orig_bios, soil_samples = generate_arc_refs(doys, start_of_season, growth_season_length, num_samples, angs, crop_type)

    max_lai = np.nanmax(orig_bios[4], axis=0)
    ndvi = (arc_refs[7] - arc_refs[3]) / (arc_refs[7] + arc_refs[3])
    max_ndvi = np.nanmax(ndvi, axis=0)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.plot(max_ndvi, max_lai/100, 'o', ms=5, alpha=0.1)
    plt.xlabel('Max NDVI')
    plt.ylabel('Max LAI (m$^2$/m$^2$)')

