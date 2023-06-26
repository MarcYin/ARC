# $⌒_⌣⌒$ *Archetypes*

Feng Yin

Department of Geography, UCL

ucfafyi@ucl.ac.uk


This is the python module for the archetype work.

```
pip install https://github.com/MarcYin/ARC/archive/refs/heads/main.zip
```


## Generating archtype ensemble

Define the necessary parameters:

- `doys`: An array of day of year values representing the dates for which you want to generate reference samples. For example, np.arange(1, 366, 5) generates samples for every 5th day of the year.
- `angs`: A tuple containing the angles (vza, sza, raa) representing the view zenith angle, solar zenith angle, and relative azimuth angle, respectively. The angles can be single values or arrays with the same length as doys.
- `num_samples`: The number of reference samples to generate.
- `start_of_season`: The starting day of the crop growth season. This indicates the day of the year when the crop growth starts.
- `growth_season_length`: The length of the crop growth season in days.
- `crop_type`: The type of crop for which to generate the reference samples. For example, 'maize'.


The generate_arc_s2_refs function generates reference samples for Sentinel-2 observations based on the specified parameters. It returns several arrays:

- `s2_refs`: Reflectance spectra for different Sentinel-2 bands: 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'.
- `pheo_samples`: Phenology parameters samples.
- `bio_samples`: Biophysical parameter scaling parameters.
- `orig_bios`: Time series of biophysical parameter samples.
- `soil_samples`: Soil parameter samples.


```python
import arc
import numpy as np

doys = np.arange(1, 366, 5)
angs = np.array([30,] * len(doys)), np.array([10,] * len(doys)), np.array([120,] * len(doys)) 

num_samples = 10000
start_of_season = 150
growth_season_length = 45
crop_type = 'maize'

# Generate reference samples
s2_refs, pheo_samples, bio_samples, orig_bios, soil_samples = arc.generate_arc_refs(doys, start_of_season, growth_season_length, num_samples, angs, crop_type)


# Plot the relationship between maximum NDVI and maximum LAI:

max_lai = np.nanmax(orig_bios[4], axis=0)
ndvi = (s2_refs[7] - s2_refs[3]) / (s2_refs[7] + s2_refs[3])
max_ndvi = np.nanmax(ndvi, axis=0)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.plot(max_ndvi, max_lai/100, 'o', ms=5, alpha=0.1)
plt.xlabel('Max NDVI')
plt.ylabel('Max LAI (m$^2$/m$^2$)')
plt.show()
```

## Testing archetype solver 

This package contains a function to solve the biophysical parameters with time series of Sentinel-2 (S2) observations. The S2 surface reflectance is downloaded from GEE with an assumed uncertainty of 10%. 

The function `arc_field` takes the following parameters:

- `start_date`: The starting date of the S2 observations.
- `end_date`: The ending date of the S2 observations.
- `geojson_path`: The path to the GeoJSON file containing the field boundary.
- `start_of_season`: The day of year (DOY) of the start of the crop growth season.
- `crop_type`: The type of crop for which to generate the reference samples. For example, 'maize'.
- `output_file_path`: The path for saving the output file.
- `num_samples`: The number of reflectance samples to generate.
- `growth_season_length`: The length of the crop growth season in days.
- `S2_data_folder`: The folder used to store S2 data.

And returns the following:
- `scale_data`: The scaling parameters used to scale the archetypes.
- `post_bio_tensor`: The posterior biophysical parameters tensor.
- `post_bio_unc_tensor`: The posterior biophysical parameters uncertainty tensor.
- `mask`: The mask of the field boundary for the tensor.
- `doys`: The DOYs of the S2 observations.

The shape of `post_bio_tensor` should be (number_doys, 7, number_valid_pixels), where 7 is the number of biophysical parameters (`N, cab, cm, cw, lai, ala, cbrown`). The shape of `post_bio_unc_tensor` should be (number_doys, 7, number_valid_pixels, 7), where 7 is the number of biophysical parameters. 

`post_bio_tensor` table with scales:

| Index | Parameter | Scale |
| --- | --- | --- |
| 0 | N | 1/100. |
| 1 | cab | 1/100. |
| 2 | cm | 1/10000. |
| 3 | cw | 1/10000. |
| 4 | lai | 1/100. |
| 5 | ala | 1/100. |
| 6 | cbrown | 1/1000. |
|||

### Full example:


#### ⚠️⚠️⚠️ Google Earth Engine authentication before using ARC

1. Create a Google Earth Engine account and sign in to [Google Earth Engine](https://earthengine.google.com/).
 
2. You can run the following command to set up the Google Earth Engine authentication:

```bash
earthengine authenticate --auth_mode=notebook
```
You should see the following message in the terminal:
```console
To authorize access needed by Earth Engine, open the following URL in a web browser and follow the instructions. If the web browser does not start automatically, please manually browse the URL below.
        https://code.earthengine.google.com/client-auth?scopes=https...
The authorization workflow will generate a code, which you should paste in the box below.
Enter verification code:
```

4. Copy the URL to your browser and sign in to your Google Earth Engine account. You will be given a verification code. Copy the verification code and paste it into the terminal. You should see the following message:
```bash
Successfully saved authorization token.
```

5. You can now exit the shell and stop the container by typing `exit` in the terminal. 
   
6. You can now run the following command to test the solver over one [South African field](https://github.com/MarcYin/ARC/blob/main/arc/test_data/SF_field.geojson):


```python
import arc
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# Constants
START_OF_SEASON = 225
CROP_TYPE = 'wheat'
NUM_SAMPLES = 100000
GROWTH_SEASON_LENGTH = 45
LAZY_EVALUATION_STEP = 100
ALPHA = 0.8
LINE_WIDTH = 2
start_date="2022-07-15"
end_date="2022-11-30"

def main():
    """Main function to execute the Arc field processing and plotting"""
    
    arc_dir = os.path.dirname(os.path.realpath(arc.__file__))
    geojson_path = f"{arc_dir}/test_data/SF_field.geojson"
    S2_data_folder = Path.home() / f"Downloads/{Path(geojson_path).stem}"
    S2_data_folder.mkdir(parents=True, exist_ok=True)
    
    scale_data, post_bio_tensor, post_bio_unc_tensor, mask, doys = arc.arc_field(
        start_date, 
        end_date, 
        geojson_path, 
        START_OF_SEASON, 
        CROP_TYPE, 
        f'{S2_data_folder}/SF_field.npz', 
        NUM_SAMPLES, 
        GROWTH_SEASON_LENGTH, 
        str(S2_data_folder),
        plot=True
    )

    plot_lai_over_time(doys, post_bio_tensor)
    plot_lai_maps(doys, post_bio_tensor, mask)

def plot_lai_over_time(doys: np.array, post_bio_tensor: np.array):
    """Plot LAI over time"""
    
    plt.figure(figsize=(12, 6))
    plt.plot(doys, post_bio_tensor[::LAZY_EVALUATION_STEP, 4,].T / 100, '-',  lw=LINE_WIDTH, alpha=ALPHA)
    plt.ylabel('LAI (m2/m2)')
    plt.xlabel('Day of year')
    plt.show()

def plot_lai_maps(doys: np.array, post_bio_tensor: np.array, mask: np.array):
    """Plot LAI maps"""
    
    lai = post_bio_tensor[:, 4].T / 100
    nrows = int(len(doys) / 5) + int(len(doys) % 5 > 0)
    fig, axs = plt.subplots(ncols=5, nrows=nrows, figsize=(20, 4*nrows))
    axs = axs.ravel()

    for i in range(len(doys)):
        lai_map = np.zeros(mask.shape) * np.nan
        lai_map[~mask] = lai[i]
        im = axs[i].imshow(lai_map, vmin=0, vmax=7)
        fig.colorbar(im, ax=axs[i], shrink=0.8, label='LAI (m2/m2)')
        axs[i].set_title('DOY: %d' % doys[i])
    
    # remove empty plots
    for i in range(len(doys), len(axs)):
        axs[i].axis('off')
    plt.show()

if __name__ == "__main__":
    main()

```