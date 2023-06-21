# ⌒*Archetype*⌒

Feng Yin

Department of Geography, UCL

ucfafyi@ucl.ac.uk


This is the python module for the archetype work.

```
pip install https://github.com/MarcYin/ARC/archive/refs/heads/main.zip
```


Example:


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
s2_refs, pheo_samples, bio_samples, orig_bios, soil_samples = arc.generate_arc_s2_refs(doys, start_of_season, growth_season_length, num_samples, angs, crop_type)


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
