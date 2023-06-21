# ⌒*Archetype*⌒

Feng Yin

Department of Geography, UCL

ucfafyi@ucl.ac.uk


This is the python module for the archetype work.

```
pip install https://github.com/MarcYin/ARC/archive/refs/heads/main.zip
```


Example:

```python
import arc
import numpy as np

doys = np.arange(1, 366, 5)
angs = np.array([30,] * len(doys)), np.array([10,] * len(doys)), np.array([120,] * len(doys)) 

num_samples = 10000
start_of_season = 100
growth_season_length = 45
crop_type = 'maize'

# Generate reference samples
s2_refs, pheo_samples, bio_samples, orig_bios, soil_samples = arc.generate_arc_s2_refs(doys, start_of_season, growth_season_length, num_samples, angs, crop_type)

max_lai = np.nanmax(orig_bios[4], axis=0)
ndvi = (s2_refs[7] - s2_refs[3]) / (s2_refs[7] + s2_refs[3])
max_ndvi = np.nanmax(ndvi, axis=0)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.plot(max_ndvi, max_lai/100, 'o', ms=5, alpha=0.1)
plt.xlabel('Max NDVI')
plt.ylabel('Max LAI (m$^2$/m$^2$)')
```
