# ARC (Archetypes for Crop Monitoring from Earth Observation) - Technical Documentation

**Version:** 0.0.2
**Author:** Feng Yin, Department of Geography, University College London
**License:** GNU Affero General Public License v3.0
**Funding:** ESA/AU EO Africa R&D Facility
**Partners:** University College London (UK) and University of The Witwatersrand (South Africa)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Scientific Background](#2-scientific-background)
3. [Architecture and Data Flow](#3-architecture-and-data-flow)
4. [Module Reference](#4-module-reference)
5. [Data Files](#5-data-files)
6. [Key Algorithms](#6-key-algorithms)
7. [Dependencies](#7-dependencies)
8. [Installation and Usage](#8-installation-and-usage)

---

## 1. Overview

ARC is a Python package for estimating crop biophysical parameters from Sentinel-2 satellite imagery. It works by:

1. Downloading Sentinel-2 surface reflectance data via the [eof](https://github.com/profLewis/eof) (EO Fetch) package, which supports multiple sensors (Sentinel-2, Landsat, MODIS, VIIRS, Sentinel-3 OLCI) from multiple platforms (AWS Earth Search, CDSE, Planetary Computer, Google Earth Engine).
2. Generating a large ensemble of simulated Sentinel-2 reflectance spectra using a neural network emulator of the PROSAIL radiative transfer model, spanning a range of crop biophysical states, phenologies, soil backgrounds, and viewing geometries.
3. Matching the observed satellite spectra to the closest members of this simulated ensemble using approximate nearest-neighbour search.
4. Computing weighted-average posterior estimates of biophysical parameters (and their uncertainties) from the best-matching ensemble members.

The output is a per-pixel, per-date estimate of seven biophysical parameters (leaf structure, chlorophyll, dry matter, water content, LAI, leaf angle, brown pigment) with associated uncertainties, enabling spatiotemporal crop monitoring at field scale.

### Supported Crops

| Crop    | Model File       | Origin   |
|---------|------------------|----------|
| Maize   | `US_001.npz`     | US data  |
| Soybean | `US_005.npz`     | US data  |
| Wheat   | `US_024.npz`     | US data  |
| Rice    | `China_000.npz`  | China data |

---

## 2. Scientific Background

### 2.1 The Inverse Problem

Satellite sensors measure spectral reflectance at the top of the canopy. The forward problem is well understood: given plant biophysical properties, canopy structure, soil background, and illumination geometry, radiative transfer models such as PROSAIL can predict the expected reflectance. The inverse problem — inferring biophysical parameters from measured reflectance — is what ARC solves.

### 2.2 PROSAIL

PROSAIL is a widely-used combination of:
- **PROSPECT** (leaf-level model): simulates leaf reflectance and transmittance from biochemical and structural parameters (N, Cab, Cm, Cw, Car, Cbrown).
- **SAIL** (canopy-level model): simulates canopy bidirectional reflectance from leaf optical properties, LAI, leaf angle distribution (ALA), soil reflectance, and solar/view geometry.

Running PROSAIL millions of times is computationally expensive. ARC replaces the full model with a pre-trained neural network that maps 15 input parameters to 10 Sentinel-2 band reflectances, achieving the same output orders of magnitude faster.

### 2.3 The Archetype Approach

Rather than attempting a direct per-pixel numerical inversion (which is ill-posed and noisy), ARC adopts an ensemble-based approach:

1. **Generate archetypes**: Sample a large number (e.g., 100,000) of plausible crop parameter combinations spanning realistic ranges for a given crop type. For each sample, simulate the full temporal trajectory of Sentinel-2 reflectance.
2. **Match observations**: For each observed pixel, find the archetype members whose simulated spectra most closely match the observations.
3. **Weighted average**: Compute a weighted average of the biophysical parameters from the best-matching archetypes, inversely weighted by spectral distance.

This yields robust parameter estimates with built-in uncertainty quantification.

### 2.4 The Seven Biophysical Parameters

| Index | Parameter | Description | Physical Range | Integer Scale Factor |
|-------|-----------|-------------|----------------|---------------------|
| 0 | N | Leaf structure parameter | 1 – 3 | ×100 |
| 1 | Cab | Chlorophyll a+b content (µg/cm²) | 0 – 140 | ×100 |
| 2 | Cm | Dry matter content (g/cm²) | 0 – 0.04 | ×10000 |
| 3 | Cw | Equivalent water thickness (cm) | 0 – 0.1 | ×10000 |
| 4 | LAI | Leaf Area Index (m²/m²) | 0 – 10 | ×100 |
| 5 | ALA | Average Leaf Angle (degrees) | 20 – 90 | ×100 |
| 6 | Cbrown | Brown pigment content | 0 – 1.5 | ×1000 |

Parameters are stored as integers in the output tensors. To recover the physical value, divide by the scale factor (e.g., LAI = `post_bio_tensor[:, 4] / 100`).

---

## 3. Architecture and Data Flow

### 3.1 High-Level Pipeline

```
User Input
  ├── Field boundary (GeoJSON)
  ├── Date range (start_date, end_date)
  ├── Crop type + phenology (start_of_season, growth_season_length)
  └── Number of ensemble samples

        ↓

[Step 1] Sentinel-2 Data Retrieval (via eof package)
  ├── Load GeoJSON → compute centroid → MGRS tile
  ├── Query data source (AWS, CDSE, Planetary Computer, or GEE)
  ├── Download images, crop to field boundary, apply cloud mask
  └── Output: s2_refs (10 bands × dates × pixels), uncertainties (10%), angles

        ↓

[Step 2] Archetype Ensemble Generation (arc_sample_generator.py)
  ├── Load crop-specific model parameters (median temporal profiles)
  ├── Generate parameter samples via Sobol quasi-random sequence
  ├── Create temporal dynamics via double-sigmoid logistic growth curves
  ├── Scale biophysical parameters using crop model medians
  ├── Compute soil reflectance + viewing geometry normalization
  ├── Run NN forward model (PROSAIL emulator) on all samples
  └── Output: arc_refs (10 bands × dates × samples), parameter arrays

        ↓

[Step 3] Approximate K-Nearest Neighbour Search (approximate_KNN_search.py)
  ├── Partition time series into overlapping temporal segments
  ├── Compute segment medians for observations and archetypes
  ├── Build PyNNDescent index with weighted Euclidean distance
  └── Output: neighbour indices (k=300 per pixel)

        ↓

[Step 4] Data Assimilation (assimilate_jax.py)
  ├── For each pixel with valid observations:
  │   ├── Retrieve K=300 neighbour spectra from archetype ensemble
  │   ├── Sort by absolute spectral difference, select top 50
  │   ├── Compute uncertainty-weighted L2 distance
  │   ├── Assign inverse-distance weights (normalized to sum to 1)
  │   ├── Compute weighted mean of biophysical parameters
  │   └── Compute weighted standard deviation (uncertainty)
  └── Output: posterior parameter tensors + uncertainties

        ↓

[Step 5] Save Results
  └── NPZ file with: post_bio_tensor, post_bio_unc_tensor,
      scale_data, geotransform, CRS, mask, DOYs, mean_ref, best_candidate
```

### 3.2 Data Dimensions

| Variable | Shape | Description |
|----------|-------|-------------|
| `s2_refs` | (10, n_dates, n_pixels) | Observed S2 reflectance |
| `s2_uncs` | (10, n_dates, n_pixels) | 10% of reflectance |
| `arc_refs` | (10, n_dates, n_samples) | Simulated archetype spectra |
| `orig_bios` | (7, n_dates, n_samples) | Archetype biophysical parameters (integer-scaled) |
| `neighbours` | (n_pixels, 300) | KNN indices into archetype ensemble |
| `post_bio_tensor` | (n_pixels, 7, n_dates) | Posterior biophysical estimates |
| `post_bio_unc_tensor` | (n_pixels, 7, n_dates) | Posterior uncertainties |
| `mask` | (rows, cols) | Boolean; True where field boundary is outside / NaN |

---

## 4. Module Reference

### 4.1 `arc/__init__.py`

Exports the two main entry points:
- `generate_arc_refs()` — generate archetype ensemble (no satellite data needed)
- `arc_field()` — full pipeline from satellite data to biophysical parameters

### 4.2 `arc/field_processor.py` — Pipeline Orchestrator

**Function: `arc_field(s2_start_date, s2_end_date, geojson_path, start_of_season, crop_type, output_file_path, num_samples=10000, growth_season_length=45, S2_data_folder='./S2_data', plot=False, data_source='aws')`**

Orchestrates the full pipeline by calling, in sequence:
1. `eof.get_s2_data()` — download and preprocess satellite data (via [eof](https://github.com/profLewis/eof) package)
2. `generate_arc_refs()` — create the archetype ensemble
3. `get_neighbours()` — find nearest neighbours
4. `assimilate()` — compute posterior parameter estimates
5. `save_data()` — write results to disk

**Parameters:**
- `s2_start_date`, `s2_end_date`: Date range for Sentinel-2 acquisition (e.g., `"2022-07-15"`)
- `geojson_path`: Path to GeoJSON file defining the field boundary polygon
- `start_of_season`: Day of year when crop growth begins (e.g., 225 for Southern Hemisphere wheat)
- `crop_type`: One of `'maize'`, `'soybean'`, `'wheat'`, `'rice'`
- `output_file_path`: Where to save the `.npz` output file
- `num_samples`: Ensemble size (default 10,000; 100,000 recommended for production)
- `growth_season_length`: Duration of the active growing season in days
- `S2_data_folder`: Local cache directory for downloaded data
- `plot`: If `True`, generates diagnostic plots of spectral fits for 10 random pixels
- `data_source`: `'aws'`, `'cdse'`, `'planetary'`, `'gee'`, or `'auto'` (picks fastest available). Default: `'aws'`.

**Returns:** `(scale_data, post_bio_tensor, post_bio_unc_tensor, mask, doys)`

---

### 4.3 `arc/arc_sample_generator.py` — Archetype Ensemble Generation

This is the core scientific module. It generates a large ensemble of plausible crop growth trajectories and their corresponding Sentinel-2 reflectance spectra.

**Main Entry Point: `generate_arc_refs(doys, start_of_season, growth_season_length, num_samples, angs, crop_type)`**

**Parameters:**
- `doys`: Array of day-of-year values for which to generate spectra (e.g., `np.arange(1, 366, 5)`)
- `start_of_season`: Start DOY of the growing season
- `growth_season_length`: Length of the growing season in days
- `num_samples`: Number of ensemble members to generate
- `angs`: Tuple of `(VZA, SZA, RAA)` arrays — viewing/solar geometry per date
- `crop_type`: Crop name string

**Returns:** `(arc_refs, pheo_samples, bio_samples, orig_bios, soil_samples)`
- `arc_refs`: Simulated S2 reflectance, shape `(10, n_dates, n_samples)`
- `pheo_samples`: Phenology parameters per sample, shape `(n_samples, 4)`
- `bio_samples`: Biophysical scaling parameters per sample, shape `(n_samples, 7)`
- `orig_bios`: Full temporal biophysical parameters (integer-scaled), shape `(7, n_dates, n_samples)`
- `soil_samples`: Soil parameters per sample, shape `(n_samples, 4)`

#### Internal Workflow of `generate_ref_samples()`

1. **Load crop model** (`load_crop_model`): Reads median temporal profiles of the 7 biophysical parameters from a `.npz` file trained on crop-type-specific data.

2. **Adjust parameter ranges** (`adjust_parameters`): Normalizes the biophysical parameter ranges by the maximum of the crop model medians, so that sampling factors multiply the median profiles.

3. **Sobol sampling** (`generate_samples`): Uses the Sobol quasi-random sequence (via `scipy.stats.qmc.Sobol`) to draw samples that cover the parameter space more uniformly than pseudo-random sampling. The sampled dimensions are:

   | Dimensions | Parameters |
   |-----------|------------|
   | 0–3 | Phenology: growth_speed, start_of_season, senescence_speed, end_of_season |
   | 4–10 | Biophysical scaling: N, Cab, Cm, Cw, LAI, ALA, Cbrown |
   | 11–14 | Soil: brightness, shape_p1, shape_p2, volume_moisture |

4. **Logistic growth curves** (`sample_logistic`): Generates a double-sigmoid temporal profile for each sample using the phenology parameters. The logistic function is:

   ```
   f(t) = p0 - p1 * (σ₁(t) + σ₂(t) - 1)
   where σ₁(t) = 1/(1 + exp(p2*(t - p3)))
         σ₂(t) = 1/(1 + exp(-p4*(t - p5)))
   ```

   This creates a bell-shaped growth curve that rises (greenup) and falls (senescence).

5. **Temporal mapping** (`get_mapping`): Aligns each sample's logistic curve to a reference curve derived from the crop model medians. This uses interpolation to map between "phenological time" (normalized growth stage) and calendar time, allowing each sample to have a different phenological timing while maintaining realistic shapes.

6. **Scale samples** (`scale_samples`): Multiplies the sampled scaling factors by the crop model median temporal profiles to produce absolute biophysical values, clipping to physically valid ranges.

7. **Create input arrays** (`create_sample`):
   - Extracts biophysical parameters at the appropriate calendar dates using the temporal mapping
   - Computes Walthall BRDF coefficient for directional effects
   - Normalizes soil parameters (brightness, shape, moisture) to [0, 1] ranges
   - Assumes carotenoid content = chlorophyll / 4

8. **Normalize for NN** (`prepare_final_input`): Transforms the 15 input parameters into the normalized space expected by the neural network:
   - `N`: `(N - 1) / 2.5`
   - `Cab`: `exp(-Cab/100)`
   - `Car`: `exp(-Car/100)`
   - `Cbrown`: unchanged
   - `Cw`: `exp(-50 * Cw)`
   - `Cm`: `exp(-50 * Cm)`
   - `LAI`: `exp(-LAI/2)`
   - `ALA`: `cos(ALA in radians)`
   - `SZA`, `VZA`: cosines of angles
   - `RAA`: `(RAA % 360) / 360`
   - Soil params `p0–p3`: already normalized

9. **NN forward pass** (`predict_input_slices`): Runs the pre-trained neural network on all input samples in batches of ~300, producing 10-band reflectance predictions.

10. **Integer scaling** (`adjust_orig_bios`): Multiplies biophysical parameters by their scale factors (e.g., LAI × 100) and converts to integers for compact storage.

#### Key Helper Functions

| Function | Purpose |
|----------|---------|
| `logistic_function(p, t)` | Double-sigmoid growth curve with 6 parameters |
| `normalize_data(data)` | Min-max normalization to [0, 1] |
| `compute_reference_parameters(medians)` | Fits a reference logistic to the crop model's median LAI profile via `scipy.optimize.least_squares` with soft L1 loss |
| `compute_walthall_coef(sza, vza, raa)` | Walthall BRDF kernel coefficient for directional reflectance effects |

---

### 4.4 EO Data Retrieval (via `eof` package)

Earth Observation data retrieval is handled by the separate [eof](https://github.com/profLewis/eof) (EO Fetch) package, which ARC uses as a dependency.

**Supported sensors:** Sentinel-2, Landsat 8/9, MODIS, VIIRS, Sentinel-3 OLCI

**Supported platforms:**

| Platform | `data_source=` | Auth Required |
|----------|----------------|---------------|
| [AWS Earth Search](https://earth-search.aws.element84.com/v1) | `'aws'` | None |
| [CDSE](https://dataspace.copernicus.eu) | `'cdse'` | [S3 keys](https://eodata.dataspace.copernicus.eu) or login |
| [Planetary Computer](https://planetarycomputer.microsoft.com/) | `'planetary'` | `pip install planetary-computer` |
| [Google Earth Engine](https://earthengine.google.com/) | `'gee'` | [GEE account](https://signup.earthengine.google.com/) |

ARC uses `eof.get_s2_data()` for Sentinel-2 data, which returns an `S2Result` dataclass. The `eof` package also supports multi-sensor retrieval via `eof.get_eo_data()` which returns an `EOResult` with data resampled to 10m, footprint ID maps, and spectral response functions.

Using `data_source='auto'` picks the first available platform (defaults to AWS).

See the [eof documentation](https://github.com/profLewis/eof#readme) for full details on sensors, credentials, caching, and the API.

---

### 4.5 `arc/approximate_KNN_search.py` — Nearest Neighbour Search

Finds the closest archetype ensemble members for each observed pixel using approximate nearest neighbour search.

**Main Function: `get_neighbours(s2_refs, s2_uncs, arc_refs, doys, steps=10, k=300)`**

**Algorithm:**
1. **Temporal partitioning** (`partition_doy`): Divides the time series into `steps=10` overlapping segments with 8-day overlap on each side. This reduces dimensionality while preserving temporal structure.

2. **Median aggregation** (`partition_data`): Computes the median reflectance within each temporal segment for each band. This produces a compressed feature vector per pixel/sample.

3. **Index construction**: Builds a PyNNDescent approximate nearest neighbour index on the archetype ensemble's compressed feature vectors.

4. **Query**: Queries the index with the observed pixels' compressed feature vectors, returning the `k=300` nearest neighbours per pixel.

**Distance Metric:**
Uses a custom Numba-JIT compiled `non_negative_weighted_euclidean` distance:
```
d(x, y) = sqrt(Σ (x_i - y_i)² × w_i)  for all i where x_i >= 0 and y_i >= 0
```
Weights are `1 / mean_band_error²`, giving higher weight to bands with lower uncertainty. Negative values (used as NaN sentinel = -9999) are excluded from the distance computation.

---

### 4.6 `arc/assimilate_jax.py` — Data Assimilation

Performs the weighted ensemble averaging to produce posterior biophysical parameter estimates. Implemented with JAX for GPU compatibility and JIT compilation.

**Function: `assimilate(s2_refs, arc_refs, s2_errs, pheo_samples, bio_samples, soil_samples, orig_bios, neighbours, num_ens=50)`**

**Per-pixel algorithm:**
1. Retrieve the 300 pre-selected neighbour spectra from the archetype ensemble.
2. Compute spectral difference: `diff = observed - archetype` for all bands and dates.
3. Sort neighbours by total absolute difference: `Σ|diff|` across bands and dates.
4. Select the top `num_ens=50` best-matching ensemble members.
5. Compute uncertainty-weighted L2 distance: `d² = Σ (diff² × unc²)` across bands and dates.
6. Compute inverse-distance weights: `w_i = (1/d_i²) / Σ(1/d_j²)`.
7. Posterior estimates: weighted mean of biophysical parameters.
8. Posterior uncertainty: weighted standard deviation with Bessel's correction (`n/(n-1)`).

The function is vectorized across all pixels using `jax.vmap` and JIT-compiled with `@jax.jit`. Pixels with no valid observations receive zero-filled outputs.

**Returns:** Tuple of 7 arrays:
- `post_bio_tensor` — posterior biophysical parameters (integer-scaled)
- `post_bio_unc_tensor` — posterior uncertainties (integer-scaled)
- `post_bio_scale_tensor` — biophysical scaling factors
- `post_pheo_tensor` — posterior phenology parameters
- `post_soil_tensor` — posterior soil parameters
- `mean_ref` — weighted mean reference spectrum
- `best_candidate` — indices of the 50 best ensemble members

---

### 4.7 `arc/NN_predict_jax.py` — Neural Network PROSAIL Emulator

A pre-trained neural network that replaces the full PROSAIL radiative transfer model for fast forward simulation.

**Function: `predict(inputs, arrModel, cal_jac=False)`**

**Architecture:**
- Multi-layer feedforward neural network with ReLU activations
- Weights loaded from `foward_prosail_model_weights.npz`
- Input: 15 normalized parameters (shape `(15, n_samples)`)
- Output: 10 Sentinel-2 band reflectances (shape `(10, n_samples)`)
- The final layer has separate weight vectors for each output band

**Input Parameters (15 dimensions, normalized):**

| # | Parameter | Normalization |
|---|-----------|---------------|
| 0 | N | (N - 1) / 2.5 |
| 1 | Cab | exp(-Cab / 100) |
| 2 | Car | exp(-Car / 100) |
| 3 | Cbrown | raw value |
| 4 | Cw | exp(-50 × Cw) |
| 5 | Cm | exp(-50 × Cm) |
| 6 | LAI | exp(-LAI / 2) |
| 7 | ALA | cos(ALA) |
| 8 | SZA | cos(SZA) |
| 9 | VZA | cos(VZA) |
| 10 | RAA | (RAA % 360) / 360 |
| 11–14 | Soil p0–p3 | pre-normalized |

**Output Bands (10):**
B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12

When `cal_jac=True`, the function also computes Jacobian matrices (derivatives of outputs with respect to inputs) via manual backpropagation through the network layers.

The function `round_predict()` provides an optimization that deduplicates near-identical inputs (rounded to a specified number of decimal places) to avoid redundant forward passes.

---

### 4.8 `arc/robust_smoothing.py` — Robust Spline Smoothing

Implements iteratively reweighted penalized least-squares smoothing for time series, resistant to outliers.

**Two implementations are provided:**
- `robust_smooth()` — dense matrix version
- `robust_smooth_sp()` — sparse matrix version (more memory-efficient)

**Function: `robust_smooth(array, Warray, x, s, d, iterations=1, axis=0)`**

**Algorithm:**
1. Construct the d-th order difference matrix D from the irregular time grid `x`.
2. For each iteration:
   - Solve the penalized regression: `(W + s·D'D) z = W·y`, where W is the diagonal weight matrix, y is the data, and s controls smoothness.
   - For `d=1` (first-order penalty): solve via LAPACK tridiagonal solver `ptsv`.
   - For `d>1`: solve via LAPACK banded Cholesky solver `pbsv`.
   - Compute residuals and update weights using Tukey's bisquare function:
     ```
     u = |residual| / (1.4826 × MAD × √(1-h))
     w_new = (1 - (u/4.685)²)² × I(u < 4.685)
     ```
   where MAD is the Median Absolute Deviation and h is a leverage factor derived from the smoothness parameter.

This is essentially the Garcia (2010) robust smoothing algorithm adapted for irregular time grids.

---

### 4.9 `arc/BSM_soil.py` — Brightness-Shape-Moisture Soil Model

Implements the BSM soil reflectance model from Verhoef & Bach (2007), as used in the SCOPE model.

**Function: `BSM(B, lat, lon, SMp, BSM_paras)`**

**Parameters:**
- `B`: Soil brightness parameter
- `lat`, `lon`: Angles in the Geometric Soil Vector (GSV) coordinate system — these are NOT geographic coordinates, but parameters controlling the spectral shape of soil reflectance
- `SMp`: Soil moisture percentage
- `BSM_paras`: Tuple of `(GSV, nw, kw)` loaded from `BSM_paras.npz`

**Algorithm:**
1. Decompose the soil spectrum as a linear combination of three Geometric Soil Vectors, weighted by brightness and angular parameters.
2. Model the effect of soil moisture using a water film model:
   - Water film thickness follows a Poisson distribution with mean µ = (SMp - 5) / SMC
   - Each film thickness layer has its own transmittance and reflectance
   - Uses the `tav()` function (Stern's formula from Lekner & Dorf 1988) for interface reflectance

---

### 4.10 `arc/arc_util.py` — Utilities

| Function | Purpose |
|----------|---------|
| `calculate_ndvi(s2_refs)` | NDVI from B8A (index 7) and B04 (index 2) |
| `time_series_filter(array, udoys)` | Applies robust smoothing and returns a boolean mask where weight > 0.5 |
| `ndvi_filter(s2_refs, s2_uncs, doys, s2_angles)` | Filters time series using NDVI, B8A, and B04 smoothing (currently commented out in the main pipeline) |
| `save_data(...)` | Saves all outputs to a compressed NumPy `.npz` file |

---

## 5. Data Files

### 5.1 Crop Model Files (`arc/data/US_*.npz`, `China_*.npz`)

Each crop model `.npz` file contains a `meds` array: the median temporal profile of the 7 biophysical parameters across 365 days, derived from training data. These medians serve as the "baseline shape" that Sobol-sampled scaling factors modulate.

### 5.2 Neural Network Weights (`arc/data/foward_prosail_model_weights.npz`)

Contains `model_weights`: a list of numpy arrays representing the weights and biases of each layer in the PROSAIL emulator network. The network was trained offline to approximate the full PROSAIL model.

### 5.3 Soil Parameters (`arc/data/BSM_paras.npz`)

Contains:
- `GSV` — Geometric Soil Vectors (spectral basis functions for soil reflectance)
- `nw` — Refractive index of water (wavelength-dependent)
- `kw` — Absorption coefficient of water (wavelength-dependent)

### 5.4 Test Field Geometries (`arc/test_data/`)

GeoJSON files defining field boundary polygons for testing:
- `SF_field.geojson` — A wheat field near Johannesburg, South Africa
- Three fields in the Netherlands (various crops)

---

## 6. Key Algorithms

### 6.1 Sobol Quasi-Random Sampling

The Sobol sequence is a low-discrepancy quasi-random sequence that fills a multi-dimensional space more uniformly than pseudo-random numbers. ARC uses it via `scipy.stats.qmc.Sobol` to sample the 15-dimensional parameter space (4 phenology + 7 biophysical + 4 soil). The actual number of samples is rounded up to the nearest power of 2.

This is important because:
- Fewer samples are needed to adequately cover the space compared to random sampling
- Edge cases and extremes of parameter ranges are better represented
- The archetype ensemble is more diverse and representative

### 6.2 Double Sigmoid Phenology Model

Crop temporal dynamics are modelled as a double-sigmoid function with 6 parameters controlling:
- `p0`, `p1`: vertical offset and amplitude
- `p2`, `p3`: growth rate and inflection point of the greenup sigmoid
- `p4`, `p5`: senescence rate and inflection point of the senescence sigmoid

This produces a bell-shaped seasonal profile that captures the typical greenup-peak-senescence pattern of annual crops. Each sampled ensemble member gets its own phenology, creating temporal diversity in the archetype library.

### 6.3 Temporal Mapping via Interpolation

Different ensemble members may reach peak growth at different calendar dates. The temporal mapping step aligns all members to a common "phenological clock" by:
1. Fitting a reference logistic to the crop model's median LAI profile
2. Creating an interpolation function from normalized phenological stage to calendar day
3. For each ensemble member, mapping its logistic curve to equivalent calendar positions

This ensures that biophysical parameter values are correctly associated with the right growth stage at each calendar date.

### 6.4 Two-Stage Nearest Neighbour Selection

The matching process uses a two-stage approach for efficiency:

**Stage 1 (KNN, k=300):** PyNNDescent approximate nearest neighbour search on temporally-compressed feature vectors. This is fast but approximate, providing a shortlist of 300 candidates per pixel.

**Stage 2 (Assimilation, top 50):** Among the 300 candidates, the full spectral distance (all bands, all dates) is computed exactly. The top 50 are selected and inverse-distance-weighted.

### 6.5 Inverse Distance Weighting with Uncertainties

The final parameter estimates use weights:
```
w_i = 1/d_i² / Σ_j(1/d_j²)
```
where:
```
d_i² = Σ_bands Σ_dates (observed - archetype_i)² × uncertainty²
```

This gives higher weight to ensemble members that match better, and incorporates observation uncertainty so that noisier bands contribute less to the distance metric.

---

## 7. Dependencies

| Package | Purpose |
|---------|---------|
| `jax` | JIT-compiled numerical computing; GPU support; automatic differentiation |
| `numpy` | Core array operations |
| `scipy` | Optimization (`least_squares`), quasi-random sampling (`Sobol`), LAPACK solvers |
| `tqdm` | Progress bars for batch processing |
| `numba` | JIT compilation of the custom distance metric |
| `pynndescent` | Approximate nearest neighbour search with custom metrics |
| `eof` | Multi-sensor EO data retrieval (Sentinel-2, Landsat, MODIS, VIIRS, S3 OLCI) from multiple platforms (AWS, CDSE, Planetary Computer, GEE) |

---

## 8. Installation and Usage

### 8.1 Installation

```bash
pip install https://github.com/profLewis/ARC/archive/refs/heads/main.zip
```

### 8.2 Prerequisites

No authentication is needed for the default AWS data source (free, fast). For other sources, see the [eof credential setup guide](https://github.com/profLewis/eof#credential-management).

### 8.3 Generating an Archetype Ensemble (without satellite data)

```python
import arc
import numpy as np

doys = np.arange(1, 366, 5)
angs = (
    np.array([30] * len(doys)),   # VZA
    np.array([10] * len(doys)),   # SZA
    np.array([120] * len(doys)),  # RAA
)

s2_refs, pheo_samples, bio_samples, orig_bios, soil_samples = arc.generate_arc_refs(
    doys=doys,
    start_of_season=150,
    growth_season_length=45,
    num_samples=10000,
    angs=angs,
    crop_type='maize'
)
```

### 8.4 Full Field Processing Pipeline

```python
import arc

scale_data, post_bio_tensor, post_bio_unc_tensor, mask, doys = arc.arc_field(
    s2_start_date="2022-07-15",
    s2_end_date="2022-11-30",
    geojson_path="path/to/field.geojson",
    start_of_season=225,
    crop_type='wheat',
    output_file_path="output.npz",
    num_samples=100000,
    growth_season_length=45,
    S2_data_folder="./S2_data",
    data_source='auto',  # or 'aws', 'cdse', 'planetary', 'gee'
)

# Extract LAI time series (physical units)
lai = post_bio_tensor[:, 4].T / 100  # shape: (n_dates, n_pixels)
```

### 8.5 Interpreting Outputs

The `post_bio_tensor` has shape `(n_pixels, 7, n_dates)`. To get physical values:

```python
N       = post_bio_tensor[:, 0, :] / 100
Cab     = post_bio_tensor[:, 1, :] / 100     # µg/cm²
Cm      = post_bio_tensor[:, 2, :] / 10000   # g/cm²
Cw      = post_bio_tensor[:, 3, :] / 10000   # cm
LAI     = post_bio_tensor[:, 4, :] / 100     # m²/m²
ALA     = post_bio_tensor[:, 5, :] / 100     # degrees
Cbrown  = post_bio_tensor[:, 6, :] / 1000
```

To create spatial maps, use the `mask` array:
```python
lai_map = np.full(mask.shape, np.nan)
lai_map[~mask] = LAI[:, date_index]
```
