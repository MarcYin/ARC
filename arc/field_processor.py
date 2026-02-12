import numpy as np
from arc.arc_util import save_data
from arc.assimilate_jax import assimilate
from arc.approximate_KNN_search import get_neighbours
from arc.arc_sample_generator import generate_arc_refs


def _get_data_reader(data_source):
    """
    Return the get_s2_official_data function for the requested data source.

    Uses lazy imports so that GEE (ee.Initialize) is only triggered when
    data_source='gee', and pystac_client is only imported for 'cdse'.
    """
    if data_source == 'cdse':
        from arc.s2_cdse_reader import get_s2_official_data
        return get_s2_official_data
    elif data_source == 'gee':
        from arc.s2_data_reader import get_s2_official_data
        return get_s2_official_data
    elif data_source == 'aws':
        from arc.s2_aws_reader import get_s2_official_data
        return get_s2_official_data
    elif data_source == 'planetary':
        from arc.s2_planetary_reader import get_s2_official_data
        return get_s2_official_data
    elif data_source == 'auto':
        from arc.credentials import select_data_source
        resolved = select_data_source()
        print(f"Auto-selected data source: {resolved}")
        return _get_data_reader(resolved)
    else:
        raise ValueError(
            f"Unknown data_source '{data_source}'. "
            f"Must be 'cdse', 'gee', 'aws', 'planetary', or 'auto'."
        )


def arc_field(s2_start_date, s2_end_date, geojson_path, start_of_season,
              crop_type, output_file_path, num_samples=10000, growth_season_length=45,
              S2_data_folder='./S2_data', plot=False, data_source='cdse'):
    """
    Performs the ARC Field pipeline which includes reading satellite data, generating samples, searching for neighbours,
    assimilating data, and saving the resulting data.

    Args:
        s2_start_date, s2_end_date : Strings representing the start and end dates for the satellite data.
        geojson_path : Path to the GeoJSON file.
        start_of_season : Start date of the growth season.
        crop_type : Type of the crop.
        output_file_path : File path to save the resulting data.
        num_samples : Number of samples to generate (default is 1,000,000).
        growth_season_length : Length of the growth season (default is 45).
        S2_data_folder : Directory where satellite data is stored.
        data_source : Data source for S2 imagery: 'cdse', 'gee', 'aws',
            'planetary', or 'auto' (picks fastest available). Default: 'cdse'.

    Returns:
        None. The results are saved to the output_file_path.
    """
    # Read satellite data
    get_s2_data = _get_data_reader(data_source)
    s2_refs, s2_uncs, s2_angles, doys, mask, geotransform, crs = get_s2_data(
        s2_start_date, s2_end_date, geojson_path, S2_data_folder
    )

    s2_refs = s2_refs[:, :, ~mask].transpose(1, 0, 2)
    s2_uncs = s2_uncs[:, :, ~mask].transpose(1, 0, 2)
    # print("S2_refs shape: ", s2_refs.shape)

    # Generate ARC samples
    arc_refs, pheo_samples, bio_samples, orig_bios, soil_samples = generate_arc_refs(
        doys, start_of_season, growth_season_length, num_samples, s2_angles, crop_type
    )

    # Get neighbours
    neighbours = get_neighbours(s2_refs, s2_uncs, arc_refs, doys)
    
    if plot:
        random_inds = np.random.choice(s2_refs.shape[-1], 10)
        for i in random_inds:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(24, 6))
            plt.plot(arc_refs[:, :, neighbours[i]].reshape(-1, 300), '-', color='b', lw=1, alpha=0.1)
            plt.plot(s2_refs[:, :, i].ravel(), 'o', color='red')
            vlines = [len(doys) * i for i in range(1, 10)]
            plt.vlines(vlines, 0, 0.6, color='black', lw=1, ls='--')
            text_locs = [len(doys) * i - 10 for i in range(1, 11)]
            bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
            plt.xticks(text_locs, bands)
            plt.show()

    # Assimilate data
    post_bio_tensor, post_bio_unc_tensor, post_bio_scale_tensor, post_pheo_tensor, post_soil_tensor, mean_ref, best_candidate = assimilate(
        s2_refs, arc_refs, s2_uncs, pheo_samples, bio_samples, soil_samples, orig_bios, neighbours
    )

    # post_bio_tensor, post_bio_unc_tensor, post_bio_scale_tensor, post_pheo_tensor, post_soil_tensor = assimilate(
    #     s2_refs, arc_refs, s2_uncs, pheo_samples, bio_samples, soil_samples, orig_bios, neighbours
    # )

    # Concatenate tensors
    scale_data = np.concatenate([post_bio_scale_tensor, post_pheo_tensor, post_soil_tensor], axis=1)

    # Save resulting data
    save_data(output_file_path, post_bio_tensor, post_bio_unc_tensor, scale_data, geotransform, crs, mask, doys, mean_ref, best_candidate)
    return scale_data, post_bio_tensor, post_bio_unc_tensor, mask, doys



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    # # Constants
    # START_OF_SEASON = 225
    # CROP_TYPE = 'wheat'
    # NUM_SAMPLES = 100000
    # GROWTH_SEASON_LENGTH = 45

    # start_date = "2022-07-15"
    # end_date = "2022-11-30"
    # geojson_path = "test_data/SF_field.geojson"

    # Constants
    START_OF_SEASON = 170
    CROP_TYPE = 'wheat'
    NUM_SAMPLES = 100000
    GROWTH_SEASON_LENGTH = 60

    start_date = "2021-05-15"
    end_date = "2021-10-01"
    geojson_path = "test_data/anny_cuypers_achter_stal_geometry.geojson"
    geojson_path = "test_data/maria_van_geldorp_achter_aardbei_geometry.geojson"
    

    S2_data_folder = Path.home() / f"Downloads/{Path(geojson_path).stem}"
    S2_data_folder.mkdir(parents=True, exist_ok=True)
    
    scale_data, post_bio_tensor, post_bio_unc_tensor, mask, doys = arc_field(
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

    plt.figure(figsize=(12, 6))
    LAZY_EVALUATION_STEP = 100
    ALPHA = 0.8
    LINE_WIDTH = 2
    plt.plot(doys, post_bio_tensor[::LAZY_EVALUATION_STEP, 4,].T / 100, '-',  lw=LINE_WIDTH, alpha=ALPHA)
    plt.ylabel('LAI (m2/m2)')
    plt.xlabel('Day of year')
    plt.show()

    lai = post_bio_tensor[:, 4].T / 100
    nrows = int(len(doys) / 5) + int(len(doys) % 5 > 0)
    fig, axs = plt.subplots(ncols = 5, nrows = nrows, figsize = (20, 4 * nrows))
    axs = axs.ravel()
    for i in range(len(doys)):
        lai_map = np.zeros(mask.shape) * np.nan
        lai_map[~mask] = lai[i]
        im = axs[i].imshow(lai_map, vmin = 0, vmax = 7)
        fig.colorbar(im, ax = axs[i], shrink = 0.8, label = 'LAI (m2/m2)')
        axs[i].set_title('DOY: %d'%doys[i])
    # remove empty plots
    for i in range(len(doys), len(axs)):
        axs[i].axis('off')
    plt.show()
