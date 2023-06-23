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

