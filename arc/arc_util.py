import numpy as np

def save_data(file_path, post_bio_tensor, post_bio_unc_tensor, dat, geotransform, crs, mask, doys):
    """
    Save data to a npz file.

    Args:
        file_path (str): The path of the npz file.
        post_bio_tensor (np.array): The post bio tensor.
        post_bio_unc_tensor (np.array): The post bio unc tensor.
        dat (np.array): The data array.
        geotransform (np.array): The geotransform.
        crs (str): The CRS.
        mask (np.array): The mask.
    """
    np.savez(file_path, post_bio_tensor=post_bio_tensor, post_bio_unc_tensor=post_bio_unc_tensor, dat=dat, geotransform=geotransform, crs=crs, mask=mask, doys=doys)
