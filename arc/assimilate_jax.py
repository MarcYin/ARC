import jax
import jax.numpy as jnp
from jax import jit
from jax.lax import div, cond, dynamic_slice
from functools import partial


@partial(jax.jit, static_argnums=(8,))
def assimilate(s2_refs, arc_refs, s2_errs, pheo_samples, bio_samples, soil_samples, orig_bios, neighbours, num_ens=50):
    """
    Assimilate different sources of data using a weighted averaging scheme.

    Args:
        s2_refs: S2 reflectance values for assimilation.
        arc_refs: Secondary reference values for assimilation.
        s2_errs: The errors in band values.
        pheo_samples: The samples for the pheo component.
        bio_samples: The samples for the bio component.
        soil_samples: The samples for the soil component.
        orig_bios: The original bio values.
        neighbours: The neighbour indices.
        num_ens: The number of ensemble members for assimilation (default is 50).

    Returns:
        post_bio_tensor: The post assimilation bio values.
        post_bio_unc_tensor: The uncertainty in post assimilation bio values.
        post_bio_scale_tensor: The scaled post assimilation bio values.
        post_pheo_tensor: The post assimilation pheo values.
        post_soil_tensor: The post assimilation soil values.
    """
    n_samples = s2_refs.shape[2]
    orig_bios_shape = orig_bios.shape

    orig_bios_dim1 = orig_bios.shape[0]
    orig_bios_dim2 = orig_bios.shape[1]

    # Mask to check if there is any valid value in the reference
    mask = jnp.isfinite(s2_refs).any(axis=(0, 1))

    def process_valid_index(i):
        ann_neighbours = neighbours[i]
        en_refs = arc_refs[:, :, ann_neighbours]
        s2_ref = s2_refs[:, :, i]
        unc = s2_errs[:, :, i]
        diff = s2_ref[..., None] - en_refs

        # Sorting on the basis of minimum absolute difference
        sorted_indices = jnp.argsort(jnp.nansum(jnp.abs(diff), axis=(0,1)))

        # Selecting ensemble members based on minimum difference
        ens_best_candidate = dynamic_slice(sorted_indices, (0,), (num_ens,))
        l2_distance = jnp.nansum(diff[:, :, ens_best_candidate]**2 * unc[:, :, None]**2, axis=(0, 1))

        # Calculating weights inversely proportional to the l2_distance
        best_candidate = ann_neighbours[ens_best_candidate]
        weight = div(1., l2_distance)
        weight = div(weight, weight.sum())

        # Calculating weighted means
        mean_pheo = jnp.sum(pheo_samples[best_candidate] * weight[:, None], axis=0)
        mean_soils = jnp.sum(soil_samples[best_candidate] * weight[:, None], axis=0)
        mean_bio_scales = jnp.sum(bio_samples[best_candidate] * weight[:, None], axis=0)

        # Calculating post assimilation bios and their uncertainty
        post_bios = jnp.sum(orig_bios[:, :, best_candidate] * weight[None, None], axis=(2)).astype(jnp.int32)
        post_bios_unc = jnp.sqrt(jnp.sum((orig_bios[:, :, best_candidate] - post_bios[:, :, None])**2 * weight[None, None] * num_ens / (num_ens-1), axis=2)).astype(jnp.int32)

        return post_bios, post_bios_unc, mean_bio_scales, mean_pheo, mean_soils

    # def process_invalid_index(i):
    #     zeros = lambda shape: jnp.zeros(shape, dtype=jnp.int32)
    #     return zeros(orig_bios_shape), zeros(orig_bios_shape), zeros(bio_samples.shape[1]), zeros(pheo_samples.shape[1]), zeros(soil_samples.shape[1])

    def process_invalid_index(i):
        post_bios = jnp.zeros((orig_bios_dim1, orig_bios_dim2), dtype=jnp.int32)
        post_bios_unc = jnp.zeros((orig_bios_dim1, orig_bios_dim2), dtype=jnp.int32)
        mean_bio_scales = jnp.zeros(bio_samples.shape[1])
        mean_pheo = jnp.zeros(pheo_samples.shape[1])
        mean_soils = jnp.zeros(soil_samples.shape[1])

        return post_bios, post_bios_unc, mean_bio_scales, mean_pheo, mean_soils
    
    def single_sample(i):
        return cond(mask[i], i, process_valid_index, i, process_invalid_index)

    results = jax.vmap(single_sample)(jnp.arange(n_samples))

    return tuple(jnp.stack(result) for result in results)
