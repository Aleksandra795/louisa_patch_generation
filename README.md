# louisa_patch_generation

Create synchronized 2D or 3D patches from aligned medical images and masks.
The package supports two patch sizes with shared centers, point-based binary
labels, per-patch object IDs, background filtering, streaming, and HDF5 export.
It runs independently of any dataset or external pipeline.

This package was implemented as a part of the project: 

**LOUISA - Intelligent system for automatic identification of brain lacunes in magnetic resonance imaging scans.**

## Local installation

From the cloned repository directory:

```bash
python -m pip install -e .
```

Requires Python 3.10+. Runtime dependencies are listed in `requirements.txt`;
installation reads that file automatically.

## Minimal example

```python
import nibabel as nib
import numpy as np
from louisa_patch_generation import PatchGenerator


if __name__ == "__main__":
    image = nib.Nifti1Image(np.ones((96, 96, 1), dtype=np.float32), np.eye(4))
    image.header.set_xyzt_units("mm")
    
    # Initialize PatchGenerator
    pg = PatchGenerator(
        {"FLAIR": image},
        annotations=[(48, 48, 0)],
        annotations_ids=[1],
    )

    # Preprocess images
    prepared_images, points = pg.preprocess(target_spacing=(1, 1, 1))
    
    # Generate paches 2D
    patches = pg.generate_patches(
        mode="2d", 
        patch_sizes=(32, 48), 
        stride=12, 
        label_border=5,
    )

    # Postprocess images
    patches = pg.postprocess(max_zero_percent=75, remove_inconsistent=True)

    print(patches["FLAIR"][32].shape)       # (N, 32, 32)
    print(patches["annotations_ids"][32][0])
    # pg.save_hdf5("patches.h5")           # Choose a new filename to save.
```

For 3D patches, supply a 3D volume large enough for the requested cubes and set
`mode="3d"`. Arrays then have shape `(N, side, side, side)`. A single 2D input
slice must still be stored as a 3D image with one slice, as above.

## Main methods and parameters

| Method | Purpose and main parameters |
| --- | --- |
| `PatchGenerator(images, mask_names=None, annotations=None, annotations_ids=None)` | `images` maps names to NiBabel images; `mask_names` names discrete masks. Point annotations have shape `(K, 3)`; optional positive integer IDs have length `K`. |
| `preprocess(target_spacing=(1, 1, 1))` | Resample to mm spacing: linear for images, nearest-neighbor for masks. Return images and scaled points. |
| `generate_patches(...)` | Choose `mode`, one or two `patch_sizes`, `stride`, `label_border`, and `slice_axis=2` for 2D. `generator=False` returns arrays; `True` streams one position at a time. |
| `postprocess(max_zero_percent=100, remove_inconsistent=False, reference=None)` | Remove synchronized positions based on exact-zero background and optionally disagreeing size labels. The default reference is the first non-mask image. |
| `save_hdf5(path)` | Save the current result, parameters and geometry. Existing files are refused. |

Inputs must already be anatomically registered and share their shape and affine.
Preprocessing preserves orientation and origin; it does not register images,
remove the skull or normalize intensities. Annotations use zero-based voxel
indices in input-array axis order, before preprocessing, and may be fractional.
`annotations=None` means unknown labels; `[]` means a known absence of points.

Two sizes must have equal parity. Their largest size defines fitting windows;
there is no padding or forced terminal window. `label_border` is an inward margin
in voxels, independently for each size; in 2D it applies only within the slice
plane. `stride` may be one integer or three axis-specific steps, including the
2D slice axis. Background equality is accepted: `max_zero_percent=75` rejects
values greater than 75%. All-mask inputs skip background filtering.

## Results

Index a patch with `patches["FLAIR"][32][i]`. Shared centers are stored in
`patches["coordinates"][i]`, and binary labels in `patches["label"][32][i]`.
If point IDs are supplied, `patches["annotations_ids"][32][i]` contains unique
ascending IDs, such as `[1]` or `[1, 2]`; `[0]` means no point meets the label
criterion. IDs are evaluated separately for each size and remain aligned after
filtering. Repeated point IDs may represent one object. Combine object IDs with
a subject identifier when comparing multiple subjects.

HDF5 stores `/images/{name}/{size}`, `/coordinates`, `/label/{size}` and
`/annotations`. Optional IDs use `/annotations_ids` for point IDs and
`/patch_annotations_ids/{size}` for per-patch variable-length ID arrays.
Without point IDs, the extra result key and ID datasets are absent. When saving
a generator, the full result is replayed, including patches already previewed.

## Visual 2D notebook

```bash
python -m pip install -e ".[notebook]"
jupyter lab examples/patches_2d.ipynb
```

Run all cells in [examples/patches_2d.ipynb](examples/patches_2d.ipynb).
Its kernel starts in `examples/`, so the data path is `data/synthetic_slice.npz`.
The included 128 x 128 x 1 image and ventricular mask are entirely synthetic. 
The notebook demonstrates
patches with and without a synthetic lacune, background filtering, object IDs
and an HDF5 round trip. Notebook dependencies are optional and are not required
by the library itself.

## Tests

```bash
python -m unittest discover -s tests -v
```
