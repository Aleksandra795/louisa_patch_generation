import logging
import json
import platform
from pathlib import Path
from itertools import islice
from importlib.metadata import version as dependency_version
import h5py
import nibabel
import numpy as np


def _json_value(value):
    """Convert NumPy values and paths used in metadata to JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON.")


def _validate_bundle(bundle, image_names, sizes, mode, dtypes, 
                     labels_available, annotations_ids=None):
    """Validate one dense result or one streamed group and return its row count."""
    expected = set(image_names) | {"coordinates", "label", "mode"}
    if annotations_ids is not None:
        expected.add("annotations_ids")
    if not isinstance(bundle, dict) or set(bundle) != expected:
        raise ValueError("A patch result must contain all image names, "
                         "coordinates, label and mode.")
    if bundle["mode"] != mode:
        raise ValueError("All saved patch results must use the configured mode.")
    coordinates = np.asarray(bundle["coordinates"])
    if (coordinates.ndim != 2 or coordinates.shape[1] != 3 
        or coordinates.dtype.kind not in "biuf"):
        raise ValueError("coordinates must be a real numeric array with shape (N, 3).")
    if not np.isfinite(coordinates).all():
        raise ValueError("Saved patch coordinates must be finite.")
    count = coordinates.shape[0]
    dimensions = 3 if mode == "3D" else 2
    for name in image_names:
        patches = bundle[name]
        if not isinstance(patches, dict) or set(patches) != set(sizes):
            raise ValueError("Each image must contain every configured patch size.")
        for size in sizes:
            array = np.asarray(patches[size])
            if array.shape != (count,) + (size,) * dimensions:
                raise ValueError("Patch arrays must have N first, "
                                 "followed by the spatial dimensions.")
            if array.dtype != dtypes[name]:
                raise ValueError("Patch array dtypes must match their "
                                 "prepared source images.")
    labels = bundle["label"]
    if not labels_available:
        if labels is not None:
            raise ValueError("Unannotated results must have label=None.")
    else:
        if not isinstance(labels, dict) or set(labels) != set(sizes):
            raise ValueError("Annotated results require a separate label "
                             "array for every size.")
        for size in sizes:
            array = np.asarray(labels[size])
            if (array.shape != (count,) or array.dtype.kind not in "biu" 
                or not np.isin(array, [0, 1]).all()):
                raise ValueError("Each label array must contain N integer "
                                 "or boolean zero/one values.")
    if annotations_ids is not None:
        ids_by_size = bundle["annotations_ids"]
        if not isinstance(ids_by_size, dict) or set(ids_by_size) != set(sizes):
            raise ValueError("Patch annotation IDs require every configured size.")
        for size in sizes:
            values = np.asarray(ids_by_size[size], dtype=object)
            if values.shape != (count,):
                raise ValueError("Patch annotation IDs must contain N ID arrays.")
            for row, ids in enumerate(values):
                ids = np.asarray(ids)
                if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu":
                    raise ValueError("Each patch requires a nonempty integer ID array.")
                if labels[size][row] == 0:
                    if not np.array_equal(ids, [0]):
                        raise ValueError("Negative patches must have annotation IDs [0].")
                elif np.any(ids == 0) or not np.isin(ids, annotations_ids).all():
                    raise ValueError("Positive patches require IDs from the supplied annotations.")
    return count


def _dense_batches(patches, image_names, sizes, count):
    """Yield views of at most eight groups from a dense N-first result."""
    for first in range(0, count, 8):
        last = min(first + 8, count)
        batch = {name: {size: patches[name][size][first:last] for size in sizes}
                 for name in image_names}
        batch["coordinates"] = patches["coordinates"][first:last]
        batch["label"] = None if patches["label"] is None else {
            size: patches["label"][size][first:last] for size in sizes}
        batch["mode"] = patches["mode"]
        if "annotations_ids" in patches and "annotations_ids" not in image_names:
            batch["annotations_ids"] = {
                size: patches["annotations_ids"][size][first:last] for size in sizes}
        yield batch


def _append_batch(output, bundles, image_names, sizes, mode, dtypes, 
                  labels_available, annotations_ids=None):
    """Validate and append a bounded set of bundles while preserving row alignment."""
    counts = [_validate_bundle(bundle, image_names, sizes, mode, dtypes, 
                               labels_available, annotations_ids)
              for bundle in bundles]
    count = sum(counts)
    if not count:
        return
    first = output["coordinates"].shape[0]
    last = first + count
    coordinates = output["coordinates"]
    coordinates.resize(last, axis=0)
    coordinates[first:last] = (np.concatenate([bundle["coordinates"] 
                                               for bundle in bundles], axis=0))
    if labels_available:
        for size in sizes:
            dataset = output[f"label/{size}"]
            dataset.resize(last, axis=0)
            dataset[first:last] = (np.concatenate([bundle["label"][size] 
                                                   for bundle in bundles]))
    if annotations_ids is not None:
        for size in sizes:
            dataset = output[f"patch_annotations_ids/{size}"]
            dataset.resize(last, axis=0)
            offset = first
            for bundle in bundles:
                for ids in bundle["annotations_ids"][size]:
                    dataset[offset] = ids
                    offset += 1
    for name in image_names:
        for size in sizes:
            dataset = output[f"images/{name}/{size}"]
            dataset.resize(last, axis=0)
            dataset[first:last] = (np.concatenate([bundle[name][size] 
                                                   for bundle in bundles], axis=0))


def _save_hdf5(path, patches, images: dict, annotations, mask_names, 
               parameters: dict, version: str,
               annotations_ids=None):
    """Save a dense result or a fresh patch iterator to one exclusive HDF5 file.

    Args:
        path (str or Path): Destination filename. Missing parents are created;
            an existing file is always refused.
        patches (dict or iterator): N-first result with image-name dictionaries,
            coordinates (N, 3), label=None or per-size label arrays, and mode.
            A streamed result must yield one group at a time, with N=1.
            When point IDs are provided, annotations_ids contains per-size
            object arrays (N,) whose entries are nonempty integer ID arrays.
        images (dict): Prepared, in-memory NiBabel images defining image names,
            output dtypes and geometry metadata, including for an empty result.
        annotations (ndarray or None): Processed zero-based point coordinates,
            shape (K, 3). None means annotations and labels are unavailable.
        mask_names (list or tuple): Names of the supplied discrete masks.
        parameters (dict): Preprocessing, generation and optional postprocessing
            settings retained by PatchGenerator.
        version (str): Algorithm version recorded in the file.
        annotations_ids (ndarray or None): Optional int64 point IDs, in the same
            order as annotations. Per-patch ID lists follow the label criterion.

    Raises:
        FileExistsError: The destination already exists.
        ValueError: A bundle has inconsistent geometry settings, shapes, labels
            or dtypes, or a streamed bundle contains more than one group.

    Returns:
        destination (Path): Saved file. /images/name/size preserves N-first
            layout; /coordinates and /label/size share its N-axis order. Unknown
            labels are represented by labels_available=False and an empty label
            group. Optional /annotations_ids stores point IDs, while
            /patch_annotations_ids/size stores one variable-length int64 array
            per patch; [0] denotes a negative label. Patch datasets are extensible
            and compressed using lzf. Writing
            buffers at most eight groups; failures retain complete=False.
    """
    generation = parameters["generation"]
    mode = generation["mode"]
    if mode not in ("2D", "3D"):
        raise ValueError("The saved generation mode must be 2D or 3D.")
    
    sizes = tuple(generation["patch_sizes"])
    image_names = tuple(images)
    dtypes = {name: np.asarray(image.dataobj).dtype for name, image in images.items()}
    labels_available = annotations is not None
    annotation_array = None
    if labels_available:
        annotation_array = np.asarray(annotations, dtype=np.float64)
        if (annotation_array.ndim != 2 or annotation_array.shape[1] != 3 
            or not np.isfinite(annotation_array).all()):
            raise ValueError("Processed annotations must be a finite array "
                             "with shape (K, 3).")
    
    dense_count = None
    if isinstance(patches, dict):
        dense_count = _validate_bundle(patches, image_names, sizes, mode, 
                                       dtypes, labels_available, annotations_ids)
    geometry = {
        name: {"shape": list(image.shape), "affine": np.asarray(image.affine).tolist(),
               "spacing": nibabel.affines.voxel_sizes(image.affine).tolist(),
               "orientation": list(nibabel.aff2axcodes(image.affine)),
               "dtype": str(dtypes[name]), "filename": image.get_filename()}
        for name, image in images.items()}
    parameters_json = json.dumps(parameters, default=_json_value, 
                                 sort_keys=True, allow_nan=False)
    geometry_json = json.dumps(geometry, default=_json_value, 
                               sort_keys=True, allow_nan=False)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.info("Saving patch result to %s", destination)
    with h5py.File(destination, "x") as output:
        output.attrs["complete"] = False
        output.attrs["algorithm_version"] = version
        output.attrs["schema_version"] = "3.1"
        output.attrs["patch_axis"] = 0
        output.attrs["array_layout"] = "N-first"
        output.attrs["mode"] = mode
        output.attrs["patch_sizes"] = np.asarray(sizes, dtype=np.int64)
        output.attrs["stride"] = np.asarray(generation["stride"], dtype=np.int64)
        output.attrs["label_border"] = int(generation["label_border"])
        output.attrs["mask_names_json"] = json.dumps(list(mask_names))
        output.attrs["parameters_json"] = parameters_json
        output.attrs["input_geometry_json"] = geometry_json
        output.attrs["labels_available"] = labels_available
        output.attrs["annotations_available"] = labels_available
        output.attrs["annotations_ids_available"] = annotations_ids is not None
        output.attrs["environment_json"] = json.dumps({
            "python": platform.python_version(), "numpy": np.__version__,
            "nibabel": nibabel.__version__, "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version,
            "scipy": dependency_version("scipy"),
            "tqdm": dependency_version("tqdm")}, sort_keys=True)
        output.create_dataset("coordinates", shape=(0, 3), maxshape=(None, 3),
                              dtype="f8", chunks=(8, 3), compression="lzf")
        label_group = output.create_group("label")
        label_group.attrs["labels_available"] = labels_available
        if labels_available:
            output.create_dataset("annotations", data=annotation_array)
            for size in sizes:
                label_group.create_dataset(str(size), shape=(0,), maxshape=(None,),
                                           dtype="i1", chunks=(256,), compression="lzf")
        
        if annotations_ids is not None:
            output.create_dataset("annotations_ids", data=annotations_ids)
            ids_group = output.create_group("patch_annotations_ids")
            ids_group.attrs["no_annotation_id"] = 0
            ids_group.attrs["membership"] = "same as label, including label_border"
            for size in sizes:
                ids_group.create_dataset(str(size), shape=(0,), maxshape=(None,),
                                         dtype=h5py.vlen_dtype(np.dtype("int64")),
                                         chunks=(256,), compression="lzf")
        
        dimensions = 3 if mode == "3D" else 2
        image_group = output.create_group("images")
        for name in image_names:
            group = image_group.create_group(name)
            for size in sizes:
                spatial_shape = (size,) * dimensions
                group.create_dataset(str(size), shape=(0,) + spatial_shape,
                                     maxshape=(None,) + spatial_shape, dtype=dtypes[name],
                                     chunks=(1,) + spatial_shape, compression="lzf")
        
        if dense_count is not None:
            for bundle in _dense_batches(patches, image_names, sizes, dense_count):
                _append_batch(output, [bundle], image_names, sizes, mode, 
                              dtypes, labels_available, annotations_ids)
        else:
            iterator = iter(patches)
            while True:
                batch = list(islice(iterator, 8))
                if not batch:
                    break
                for bundle in batch:
                    if _validate_bundle(bundle, image_names, sizes, mode, 
                                        dtypes, labels_available, annotations_ids) > 1:
                        raise ValueError("A streamed patch bundle must contain "
                                         "at most one group.")
                _append_batch(output, batch, image_names, sizes, mode, dtypes,
                              labels_available, annotations_ids)
        output.attrs["patch_count"] = output["coordinates"].shape[0]
        output.attrs["complete"] = True
        output.flush()
        logger.info("Saved %d patch groups to %s", output.attrs["patch_count"], destination)
    return destination
