import logging
import itertools
import math
import numbers
import numpy as np


def _positive_integer(value, name):
    """Validate a positive integer without accepting boolean values."""
    if isinstance(value, (bool, np.bool_)) or not \
        isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{name} must contain positive integers.")
    return int(value)


def _generation_parameters(mode: str = "3d", patch_sizes=32, stride=32,
                           label_border: int = 0, slice_axis: int = 2,
                           generator: bool = False, verbosity: int = 0):
    """Validate and normalize the class's patch-generation parameters.

    Args:
        mode (str): Case-insensitive '3d' or '2d'. Defaults to '3d'.
        patch_sizes (int or tuple): One or two distinct positive side lengths.
            Paired sizes must have equal parity. Defaults to 32.
        stride (int or tuple): Positive step on all axes, or three independent
            positive steps in processed-array axis order. Defaults to 32.
        label_border (int): Nonnegative margin in processed voxels on the active
            patch axes. It must leave a nonempty interior at every size.
        slice_axis (int): Axis removed from 2D outputs, default 2; ignored in 3D.
        generator (bool): Whether the class returns an iterator, default False.
        verbosity (int): Nonnegative logging/progress level, default 0.

    Raises:
        ValueError: A generation parameter is invalid or incompatible.

    Returns:
        parameters (dict): Uppercase mode, ascending side-length tuple,
            three-element stride tuple, integer border, resolved slice axis,
            boolean generator flag and integer verbosity.
    """
    if not isinstance(mode, str) or mode.lower() not in ("3d", "2d"):
        raise ValueError("mode must be '3d' or '2d'.")
    
    mode = mode.upper()
    
    if isinstance(patch_sizes, numbers.Integral):
        sizes = (_positive_integer(patch_sizes, "patch_sizes"),)
    else:
        if not isinstance(patch_sizes, (tuple, list, np.ndarray)):
            raise ValueError("patch_sizes must be an integer "
                             "or one or two side lengths.")
        
        if isinstance(patch_sizes, np.ndarray) and patch_sizes.ndim != 1:
            raise ValueError("patch_sizes must be a one-dimensional sequence.")
        
        if len(patch_sizes) not in (1, 2):
            raise ValueError("Provide one or two patch sizes.")
        sizes = tuple(_positive_integer(value, "patch_sizes") for value in patch_sizes)
    
    sizes = tuple(sorted(sizes))
    
    if len(set(sizes)) != len(sizes):
        raise ValueError("Patch sizes must be distinct.")
    
    if len(sizes) == 2 and sizes[0] % 2 != sizes[1] % 2:
        raise ValueError("Paired sizes must have equal parity to share "
                         "exact geometric centers.")
    
    if isinstance(stride, numbers.Integral):
        steps = (_positive_integer(stride, "stride"),) * 3
    else:
        if not isinstance(stride, (tuple, list, np.ndarray)):
            raise ValueError("stride must be a positive integer or "
                             "three positive integers.")
        if isinstance(stride, np.ndarray) and stride.ndim != 1:
            raise ValueError("stride must be a one-dimensional sequence.")
        if len(stride) != 3:
            raise ValueError("stride must contain three axis-specific steps.")
        steps = tuple(_positive_integer(value, "stride") for value in stride)
    
    if (isinstance(label_border, (bool, np.bool_)) or not 
        isinstance(label_border, numbers.Integral) or label_border < 0):
        raise ValueError("label_border must be a nonnegative integer.")
    
    label_border = int(label_border)
    if 2 * label_border >= sizes[0]:
        raise ValueError("label_border must leave a nonempty "
                         "interior in the smallest patch.")
    
    if mode == "2D":
        if (isinstance(slice_axis, (bool, np.bool_)) or not 
        isinstance(slice_axis, numbers.Integral) or slice_axis not in (0, 1, 2)):
            raise ValueError("slice_axis must be 0, 1, or 2 for 2D patches.")
        slice_axis = int(slice_axis)
    else:
        slice_axis = None
    
    if not isinstance(generator, (bool, np.bool_)):
        raise ValueError("generator must be True or False.")
    
    if (isinstance(verbosity, (bool, np.bool_)) or not 
        isinstance(verbosity, numbers.Integral) or verbosity < 0):
        raise ValueError("verbosity must be a nonnegative integer.")
    
    return {"mode": mode, "patch_sizes": sizes, "stride": steps,
            "label_border": label_border, "slice_axis": slice_axis,
            "generator": bool(generator), "verbosity": int(verbosity)}


def _patch_grid(shape, parameters):
    """Describe the largest fitting patch grid without storing its positions."""
    active_axes = tuple(axis for axis in range(3) if axis != parameters["slice_axis"])
    extent = np.ones(3, dtype=np.int64)
    extent[list(active_axes)] = max(parameters["patch_sizes"])
    ranges = tuple(range(0, int(shape[axis] - extent[axis] + 1), parameters["stride"][axis])
                   for axis in range(3))
    return active_axes, extent, ranges, math.prod(len(values) for values in ranges)


def _point_membership(points, start, extent, active_axes, border):
    """Return a point membership mask using patch support and the inward margin.

    Patch support is half-open [start - 0.5, stop - 0.5). With a positive
    border, distances equal to that margin are accepted at both inner faces;
    the one-voxel 2D slice axis always retains its original half-open support.
    """
    inside = np.all((points >= start - 0.5) & (points < start + extent - 0.5), axis=1)
    if border:
        axes = list(active_axes)
        inside &= np.all((points[:, axes] >= start[axes] - 0.5 + border) &
                         (points[:, axes] <= start[axes] + extent[axes] - 0.5 - border), axis=1)
    return inside


def _iterate_patches(images: dict, annotations, parameters: dict, annotations_ids=None):
    """Return synchronized voxel-space patch dictionaries with a singleton N axis.

    Args:
        images (dict): In-memory NiBabel images already checked on one 3D grid.
        annotations (ndarray or None): Current zero-based continuous voxel points
            with shape (K, 3), or None for unknown labels.
        parameters (dict): Validated output of _generation_parameters.
        annotations_ids (ndarray or None): Optional int64 ID for each point.

    Returns:
        patches (iterator): One dictionary per center, containing each input
            name mapped to {size: ndarray}, coordinates (1, 3), label (None or
            {size: int8 array of length one}) and uppercase mode. Image arrays
            have shape (1, size, size, size) in 3D or (1, size, size) in 2D.
            They are views. The largest size defines fitting positions starting
            at zero, with no padding or forced terminal position. Slice stride
            applies to the slice axis too. No physical/world coordinates are
            exposed, and annotation points never move the grid.
            With supplied point IDs, annotations_ids maps sizes to object arrays
            (1,) containing unique ascending IDs or [0] for negative patches.
    """
    arrays = {name: np.asanyarray(image.dataobj) for name, image in images.items()}
    shape = next(iter(arrays.values())).shape
    active_axes, largest_extent, ranges, count = _patch_grid(shape, parameters)
    sizes = parameters["patch_sizes"]
    largest = max(sizes)
    slice_axis = parameters["slice_axis"]
    points = None if annotations is None else np.asarray(annotations, dtype=float)
    logger = logging.getLogger(__name__)
    if parameters["verbosity"]:
        logger.info("Generating %d patch groups: mode=%s sizes=%s stride=%s border=%d",
                    count, parameters["mode"], sizes, parameters["stride"], 
                    parameters["label_border"])

    def iterate():
        positions = itertools.product(*ranges)
        progress = None
        if parameters["verbosity"] >= 2 and count >= 1000:
            from tqdm.auto import tqdm
            progress = tqdm(positions, total=count, 
                            desc="Generating patches", unit="group")
            positions = progress
        completed = 0
        try:
            for position in positions:
                largest_start = np.asarray(position, dtype=np.int64)
                center = largest_start.astype(np.float64) + (largest_extent - 1) / 2
                result = {name: {} for name in arrays}
                labels = None if points is None else {}
                if annotations_ids is not None:
                    result["annotations_ids"] = {}
                for size in sizes:
                    extent = np.ones(3, dtype=np.int64)
                    extent[list(active_axes)] = size
                    start = largest_start.copy()
                    start[list(active_axes)] += (largest - size) // 2
                    selection = tuple(slice(int(start[axis]), 
                                            int(start[axis] + extent[axis]))
                                      for axis in range(3))
                    for name, array in arrays.items():
                        patch = array[selection]
                        if slice_axis is not None:
                            patch = np.squeeze(patch, axis=slice_axis)
                        result[name][size] = patch[np.newaxis, ...]
                    if labels is not None:
                        inside = _point_membership(points, start, extent, 
                                                   active_axes, parameters["label_border"])
                        labels[size] = np.asarray([int(np.any(inside))], dtype=np.int8)
                        if annotations_ids is not None:
                            ids = np.unique(annotations_ids[inside])
                            values = np.empty(1, dtype=object)
                            values[0] = ids if len(ids) else np.asarray([0], dtype=np.int64)
                            result["annotations_ids"][size] = values
                result["coordinates"] = center[np.newaxis, :]
                result["label"] = labels
                result["mode"] = parameters["mode"]
                completed += 1
                yield result
        finally:
            if progress is not None:
                progress.close()
        if parameters["verbosity"]:
            logger.info("Generated %d patch groups.", completed)

    return iterate()


def _materialize_patches(images: dict, annotations, parameters: dict, annotations_ids=None):
    """Allocate complete N-first arrays and fill them without an intermediate list.

    Args:
        images (dict): Validated, matching in-memory NiBabel images.
        annotations (ndarray or None): Current voxel-space point annotations.
        parameters (dict): Validated output of _generation_parameters.
        annotations_ids (ndarray or None): Optional int64 ID for each point.

    Returns:
        patches (dict): Input image names mapped to size-specific arrays with N
            first, shared coordinates (N, 3), uppercase mode, and label=None or
            a per-size dictionary of int8 arrays (N,). Empty grids retain all
            keys and have N=0. Array dtypes match the prepared image data.
            Optional annotations_ids maps sizes to object arrays (N,) containing
            the same ID lists as streamed results.
    """
    shape = next(iter(images.values())).shape
    _, _, _, count = _patch_grid(shape, parameters)
    dimensions = 3 if parameters["mode"] == "3D" else 2
    sizes = parameters["patch_sizes"]
    result = {name: {size: np.empty((count,) + (size,) * dimensions, 
                                    dtype=np.asanyarray(image.dataobj).dtype)
                     for size in sizes}
              for name, image in images.items()}
    result["coordinates"] = np.empty((count, 3), dtype=np.float64)
    result["label"] = (
        None
        if annotations is None
        else {size: np.empty(count, dtype=np.int8) for size in sizes}
    )
    result["mode"] = parameters["mode"]
    if annotations_ids is not None:
        result["annotations_ids"] = {size: np.empty(count, dtype=object) for size in sizes}
    for index, bundle in enumerate(_iterate_patches(images, 
                                                    annotations, parameters, 
                                                    annotations_ids)):
        result["coordinates"][index] = bundle["coordinates"][0]
        for name in images:
            for size in sizes:
                result[name][size][index] = bundle[name][size][0]
        if result["label"] is not None:
            for size in sizes:
                result["label"][size][index] = bundle["label"][size][0]
                if annotations_ids is not None:
                    result["annotations_ids"][size][index] = bundle["annotations_ids"][size][0]
    return result


def _filter_patches(patches, image_names, mask_names, max_zero_percent: float = 100,
                    remove_inconsistent: bool = False, reference: str | None = None):
    """Filter dense or streamed N-first patch results with one synchronized mask.

    Args:
        patches (dict or iterator): A dense result or iterator from generation.
        image_names (list): Input names in their original dictionary order.
        mask_names (list): Names classified as discrete masks by the user.
        max_zero_percent (float): Maximum exact-zero percentage, inclusive,
            between 0 and 100. Defaults to 100.
        remove_inconsistent (bool): Remove complete centers whose available
            labels disagree between sizes. Defaults to False.
        reference (str or None): Named background reference, otherwise the first
            name not in mask_names. Explicit masks are allowed when intensity
            images are also present. All-mask inputs always skip this filter.

    Raises:
        ValueError: A filter parameter or explicit reference is invalid.
        TypeError: patches is neither a dictionary nor an iterable.

    Returns:
        filtered (dict or iterator): The same representation as the input, with
            all modalities, sizes, labels and coordinates retained or removed
            together. Any reference size above the threshold rejects the row.
            Missing annotations do not count as inconsistent labels. All-mask
            input still permits inconsistent-label removal. Unchanged dense
            results and retained streamed groups preserve their original objects.
    """
    if (isinstance(max_zero_percent, (bool, np.bool_)) or not 
        isinstance(max_zero_percent, numbers.Real)):
        raise ValueError("max_zero_percent must be a finite number from 0 to 100.")
    if not np.isfinite(max_zero_percent) or not 0 <= max_zero_percent <= 100:
        raise ValueError("max_zero_percent must be a finite number from 0 to 100.")
    if not isinstance(remove_inconsistent, (bool, np.bool_)):
        raise ValueError("remove_inconsistent must be True or False.")
    names = tuple(image_names)
    masks = set(mask_names)
    if reference is not None and (not isinstance(reference, str) or reference not in names):
        raise ValueError("reference must identify one of the input images.")
    intensity_names = tuple(name for name in names if name not in masks)
    selected_reference = (reference if reference is not None 
                          else (intensity_names[0] if intensity_names else None))
    apply_background = bool(intensity_names) and max_zero_percent < 100

    def keep_rows(bundle):
        count = bundle["coordinates"].shape[0]
        keep = np.ones(count, dtype=bool)
        if apply_background:
            for array in bundle[selected_reference].values():
                axes = tuple(range(1, array.ndim))
                background = (100.0 * np.count_nonzero(array == 0, axis=axes) 
                              / math.prod(array.shape[1:]))
                keep &= background <= max_zero_percent
        labels = bundle["label"]
        if remove_inconsistent and labels is not None and len(labels) > 1:
            baseline = next(iter(labels.values()))
            for values in labels.values():
                keep &= values == baseline
        return keep

    def select_rows(bundle, keep):
        if np.all(keep):
            return bundle
        result = {name: {size: array[keep] for size, array in bundle[name].items()}
                  for name in names}
        result["coordinates"] = bundle["coordinates"][keep]
        result["label"] = None if bundle["label"] is None else {
            size: values[keep] for size, values in bundle["label"].items()}
        result["mode"] = bundle["mode"]
        if "annotations_ids" in bundle and "annotations_ids" not in names:
            result["annotations_ids"] = {
                size: values[keep] for size, values in bundle["annotations_ids"].items()}
        return result

    if isinstance(patches, dict):
        return select_rows(patches, keep_rows(patches))
    groups = iter(patches)

    def iterate():
        for bundle in groups:
            keep = keep_rows(bundle)
            if np.any(keep):
                yield select_rows(bundle, keep)

    return iterate()
