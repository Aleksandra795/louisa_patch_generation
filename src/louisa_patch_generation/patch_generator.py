import logging
import numbers
from pathlib import Path
import nibabel
from nibabel.processing import resample_from_to
import numpy as np
from ._sampling import (
    _generation_parameters, 
    _iterate_patches, 
    _materialize_patches, 
    _filter_patches,
    )
from ._storage import _save_hdf5


def _image_geometry(images):
    """Validate aligned 3D NiBabel image headers without loading voxel arrays."""
    if not isinstance(images, dict) or not images:
        raise ValueError("images must be a nonempty dictionary of NiBabel images.")
    first_shape = None
    first_affine = None
    
    for name, image in images.items():
        if (not isinstance(name, str) or not name 
            or name in ("coordinates", "label", "mode", ".", "..") or "/" in name):
            raise ValueError("Image names must be nonempty, contain no slashes,"
                             " and differ from coordinates, label and mode.")
        
        if not isinstance(image, (nibabel.Nifti1Image, nibabel.Nifti2Image)):
            raise ValueError(f"{name}: provide an image returned by nibabel.load"
                             " or an in-memory NIfTI image.")
        
        if len(image.shape) != 3 or any(side <= 0 for side in image.shape):
            raise ValueError(f"{name}: a nonempty 3D image is required.")
        affine = np.asarray(image.affine, dtype=float)
        
        if (affine.shape != (4, 4) or not np.isfinite(affine).all() 
            or np.linalg.matrix_rank(affine[:3, :3]) != 3):
            raise ValueError(f"{name}: the affine must be finite and invertible.")
        
        if not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-7):
            raise ValueError(f"{name}: invalid homogeneous affine row.")
        spacing = nibabel.affines.voxel_sizes(affine)
        directions = affine[:3, :3] / spacing
        
        if not np.allclose(directions.T @ directions, np.eye(3), rtol=0, atol=1e-5):
            raise ValueError(f"{name}: sheared voxel grids are unsupported.")
        
        if not np.allclose(image.header.get_zooms()[:3], spacing, rtol=1e-5, atol=1e-5):
            raise ValueError(f"{name}: header spacing disagrees with image geometry.")
        
        if image.header.get_xyzt_units()[0] not in ("mm", "unknown"):
            raise ValueError(f"{name}: spacing must be expressed in millimetres.")
        
        if first_shape is None:
            first_shape, first_affine = image.shape, affine.copy()
        elif (image.shape != first_shape or not 
              np.allclose(affine, first_affine, rtol=0, atol=1e-5)):
            raise ValueError("All input images must already have matching shapes"
                             " and full voxel geometry; prepare their orientation "
                             "and registration before using this class.")
    return first_shape, first_affine


def _loaded_images(images):
    """Load each image once while retaining its existing orientation and labels."""
    result = {}
    for name, image in images.items():
        data = np.asanyarray(image.dataobj)
        if data.dtype.kind not in "biuf":
            raise ValueError(f"{name}: real numeric voxel data are required.")
        result[name] = image.__class__(data, image.affine, header=image.header.copy())
    return result


class PatchGenerator:
    """Prepare and extract aligned patches using zero-based voxel annotations.

    Args:
        images (dict): Names mapped to registered 3D images from nibabel.load or
            in-memory NIfTI images. Matching full geometry is required. Images
            retain their original axis order and directions throughout processing.
        mask_names (str or list, optional): Names of discrete masks. None means
            no masks. Names control interpolation and automatic background-reference
            selection; mask values are not inferred from intensities or filenames.
        annotations (tuple or ndarray, optional): One zero-based (i, j, k) point
            or an (N, 3) collection in input-array axis order. None means unknown
            labels; [] means a known absence of points. Fractional positions are
            retained. Defaults to None.
        annotations_ids (list or ndarray, optional): Positive integer IDs, one
            per annotation row. Zero is reserved for negative patches. Repeated
            IDs may describe multiple points of the same object. Defaults None.

    Raises:
        ValueError: Image geometry, mask names, annotation coordinates or IDs are invalid.

    Attributes:
        images (dict): Current images, initially the inputs and then the outputs
            of preprocessing. Unknown NIfTI spatial units are accepted under the
            explicit convention that spacing is in millimetres.
        annotations (ndarray or None): Current continuous voxel positions, shape
            (N, 3), or None. No world-coordinate annotations are required.
        annotations_ids (ndarray or None): Int64 point IDs, unchanged by resampling.
        patches (dict or iterator or None): Latest generated or filtered result.
    """

    def __init__(self, images: dict, mask_names=None, annotations=None, 
                 annotations_ids=None):
        _image_geometry(images)
        
        if mask_names is None:
            mask_names = []
        elif isinstance(mask_names, str):
            mask_names = [mask_names]
        elif not isinstance(mask_names, list):
            raise ValueError("mask_names must be a string, a list of names, or None.")
        
        if any(not isinstance(name, str) or name not in images for name in mask_names):
            raise ValueError("mask_names must identify existing input images.")
        
        if annotations is not None:
            annotations = np.array(annotations, dtype=float, copy=True)
            if annotations.shape == (3,):
                annotations = annotations.reshape(1, 3)
            elif annotations.ndim == 1 and annotations.size == 0:
                annotations = annotations.reshape(0, 3)
            if (annotations.ndim != 2 or annotations.shape[1] != 3 or 
                not np.isfinite(annotations).all()):
                raise ValueError("annotations must be a finite point or an array "
                                 "with shape (N, 3).")
        
        if annotations_ids is not None:
            if annotations is None:
                raise ValueError("annotations_ids require annotations.")
            
            if not isinstance(annotations_ids, (list, tuple, np.ndarray)):
                raise ValueError("annotations_ids must be a one-dimensional list"
                                 " of positive integer IDs.")
            ids = np.asarray(annotations_ids, dtype=object)
            if ids.ndim != 1 or len(ids) != len(annotations):
                raise ValueError("annotations_ids must contain one ID per annotation row.")
            
            if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
                   or not 0 < value <= np.iinfo(np.int64).max for value in ids):
                raise ValueError("Annotation IDs must be positive int64 integers; "
                                 "zero is reserved.")
            
            if "annotations_ids" in images:
                raise ValueError("The image name annotations_ids is reserved when IDs "
                                 "are supplied.")
            
            annotations_ids = np.asarray(ids, dtype=np.int64)
        self.annotations_ids = annotations_ids
        self.mask_names = list(dict.fromkeys(mask_names))
        self._input_geometry = {
            name: {"filename": image.get_filename(), "shape": list(image.shape),
                   "spacing": nibabel.affines.voxel_sizes(image.affine).tolist(),
                   "affine": np.asarray(image.affine).tolist()}
            for name, image in images.items()}
        self._source_images = dict(images)
        self._source_annotations = None if annotations is None else annotations.copy()
        self.images = dict(images)
        self.annotations = None if annotations is None else annotations.copy()
        self.patches = None
        self._generated_patches = None
        self._preprocessing_parameters = None
        self._generation_parameters = None
        self._postprocessing_parameters = None


    def preprocess(self, target_spacing=(1, 1, 1), verbosity: int = 0):
        """Resample aligned inputs and scale voxel annotations without reorientation.

        Args:
            target_spacing (tuple or float): Positive finite mm spacing, scalar
                or three values. Defaults to (1, 1, 1).
            verbosity (int): 0 is silent; a positive integer enables stage logging.

        Raises:
            ValueError: Target spacing, logging level or input geometry is invalid.

        Returns:
            images (dict): Prepared NiBabel images with the original names and
                orientation. Discrete masks use nearest-neighbor interpolation;
                other resampled images use linear float32 interpolation. No skull
                stripping or normalization is performed.
            annotations (ndarray or None): Original voxel positions multiplied
                by original_spacing / target_spacing, with no rounding.

        Notes:
            The original origin and axis directions are retained. Target shape
            is ceil((n - 1) * original_spacing / target_spacing - 1e-7) + 1, so
            samples beyond the original field are filled with zero. Repeated
            calls start from the original data instead of repeatedly resampling.
            Spacing already equal within NIfTI precision (rtol=1e-6, atol=1e-7)
            leaves arrays, affine and annotations unchanged.
        """
        if (isinstance(verbosity, (bool, np.bool_)) 
            or not isinstance(verbosity, (int, np.integer)) or verbosity < 0):
            raise ValueError("verbosity must be a nonnegative integer.")
        
        if np.asarray(target_spacing).dtype.kind == "b":
            raise ValueError("target_spacing must contain millimetre values, not booleans.")
        spacing = np.asarray(target_spacing, dtype=float)
        
        if spacing.ndim == 0:
            spacing = np.repeat(spacing, 3)
        
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("target_spacing must contain one or three finite positive values.")
        shape, affine = _image_geometry(self._source_images)
        self._source_images = _loaded_images(self._source_images)
        original_spacing = nibabel.affines.voxel_sizes(affine)
        same_spacing = np.allclose(original_spacing, spacing, rtol=1e-6, atol=1e-7)
        target_shape = shape if same_spacing else tuple((np.ceil(
            (np.asarray(shape) - 1) * original_spacing / spacing - 1e-7).astype(int) + 1).tolist())
        target_affine = affine.copy()
        
        if not same_spacing:
            target_affine[:3, :3] *= spacing / original_spacing
        prepared = {}
        
        for name, image in self._source_images.items():
            if same_spacing:
                prepared[name] = image
                continue
            source = image
            if name not in self.mask_names:
                source = image.__class__(np.asarray(image.dataobj, dtype=np.float32),
                                         image.affine, header=image.header.copy())
                source.set_data_dtype(np.float32)
            prepared[name] = resample_from_to(source, (target_shape, target_affine),
                                            order=0 if name in self.mask_names else 1,
                                            mode="constant", cval=0)
        _image_geometry(prepared)
        self.images = prepared
        self.annotations = None if self._source_annotations is None else (
            self._source_annotations.copy() if same_spacing 
            else self._source_annotations * original_spacing / spacing)
        self._preprocessing_parameters = {"target_spacing": spacing.tolist(),
                                           "input_spacing": original_spacing.tolist(),
                                           "input_shape": list(shape), "output_shape": list(target_shape),
                                           "resampled": not bool(same_spacing), "verbosity": int(verbosity)}
        self._generation_parameters = None
        self._postprocessing_parameters = None
        self._generated_patches = None
        self.patches = None
        
        if verbosity:
            logging.getLogger(__name__).info("Prepared %d images: shape=%s spacing=%s orientation=%s",
                                            len(prepared), target_shape, spacing, nibabel.aff2axcodes(target_affine))
        return self.images, self.annotations


    def generate_patches(self, mode: str = "3d", patch_sizes=32, stride=32,
                         label_border: int = 0, slice_axis: int = 2,
                         generator: bool = False, verbosity: int = 0):
        """Generate synchronized patches, with N on the first array axis.

        Args:
            mode (str): '3d' for cubes or '2d' for square slices. Defaults to '3d'.
            patch_sizes (int or tuple): One or two distinct positive sides with
                equal parity. Defaults to 32. Largest size sets the common grid.
            stride (int or tuple): Positive step or three axis-specific steps.
                In 2D the slice axis is stepped too. Defaults to 32.
            label_border (int): Minimum physical-face margin in processed voxels
                on each active axis, applied independently to every size. A point
                in a margin receives label 0. Defaults to 0; 2D margins apply only
                in-plane. A positive margin is inclusive on both inner faces;
                ordinary patch support remains half-open [start-.5, stop-.5).
            slice_axis (int): Slice axis in 2D, default 2; ignored in 3D.
            generator (bool): False returns all arrays; True returns an iterator
                whose dictionaries have one patch on the first axis. Defaults False.
            verbosity (int): 0 silent, 1 stage logging, >=2 progress. Defaults 0.

        Raises:
            ValueError: Generation parameters or current geometry are invalid.

        Returns:
            patches (dict or iterator): Original image names map to size ->
                arrays (N, side, side, side), or (N, side, side) in 2D.
                'coordinates' has shape (N, 3), 'label' is None or size -> integer
                array (N,), and 'mode' is '3D' or '2D'. Centers are zero-based
                geometric voxel centers, shared by sizes and modalities.
            When point IDs were supplied, 'annotations_ids' maps each size to an
            object array (N,) of unique ascending int64 ID arrays. [0] denotes
            no point satisfying the label criterion, including label_border.
        """
        parameters = _generation_parameters(mode, patch_sizes, stride, label_border,
                                             slice_axis, generator, verbosity)
        _image_geometry(self.images)
        self.images = _loaded_images(self.images)
        if self._preprocessing_parameters is None:
            self._source_images = dict(self.images)
        if parameters["generator"]:
            patches = _iterate_patches(self.images, self.annotations, 
                                       parameters, self.annotations_ids)
            self._generated_patches = None
        else:
            patches = _materialize_patches(self.images, self.annotations, 
                                           parameters, self.annotations_ids)
            self._generated_patches = patches
        self._generation_parameters = parameters
        self._postprocessing_parameters = None
        self.patches = patches
        return patches

    def postprocess(self, max_zero_percent: float = 100, remove_inconsistent: bool = False,
                    reference: str | None = None):
        """Filter synchronized patches while retaining the original output layout.

        Args:
            max_zero_percent (float): Maximum exact-zero percent in every size
                of the reference image, inclusive, from 0 to 100. Defaults 100.
            remove_inconsistent (bool): Drop a paired position if the two labels
                differ. Unknown labels are not compared. Defaults False.
            reference (str, optional): Background reference, default the first
                image not named in mask_names. An explicit name overrides this.
                If all inputs are masks, the background criterion is skipped.

        Raises:
            ValueError: Generation was not run or filter parameters are invalid.

        Returns:
            patches (dict or iterator): Filtered arrays or one-group dictionaries,
                matching the selected generation mode. All modalities, labels
            and coordinates, including optional ID lists, stay synchronized.
            Each call applies its settings
                to the full generated grid, not to a previously filtered subset.
        """
        if self._generation_parameters is None:
            raise ValueError("Call generate_patches before postprocess.")
        reference_name = reference
        if reference_name is None:
            reference_name = next((name for name in self.images if name not in self.mask_names), 
                                  None)
        if self._generation_parameters["generator"]:
            source = _iterate_patches(self.images, self.annotations, 
                                      self._generation_parameters, self.annotations_ids)
        else:
            source = self._generated_patches
        filtered = _filter_patches(source, list(self.images), self.mask_names,
                                   max_zero_percent, remove_inconsistent, reference_name)
        self._postprocessing_parameters = {"max_zero_percent": float(max_zero_percent),
                                           "remove_inconsistent": bool(remove_inconsistent),
                                           "reference": reference_name}
        self.patches = filtered
        return filtered


    def save_hdf5(self, path):
        """Save the current complete result with one destination argument.

        Args:
            path (str or Path): Full destination filename. Parent directories
                are created, and existing files are always refused.

        Raises:
            ValueError: Patches have not been generated or result data are invalid.
            FileExistsError: The destination already exists.

        Returns:
            path (Path): Written HDF5 file with patch-first image datasets,
                shared coordinates, separate size labels, annotations, parameters
                and geometry. A generator is replayed from the beginning, so a
                previous next() preview cannot remove patches from the saved file.
            Optional point IDs are saved as /annotations_ids (K,) and per-patch
            variable-length int64 lists as /patch_annotations_ids/size (N,).
        """
        if self._generation_parameters is None:
            raise ValueError("Call generate_patches before save_hdf5.")
        if self._generation_parameters["generator"]:
            patches = _iterate_patches(self.images, self.annotations, 
                                       self._generation_parameters, self.annotations_ids)
            if self._postprocessing_parameters is not None:
                patches = _filter_patches(patches, list(self.images), self.mask_names,
                                           **self._postprocessing_parameters)
        else:
            patches = self.patches
        parameters = {"preprocessing": self._preprocessing_parameters,
                      "generation": self._generation_parameters,
                      "postprocessing": self._postprocessing_parameters,
                      "input_geometry": self._input_geometry,
                      "input_annotations": (None if self._source_annotations is None 
                                            else self._source_annotations.tolist())}
        return _save_hdf5(Path(path), patches, self.images, self.annotations,
                          self.mask_names, parameters, "0.4.0", self.annotations_ids)
