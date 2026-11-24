import json
import tempfile
import unittest
from pathlib import Path
import h5py
import nibabel as nib
import numpy as np
from louisa_patch_generation import PatchGenerator


def _image(data, affine=None, dtype=np.float32):
    """Create a synthetic NiBabel image with explicit millimetre units."""
    image = nib.Nifti1Image(np.asarray(data, dtype=dtype), 
                             np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units("mm")
    return image


def _coordinates(shape):
    """Encode array indices so shifted or swapped crops have different values."""
    x, y, z = np.indices(shape)
    return (10000 * x + 100 * y + z + 1).astype(np.float32)


class PreprocessingTests(unittest.TestCase):
    def test_nearly_unit_header_spacing_preserves_shape_affine_and_annotations(self):
        affine = np.diag([1.00000005, 1.0, 1.0, 1.0])
        pipeline = PatchGenerator({"FLAIR": _image(np.ones((33, 3, 3)), affine)},
                                  annotations=(5.25, 1, 1))
        images, annotations = pipeline.preprocess()
        self.assertEqual(images["FLAIR"].shape, (33, 3, 3))
        np.testing.assert_array_equal(images["FLAIR"].affine, affine)
        np.testing.assert_array_equal(annotations, [[5.25, 1.0, 1.0]])

    def test_spacing_and_annotations_change_without_flipping_lps_axes(self):
        affine = np.diag([-2.0, -3.0, 4.0, 1.0])
        affine[:3, 3] = [10, 20, -7]
        values = _coordinates((3, 3, 3))
        points = np.asarray([[1.25, 1.5, 0.5]])
        original = _image(values, affine)
        pipeline = PatchGenerator({"FLAIR": original}, annotations=points)
        images, annotations = pipeline.preprocess()
        self.assertEqual(images["FLAIR"].shape, (5, 7, 9))
        self.assertEqual(nib.aff2axcodes(images["FLAIR"].affine), ("L", "P", "S"))
        np.testing.assert_allclose(images["FLAIR"].affine[:3, 3], affine[:3, 3])
        np.testing.assert_allclose(images["FLAIR"].affine[:3, :3], np.diag([-1, -1, 1]))
        np.testing.assert_allclose(annotations, [[2.5, 4.5, 2.0]])
        self.assertIsInstance(annotations, np.ndarray)
        self.assertEqual(annotations.dtype.kind, "f")
        np.testing.assert_array_equal(np.asanyarray(original.dataobj), values)
        np.testing.assert_array_equal(original.affine, affine)
        np.testing.assert_array_equal(points, [[1.25, 1.5, 0.5]])

    def test_intensities_are_linear_and_named_masks_keep_discrete_labels(self):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        values = np.indices((3, 3, 3))[0]
        mask = np.where(values >= 1, 7, 0)
        pipeline = PatchGenerator({"FLAIR": _image(values, affine, np.int16),
                                   "VENTRICLES": _image(mask, affine, np.uint8)},
                                  mask_names="VENTRICLES")
        images, annotations = pipeline.preprocess()
        self.assertIsNone(annotations)
        self.assertEqual(images["FLAIR"].shape, (5, 5, 5))
        self.assertAlmostEqual(float(np.asanyarray(images["FLAIR"].dataobj)[1, 2, 2]), 0.5)
        self.assertEqual(np.asanyarray(images["FLAIR"].dataobj).dtype, np.dtype(np.float32))
        self.assertTrue(set(np.unique(np.asanyarray(images["VENTRICLES"].dataobj))).issubset({0, 7}))
        np.testing.assert_allclose(images["FLAIR"].affine, images["VENTRICLES"].affine)

    def test_repeated_preprocessing_uses_original_coordinates(self):
        pipeline = PatchGenerator({"FLAIR": _image(np.ones((3, 3, 3)), np.diag([2, 2, 2, 1]))},
                                  annotations=(1, 1, 1))
        _, first = pipeline.preprocess(target_spacing=(1, 1, 1))
        first_copy = first.copy()
        images, second = pipeline.preprocess(target_spacing=(0.5, 0.5, 0.5))
        np.testing.assert_allclose(first_copy, [[2, 2, 2]])
        np.testing.assert_allclose(second, [[4, 4, 4]])
        self.assertEqual(images["FLAIR"].shape, (9, 9, 9))

    def test_equal_dimensions_with_shifted_origins_are_rejected(self):
        shifted = np.eye(4)
        shifted[0, 3] = 1
        with self.assertRaises(ValueError):
            pipeline = PatchGenerator({"FLAIR": _image(np.ones((3, 3, 3))),
                                       "T1": _image(np.ones((3, 3, 3)), shifted)})
            pipeline.preprocess()

    def test_nibabel_file_proxies_are_accepted_without_manual_wrapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.nii"
            nib.save(_image(_coordinates((6, 6, 6))), path)
            loaded = nib.load(path)
            self.assertTrue(nib.is_proxy(loaded.dataobj))
            pipeline = PatchGenerator({"FLAIR": loaded})
            images, _ = pipeline.preprocess()
            patches = pipeline.generate_patches(patch_sizes=2, stride=2)
            self.assertIsInstance(np.asanyarray(images["FLAIR"].dataobj), np.ndarray)
            self.assertEqual(patches["coordinates"].shape, (27, 3))


class GenerationTests(unittest.TestCase):
    def test_default_dense_output_has_shared_32_48_centers_and_separate_labels(self):
        values = _coordinates((64, 64, 64))
        pipeline = PatchGenerator({"FLAIR": _image(values), "T1": _image(values * 3 + 2)},
                                  annotations=(1, 1, 1))
        pipeline.preprocess()
        patches = pipeline.generate_patches(patch_sizes=(32, 48), stride=16)
        self.assertIsInstance(patches, dict)
        self.assertIs(pipeline.patches, patches)
        self.assertEqual(set(patches), {"FLAIR", "T1", "coordinates", "label", "mode"})
        self.assertEqual(patches["mode"], "3D")
        self.assertEqual(patches["coordinates"].shape, (8, 3))
        np.testing.assert_allclose(patches["coordinates"][0], [23.5, 23.5, 23.5])
        self.assertEqual(patches["FLAIR"][32].shape, (8, 32, 32, 32))
        self.assertEqual(patches["FLAIR"][48].shape, (8, 48, 48, 48))
        np.testing.assert_array_equal(patches["FLAIR"][32], patches["FLAIR"][48][:, 8:40, 8:40, 8:40])
        np.testing.assert_array_equal(patches["FLAIR"][32][0], values[8:40, 8:40, 8:40])
        for size in (32, 48):
            np.testing.assert_array_equal(patches["T1"][size], patches["FLAIR"][size] * 3 + 2)
            self.assertEqual(patches["label"][size].shape, (8,))
        self.assertEqual(int(patches["label"][32][0]), 0)
        self.assertEqual(int(patches["label"][48][0]), 1)

    def test_generator_and_dense_results_agree_in_3d_and_all_2d_axes(self):
        values = _coordinates((6, 7, 8))
        cases = [("3d", 2)] + [("2d", axis) for axis in range(3)]
        for mode, axis in cases:
            with self.subTest(mode=mode, axis=axis):
                dense_pipeline = PatchGenerator({"FLAIR": _image(values)}, 
                                                annotations=[(1, 1, 1), (5, 6, 7)])
                dense_pipeline.preprocess()
                dense = dense_pipeline.generate_patches(mode=mode, 
                                                        patch_sizes=(2, 4), 
                                                        stride=(2, 3, 2), 
                                                        slice_axis=axis)
                stream_pipeline = PatchGenerator({"FLAIR": _image(values)}, 
                                                 annotations=[(1, 1, 1), (5, 6, 7)])
                stream_pipeline.preprocess()
                stream = stream_pipeline.generate_patches(mode=mode, 
                                                          patch_sizes=(2, 4), 
                                                          stride=(2, 3, 2), 
                                                          slice_axis=axis, 
                                                          generator=True)
                self.assertIs(iter(stream), stream)
                groups = list(stream)
                np.testing.assert_array_equal(np.concatenate([group["coordinates"] for group in groups]), 
                                              dense["coordinates"])
                for size in (2, 4):
                    dimensions = 2 if mode == "2d" else 3
                    self.assertEqual(groups[0]["FLAIR"][size].shape, (1,) + (size,) * dimensions)
                    np.testing.assert_array_equal(np.concatenate([group["FLAIR"][size] for group in groups], 
                                                                 axis=0), dense["FLAIR"][size])
                    np.testing.assert_array_equal(np.concatenate([group["label"][size] for group in groups]), 
                                                  dense["label"][size])
                if mode == "2d":
                    selection = [slice(0, 4)] * 3
                    selection[axis] = 0
                    np.testing.assert_array_equal(dense["FLAIR"][4][0], 
                                                  values[tuple(selection)])

    def test_image_edges_are_omitted_without_padding_or_terminal_windows(self):
        pipeline = PatchGenerator({"FLAIR": _image(np.ones((7, 2, 2)))}, 
                                  annotations=[])
        pipeline.preprocess()
        patches = pipeline.generate_patches(patch_sizes=2, stride=3)
        np.testing.assert_allclose(patches["coordinates"][:, 0], [0.5, 3.5])
        empty = pipeline.generate_patches(patch_sizes=8, stride=3)
        self.assertEqual(empty["FLAIR"][8].shape, (0, 8, 8, 8))
        self.assertEqual(empty["coordinates"].shape, (0, 3))
        self.assertEqual(empty["label"][8].shape, (0,))

    def test_label_border_is_evaluated_separately_for_each_size(self):
        pipeline = PatchGenerator({"FLAIR": _image(np.ones((8, 8, 8)))}, 
                                  annotations=(2, 3, 3))
        pipeline.preprocess()
        ordinary = pipeline.generate_patches(patch_sizes=(4, 8), label_border=0)
        self.assertEqual(int(ordinary["label"][4][0]), 1)
        bordered = pipeline.generate_patches(patch_sizes=(4, 8), label_border=1)
        self.assertEqual(int(bordered["label"][4][0]), 0)
        self.assertEqual(int(bordered["label"][8][0]), 1)

    def test_positive_margin_includes_equality_at_both_inner_faces(self):
        cases = [(0.5, 1, 1), (2.5, 1, 1), (0.49999, 1, 0), (2.50001, 1, 0),
                 (3.5, 0, 0), (3.49999, 0, 1)]
        for x, border, expected in cases:
            with self.subTest(x=x, border=border):
                pipeline = PatchGenerator({"FLAIR": _image(np.ones((4, 4, 4)))}, 
                                          annotations=(x, 1, 1))
                pipeline.preprocess()
                result = pipeline.generate_patches(patch_sizes=4, label_border=border)
                self.assertEqual(int(result["label"][4][0]), expected)

    def test_2d_margin_applies_only_to_the_two_in_plane_axes(self):
        values = _coordinates((4, 4, 4))
        for axis in range(3):
            with self.subTest(axis=axis):
                pipeline = PatchGenerator({"FLAIR": _image(values)}, 
                                          annotations=(1, 1, 1))
                pipeline.preprocess()
                result = pipeline.generate_patches(mode="2d", patch_sizes=4, 
                                                   stride=1, label_border=1, 
                                                   slice_axis=axis)
                self.assertEqual(result["FLAIR"][4].shape, (4, 4, 4))
                row = int(np.flatnonzero(result["coordinates"][:, axis] == 1)[0])
                self.assertEqual(int(result["label"][4][row]), 1)
                self.assertEqual(int(result["label"][4].sum()), 1)
                np.testing.assert_array_equal(result["FLAIR"][4][row], 
                                              np.take(values, 1, axis=axis))

    def test_invalid_generation_parameters_are_rejected_eagerly(self):
        pipeline = PatchGenerator({"FLAIR": _image(np.ones((6, 6, 6)))})
        pipeline.preprocess()
        cases = [{"patch_sizes": 0}, {"patch_sizes": (2, 3)}, {"patch_sizes": (2, 4, 6)},
                 {"stride": 0}, {"stride": (1, 2)}, {"mode": "wrong"},
                 {"mode": "2d", "slice_axis": 3}, {"label_border": -1},
                 {"patch_sizes": 2, "label_border": 1}, {"label_border": True},
                 {"generator": "yes"}, {"verbosity": -1}]
        for parameters in cases:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                pipeline.generate_patches(**parameters)
        with self.assertRaises(ValueError):
            PatchGenerator({"FLAIR": _image(np.ones((2, 2, 2)))}, mask_names="missing")


class PostprocessingTests(unittest.TestCase):
    def test_default_reference_is_first_nonmask_and_can_be_overridden(self):
        pipeline = PatchGenerator({"VENTRICLES": _image(np.zeros((2, 2, 2))),
                                   "FLAIR": _image(np.ones((2, 2, 2)))},
                                  mask_names="VENTRICLES")
        pipeline.preprocess()
        pipeline.generate_patches(patch_sizes=2)
        default = pipeline.postprocess(max_zero_percent=0)
        self.assertEqual(default["coordinates"].shape[0], 1)
        explicit = pipeline.postprocess(max_zero_percent=0, 
                                        reference="VENTRICLES")
        self.assertEqual(explicit["coordinates"].shape[0], 0)
        restored = pipeline.postprocess(max_zero_percent=0)
        self.assertEqual(restored["coordinates"].shape[0], 1)

    def test_background_threshold_is_inclusive_and_keeps_all_arrays_aligned(self):
        values = np.ones((4, 4, 4), dtype=np.float32)
        values[:2] = 0
        pipeline = PatchGenerator({"FLAIR": _image(values), 
                                   "T1": _image(values + 10)}, annotations=[])
        pipeline.preprocess()
        original = pipeline.generate_patches(patch_sizes=(2, 4))
        retained = pipeline.postprocess(max_zero_percent=50)
        np.testing.assert_array_equal(retained["coordinates"], 
                                      original["coordinates"])
        for name in ("FLAIR", "T1"):
            for size in (2, 4):
                np.testing.assert_array_equal(retained[name][size], 
                                              original[name][size])
        rejected = pipeline.postprocess(max_zero_percent=49.9)
        self.assertEqual(rejected["coordinates"].shape, (0, 3))
        for size in (2, 4):
            self.assertEqual(rejected["label"][size].shape, (0,))
            self.assertEqual(rejected["FLAIR"][size].shape[0], 0)

    def test_background_filter_selects_matching_patch_rows_on_leading_axis(self):
        values = _coordinates((4, 4, 4))
        values[:2] = 0
        pipeline = PatchGenerator({"FLAIR": _image(values), 
                                   "T1": _image(values * 2 + 11)},
                                  annotations=(3, 1, 1))
        pipeline.preprocess()
        original = pipeline.generate_patches(patch_sizes=2, stride=2)
        filtered = pipeline.postprocess(max_zero_percent=0)
        self.assertEqual(filtered["FLAIR"][2].shape, (4, 2, 2, 2))
        np.testing.assert_array_equal(filtered["coordinates"], 
                                      original["coordinates"][4:])
        for name in ("FLAIR", "T1"):
            np.testing.assert_array_equal(filtered[name][2], original[name][2][4:])
        np.testing.assert_array_equal(filtered["label"][2], original["label"][2][4:])
        np.testing.assert_array_equal(filtered["label"][2], [1, 0, 0, 0])

    def test_inconsistent_size_labels_remove_entire_rows_in_both_modes(self):
        values = _coordinates((6, 6, 6))
        for generator_mode in (False, True):
            with self.subTest(generator=generator_mode):
                pipeline = PatchGenerator({"FLAIR": _image(values), 
                                           "T1": _image(values * 2)}, 
                                          annotations=(0, 0, 0))
                pipeline.preprocess()
                original = pipeline.generate_patches(patch_sizes=(2, 4), 
                                                     stride=2)
                expected_coordinates = original["coordinates"][1:].copy()
                expected_arrays = {name: {size: original[name][size][1:].copy() for size in (2, 4)}
                                   for name in ("FLAIR", "T1")}
                pipeline.generate_patches(patch_sizes=(2, 4), stride=2, generator=generator_mode)
                filtered = pipeline.postprocess(remove_inconsistent=True)
                if generator_mode:
                    groups = list(filtered)
                    coordinates = np.concatenate([group["coordinates"] for group in groups])
                    arrays = {name: {size: np.concatenate([group[name][size] for group in groups], 
                                                          axis=0)
                                     for size in (2, 4)} for name in ("FLAIR", "T1")}
                else:
                    coordinates = filtered["coordinates"]
                    arrays = filtered
                np.testing.assert_array_equal(coordinates, expected_coordinates)
                for name in ("FLAIR", "T1"):
                    for size in (2, 4):
                        np.testing.assert_array_equal(arrays[name][size], 
                                                      expected_arrays[name][size])

    def test_masks_only_skip_background_but_still_filter_inconsistent_labels(self):
        for generator_mode in (False, True):
            with self.subTest(generator=generator_mode):
                pipeline = PatchGenerator({"VENTRICLES": _image(np.zeros((6, 6, 6)))},
                                          mask_names="VENTRICLES", annotations=(0, 0, 0))
                pipeline.preprocess()
                pipeline.generate_patches(patch_sizes=(2, 4), stride=2,
                                          generator=generator_mode)
                all_groups = pipeline.postprocess(max_zero_percent=0)
                if generator_mode:
                    self.assertEqual(len(list(all_groups)), 8)
                else:
                    self.assertEqual(all_groups["coordinates"].shape[0], 8)
                consistent = pipeline.postprocess(max_zero_percent=0, 
                                                  remove_inconsistent=True)
                if generator_mode:
                    self.assertEqual(len(list(consistent)), 7)
                else:
                    self.assertEqual(consistent["coordinates"].shape[0], 7)

    def test_none_and_empty_annotations_remain_distinct_after_filtering(self):
        for annotations in (None, [], (1, 1, 1)):
            with self.subTest(annotations=annotations):
                pipeline = PatchGenerator({"FLAIR": _image(np.ones((4, 4, 4)))}, 
                                          annotations=annotations)
                _, scaled = pipeline.preprocess()
                pipeline.generate_patches(patch_sizes=(2, 4))
                result = pipeline.postprocess(max_zero_percent=0, 
                                              remove_inconsistent=True)
                if annotations is None:
                    self.assertIsNone(scaled)
                    self.assertIsNone(result["label"])
                elif len(annotations) == 0:
                    self.assertEqual(scaled.shape, (0, 3))
                    for size in (2, 4):
                        self.assertTrue(np.all(result["label"][size] == 0))
                else:
                    self.assertEqual(scaled.shape, (1, 3))
                    self.assertEqual(scaled.dtype.kind, "f")
                    np.testing.assert_allclose(scaled, [[1, 1, 1]])
                    for size in (2, 4):
                        self.assertEqual(int(result["label"][size][0]), 1)


class StorageTests(unittest.TestCase):
    def test_materialized_save_preserves_filtered_rows_and_latest_array_edits(self):
        values = _coordinates((6, 6, 6))
        pipeline = PatchGenerator({"FLAIR": _image(values), 
                                   "T1": _image(values + 4)}, 
                                  annotations=(0, 0, 0))
        pipeline.preprocess()
        pipeline.generate_patches(patch_sizes=(2, 4), stride=2)
        filtered = pipeline.postprocess(remove_inconsistent=True)
        filtered["FLAIR"][2][0, 0, 0, 0] = -123
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patient.h5"
            self.assertEqual(Path(pipeline.save_hdf5(path)), path)
            with h5py.File(path, "r") as handle:
                self.assertTrue(handle.attrs["complete"])
                self.assertEqual(handle.attrs["algorithm_version"], "0.4.0")
                self.assertEqual(handle.attrs["schema_version"], "3.1")
                self.assertEqual(handle.attrs["mode"], "3D")
                self.assertTrue(handle.attrs["labels_available"])
                np.testing.assert_array_equal(handle["coordinates"][:], 
                                              filtered["coordinates"])
                for name in ("FLAIR", "T1"):
                    for size in (2, 4):
                        dataset = handle[f"images/{name}/{size}"]
                        self.assertEqual(dataset.shape, (7, size, size, size))
                        self.assertEqual(dataset.compression, "lzf")
                        self.assertEqual(dataset.chunks, (1, size, size, size))
                        self.assertEqual(dataset.maxshape, (None, size, size, size))
                        np.testing.assert_array_equal(dataset[:], filtered[name][size])
                for size in (2, 4):
                    np.testing.assert_array_equal(handle[f"label/{size}"][:], 
                                                  filtered["label"][size])
                parameters = json.loads(handle.attrs["parameters_json"])
                self.assertEqual(parameters["generation"]["label_border"], 0)
                self.assertTrue(parameters["postprocessing"]["remove_inconsistent"])

    def test_generator_save_replays_from_start_after_previews(self):
        values = _coordinates((6, 6, 6))
        pipelines = []
        for generator_mode in (False, True):
            pipeline = PatchGenerator({"FLAIR": _image(values), 
                                       "T1": _image(values * 2)}, 
                                      annotations=(0, 0, 0))
            pipeline.preprocess()
            generated = pipeline.generate_patches(patch_sizes=(2, 4), 
                                                  stride=2, generator=generator_mode)
            if generator_mode:
                next(generated)
            filtered = pipeline.postprocess(remove_inconsistent=True)
            if generator_mode:
                next(filtered)
            pipelines.append(pipeline)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "dense.h5", Path(directory) / "stream.h5"]
            for pipeline, path in zip(pipelines, paths):
                pipeline.save_hdf5(path)
            with h5py.File(paths[0], "r") as dense, h5py.File(paths[1], "r") as stream:
                self.assertEqual(stream["coordinates"].shape, (7, 3))
                np.testing.assert_array_equal(stream["coordinates"][:],
                                              dense["coordinates"][:])
                for name in ("FLAIR", "T1"):
                    for size in (2, 4):
                        self.assertEqual(stream[f"images/{name}/{size}"].shape, 
                                         (7, size, size, size))
                        self.assertEqual(stream[f"images/{name}/{size}"].chunks, 
                                         (1, size, size, size))
                        np.testing.assert_array_equal(stream[f"images/{name}/{size}"][:], 
                                                      dense[f"images/{name}/{size}"][:])
                for size in (2, 4):
                    np.testing.assert_array_equal(stream[f"label/{size}"][:], 
                                                  dense[f"label/{size}"][:])

    def test_empty_hdf5_keeps_shape_unknown_labels_and_no_overwrite(self):
        for mode in ("3d", "2d"):
            for annotations in (None, []):
                with self.subTest(mode=mode, annotations=annotations), tempfile.TemporaryDirectory() as directory:
                    pipeline = PatchGenerator({"FLAIR": _image(np.ones((4, 4, 4)))}, 
                                              annotations=annotations)
                    pipeline.preprocess()
                    pipeline.generate_patches(mode=mode, patch_sizes=6, generator=True)
                    pipeline.postprocess(max_zero_percent=0)
                    path = Path(directory) / "empty.h5"
                    pipeline.save_hdf5(path)
                    with h5py.File(path, "r") as handle:
                        self.assertTrue(handle.attrs["complete"])
                        self.assertEqual(handle["coordinates"].shape, (0, 3))
                        dimensions = 2 if mode == "2d" else 3
                        self.assertEqual(handle["images/FLAIR/6"].shape, 
                                         (0,) + (6,) * dimensions)
                        self.assertEqual(bool(handle.attrs["labels_available"]), 
                                         annotations is not None)
                        self.assertEqual(bool(handle.attrs["annotations_available"]), 
                                         annotations is not None)
                        if annotations is None:
                            self.assertEqual(len(handle["label"]), 0)
                        else:
                            self.assertEqual(handle["label/6"].shape, (0,))
                            self.assertEqual(handle["annotations"].shape, (0, 3))
                    with self.assertRaises(FileExistsError):
                        pipeline.save_hdf5(path)
                    with h5py.File(path, "r") as handle:
                        self.assertTrue(handle.attrs["complete"])


class AnnotationIdsTests(unittest.TestCase):
    def test_ids_require_one_positive_integer_per_point(self):
        images = {"FLAIR": _image(np.ones((4, 4, 4)))}
        cases = [(None, [49]), ([(1, 1, 1)], []), ([], [49]),
                 ([(1, 1, 1)], [[49]]), ([(1, 1, 1)], [0]),
                 ([(1, 1, 1)], [-1]), ([(1, 1, 1)], [49.0]),
                 ([(1, 1, 1)], [True]), ([(1, 1, 1)], ["49"]),
                 ([(1, 1, 1)], [2 ** 63]), ([(1, 1, 1)], 49)]
        for points, ids in cases:
            with self.subTest(points=points, ids=ids), self.assertRaises(ValueError):
                PatchGenerator(images, annotations=points, annotations_ids=ids)
        pg = PatchGenerator(images, annotations=(1, 1, 1), annotations_ids=[49])
        np.testing.assert_array_equal(pg.annotations_ids, [49])

    def test_resampling_preserves_ids_and_input_arrays(self):
        ids = np.asarray([49, 49, 52], dtype=np.int64)
        pg = PatchGenerator({"FLAIR": _image(np.ones((3, 3, 3)), 
                                             np.diag([2, 2, 2, 1]))},
                            annotations=[(1, 1, 1), (1.25, 1, 1), (0, 0, 0)], 
                            annotations_ids=ids)
        ids[0] = 99
        for spacing in (1, 0.5):
            _, points = pg.preprocess(target_spacing=spacing)
            np.testing.assert_array_equal(pg.annotations_ids, [49, 49, 52])
            np.testing.assert_allclose(points[0], np.asarray([2, 2, 2]) / spacing)

    def test_multiple_and_repeated_ids_are_deduplicated_per_size_without_changing_labels(self):
        images = {"FLAIR": _image(np.ones((8, 8, 8)))}
        points = [(2, 3, 3), (3, 3, 3), (4, 4, 4), (4.25, 4, 4)]
        baseline = PatchGenerator(images, annotations=points)
        expected = baseline.generate_patches(patch_sizes=(4, 8), label_border=1)
        pg = PatchGenerator(images, annotations=points, annotations_ids=[7, 49, 52, 49])
        result = pg.generate_patches(patch_sizes=(4, 8), label_border=1)
        np.testing.assert_array_equal(result["annotations_ids"][4][0], [49, 52])
        np.testing.assert_array_equal(result["annotations_ids"][8][0], [7, 49, 52])
        for size in (4, 8):
            np.testing.assert_array_equal(result["label"][size], expected["label"][size])
            np.testing.assert_array_equal(result["FLAIR"][size], expected["FLAIR"][size])
        pg = PatchGenerator(images, annotations=(2, 3, 3), annotations_ids=[49])
        result = pg.generate_patches(patch_sizes=(4, 8), label_border=1)
        np.testing.assert_array_equal(result["annotations_ids"][4][0], [0])
        np.testing.assert_array_equal(result["annotations_ids"][8][0], [49])

    def test_id_membership_matches_border_faces_in_3d_and_every_2d_axis(self):
        points = [(0.5, 1, 1), (2.5, 1, 1), (0.49999, 1, 1),
                  (2.50001, 1, 1), (3.5, 1, 1)]
        pg = PatchGenerator({"FLAIR": _image(np.ones((4, 4, 4)))},
                            annotations=points, annotations_ids=[1, 2, 3, 4, 5])
        result = pg.generate_patches(patch_sizes=4, label_border=1)
        np.testing.assert_array_equal(result["annotations_ids"][4][0], [1, 2])
        for axis in range(3):
            for streaming in (False, True):
                result = pg.generate_patches(mode="2d", patch_sizes=4, stride=1,
                                             label_border=1, slice_axis=axis, 
                                             generator=streaming)
                groups = list(result) if streaming else [result]
                for bundle in groups:
                    for row, coordinate in enumerate(bundle["coordinates"]):
                        accepted = []
                        for point, identifier in zip(points, [1, 2, 3, 4, 5]):
                            in_slice = (coordinate[axis] - 0.5 <= 
                                        point[axis] < coordinate[axis] + 0.5)
                            in_plane = (all(0.5 <= point[other] <= 2.5 
                                            for other in range(3) if other != axis))
                            if in_slice and in_plane:
                                accepted.append(identifier)
                        np.testing.assert_array_equal(bundle["annotations_ids"][4][row], 
                                                      accepted or [0])
                        self.assertEqual(bundle["label"][4][row], int(bool(accepted)))

    def test_dense_and_stream_filtering_preserve_id_alignment_and_replay(self):
        values = _coordinates((6, 6, 6))
        values[:2] = 0
        for streaming in (False, True):
            pg = PatchGenerator({"FLAIR": _image(values)},
                                annotations=[(0, 0, 0), (3, 1, 1)], 
                                annotations_ids=[49, 52])
            generated = pg.generate_patches(patch_sizes=(2, 4), 
                                            stride=2, generator=streaming)
            if streaming:
                next(generated)
            filtered = pg.postprocess(max_zero_percent=50, remove_inconsistent=True)
            groups = list(filtered) if streaming else [filtered]
            expected_coordinates = np.concatenate([group["coordinates"] for group in groups])
            expected_ids = {size: np.concatenate([group["annotations_ids"][size] for group in groups])
                            for size in (2, 4)}
            for row, coordinate in enumerate(expected_coordinates):
                ids = [52] if np.array_equal(coordinate, [3.5, 1.5, 1.5]) else [0]
                for size in (2, 4):
                    np.testing.assert_array_equal(expected_ids[size][row], ids)
            with tempfile.TemporaryDirectory() as directory:
                path = pg.save_hdf5(Path(directory) / "ids.h5")
                with h5py.File(path, "r") as saved:
                    self.assertTrue(saved.attrs["annotations_ids_available"])
                    np.testing.assert_array_equal(saved["annotations_ids"][:], [49, 52])
                    np.testing.assert_array_equal(saved["annotations"][:], [[0, 0, 0], [3, 1, 1]])
                    np.testing.assert_array_equal(saved["coordinates"][:], expected_coordinates)
                    for size in (2, 4):
                        self.assertEqual(saved[f"patch_annotations_ids/{size}"].shape, 
                                         (len(expected_coordinates),))
                        for row, ids in enumerate(expected_ids[size]):
                            np.testing.assert_array_equal(saved[f"patch_annotations_ids/{size}"][row], ids)
                            self.assertEqual(saved[f"label/{size}"][row], int(ids[0] != 0))
            restored = pg.postprocess()
            all_coordinates = (np.concatenate([b["coordinates"] for b in restored]) 
                               if streaming else restored["coordinates"])
            self.assertEqual(len(all_coordinates), 8)

    def test_hdf5_stores_ragged_multiple_ids_and_negative_zero(self):
        for streaming in (False, True):
            pg = PatchGenerator({"FLAIR": _image(np.ones((8, 4, 4)))},
                                annotations=[(1, 1, 1), (2, 2, 2)], annotations_ids=[49, 52])
            pg.generate_patches(patch_sizes=4, stride=4, generator=streaming)
            with tempfile.TemporaryDirectory() as directory:
                path = pg.save_hdf5(Path(directory) / "ragged.h5")
                with h5py.File(path, "r") as saved:
                    np.testing.assert_array_equal(saved["patch_annotations_ids/4"][0], [49, 52])
                    np.testing.assert_array_equal(saved["patch_annotations_ids/4"][1], [0])
                    np.testing.assert_array_equal(saved["label/4"][:], [1, 0])

    def test_empty_points_and_empty_grids_save_with_available_ids(self):
        for streaming in (False, True):
            for mode in ("3d", "2d"):
                pg = PatchGenerator({"FLAIR": _image(np.ones((4, 4, 4)))},
                                    annotations=[], annotations_ids=[])
                for size in (2, 6):
                    result = pg.generate_patches(mode=mode, patch_sizes=size, 
                                                 stride=2, generator=streaming)
                    for bundle in (list(result) if streaming else [result]):
                        for ids in bundle["annotations_ids"][size]:
                            np.testing.assert_array_equal(ids, [0])
                    with tempfile.TemporaryDirectory() as directory:
                        path = pg.save_hdf5(Path(directory) / "empty.h5")
                        with h5py.File(path, "r") as saved:
                            self.assertEqual(saved["annotations_ids"].shape, (0,))
                            self.assertTrue(saved.attrs["annotations_ids_available"])
                            self.assertEqual(saved[f"patch_annotations_ids/{size}"].shape,
                                             saved[f"label/{size}"].shape)

    def test_omitted_ids_preserve_original_result_keys_and_hdf5_datasets(self):
        pg = PatchGenerator({"FLAIR": _image(np.ones((4, 4, 4)))}, annotations=(1, 1, 1))
        result = pg.generate_patches(patch_sizes=2)
        self.assertNotIn("annotations_ids", result)
        with tempfile.TemporaryDirectory() as directory:
            path = pg.save_hdf5(Path(directory) / "no_ids.h5")
            with h5py.File(path, "r") as saved:
                self.assertFalse(saved.attrs["annotations_ids_available"])
                self.assertNotIn("annotations_ids", saved)
                self.assertNotIn("patch_annotations_ids", saved)


if __name__ == "__main__":
    unittest.main()
