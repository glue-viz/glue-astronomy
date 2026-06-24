import pytest

import numpy as np
from astropy.wcs import WCS
from glue_astronomy.data.hips import HiPSData
from glue.tests.visual.helpers import visual_test
from glue.viewers.image.viewer import SimpleImageViewer
from glue.viewers.profile.viewer import SimpleProfileViewer
from glue.viewers.histogram.viewer import SimpleHistogramViewer
from glue.core.application_base import Application
from glue.core.roi import RectangularROI
from glue.core.subset import RoiSubsetState
from echo import delay_callback

try:
    from reproject import reproject_interp
    from reproject.hips import reproject_to_hips
except ImportError:
    pytest.skip(allow_module_level=True)


@pytest.fixture(scope="session")
def example_hips_dataset(tmp_path_factory):

    array = np.arange(20000).reshape((100, 200))

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = 'RA---TAN', 'DEC--TAN'
    wcs.wcs.crval = 20, 40
    wcs.wcs.cdelt = -0.02, 0.02
    wcs.wcs.crpix = 30, 35

    hips_directory = tmp_path_factory.mktemp('hips') / 'hips'

    reproject_to_hips((array, wcs), output_directory=hips_directory,
                       coord_system_out='equatorial',
                       reproject_function=reproject_interp, level=3)

    return hips_directory


@pytest.fixture(scope="session")
def example_hips3d_dataset(tmp_path_factory):

    array = np.arange(50000).reshape((10, 50, 100))

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = 'RA---TAN', 'DEC--TAN', 'FREQ'
    wcs.wcs.crval = 20, 40, 1e9
    wcs.wcs.cdelt = -0.2, 0.2, 1e8
    wcs.wcs.crpix = 30, 35, 1

    hips_directory = tmp_path_factory.mktemp('hips') / 'hips3d'

    reproject_to_hips((array, wcs), output_directory=hips_directory,
                       coord_system_out='equatorial',
                       reproject_function=reproject_interp, level=1,
                       tile_size=256,
                       tile_depth=8)

    return hips_directory


@visual_test
def test_hips_data_image(example_hips_dataset):

    hips_data = HiPSData(example_hips_dataset, label='HiPS Data')

    app = Application()
    app.data_collection.append(hips_data)

    viewer = app.new_data_viewer(SimpleImageViewer)
    viewer.add_data(hips_data)

    with delay_callback(viewer.state, 'x_min', 'x_max', 'y_min', 'y_max'):
        viewer.state.x_min = 11000
        viewer.state.x_max = 11600
        viewer.state.y_min = 7000
        viewer.state.y_max = 7800

    app.data_collection.new_subset_group(label='subset1',
                                         subset_state=hips_data.main_components[0] > 10000)
    app.data_collection.new_subset_group(label='subset2',
                                         subset_state=hips_data.pixel_component_ids[1] > 11400)

    return viewer.figure


@visual_test
def test_hips3d_data_image(example_hips3d_dataset):

    print(example_hips3d_dataset)

    hips_data = HiPSData(example_hips3d_dataset, label='HiPS3D Data')

    app = Application()
    app.data_collection.append(hips_data)

    viewer = app.new_data_viewer(SimpleImageViewer)
    viewer.add_data(hips_data)

    assert viewer.layers[0].enabled

    with delay_callback(viewer.state, 'x_min', 'x_max', 'y_min', 'y_max'):
        viewer.state.x_min = 1300
        viewer.state.x_max = 1600
        viewer.state.y_min = 800
        viewer.state.y_max = 1100

    viewer.state.slices = (11, 0, 0)

    app.data_collection.new_subset_group(label='subset1',
                                         subset_state=hips_data.main_components[0] > 33000)
    app.data_collection.new_subset_group(label='subset2',
                                         subset_state=hips_data.pixel_component_ids[2] > 1500)

    return viewer.figure


def _find_data_pixel(hips_data):
    # Locate a full-resolution spatial pixel that contains data, by scanning a
    # tile-aligned block over the region known to contain the dataset footprint.
    block = np.asarray(hips_data._array[0:hips_data.shape[0], 768:1280, 1280:1792])
    finite = np.isfinite(block).any(axis=0)
    ys, xs = np.where(finite)
    assert len(ys) > 0
    middle = len(ys) // 2
    return 768 + int(ys[middle]), 1280 + int(xs[middle])


def test_hips3d_profile(example_hips3d_dataset):

    # Regression test for a shape mismatch that occurred when showing a HiPS3D
    # dataset in the profile viewer with the profile running along a spatial
    # (and therefore downsampled) axis: the profile x axis is at full
    # resolution but compute_statistic used to return a coarse-resolution
    # profile, so the two could not be broadcast together when drawing.

    hips_data = HiPSData(example_hips3d_dataset, label='HiPS3D Data')

    app = Application()
    app.data_collection.append(hips_data)

    viewer = app.new_data_viewer(SimpleProfileViewer)
    viewer.add_data(hips_data)

    # Run the profile along a spatial axis - this is what used to crash.
    viewer.state.x_att = hips_data.pixel_component_ids[1]

    viewer.figure.canvas.draw()

    x, y = viewer.layers[0].state.profile
    assert len(x) == len(y) == hips_data.shape[1]


def test_hips3d_profile_subset(example_hips3d_dataset):

    hips_data = HiPSData(example_hips3d_dataset, label='HiPS3D Data')

    app = Application()
    app.data_collection.append(hips_data)

    image = app.new_data_viewer(SimpleImageViewer)
    image.add_data(hips_data)

    yc, xc = _find_data_pixel(hips_data)

    px = hips_data.pixel_component_ids
    subset_state = ((px[1] > yc - 0.5) & (px[1] < yc + 0.5) &
                    (px[2] > xc - 0.5) & (px[2] < xc + 0.5))
    app.data_collection.new_subset_group(label='pixel', subset_state=subset_state)

    viewer = app.new_data_viewer(SimpleProfileViewer)
    viewer.add_data(hips_data)

    viewer.figure.canvas.draw()

    for layer in viewer.layers:
        x, y = layer.state.profile
        assert len(x) == len(y)

    # The single-pixel spectral profile of the subset should be at full
    # resolution and contain real (finite) values.
    subset = app.data_collection.subset_groups[0].subsets[0]
    profile = hips_data.compute_statistic('mean', hips_data.main_components[0],
                                          axis=(1, 2),
                                          subset_state=subset.subset_state)
    assert len(profile) == hips_data.shape[0]
    assert np.isfinite(profile).any()


def test_hips3d_profile_roi_subset(example_hips3d_dataset):

    # Regression test: a small ROI selection (as produced by the selection
    # tool) must not cause the full-resolution mask to be evaluated, which
    # would use a prohibitive amount of memory for a large HiPS. We check that
    # the bounding box is located cheaply (it is tiny, not the whole array) and
    # that the resulting profile is correct.

    hips_data = HiPSData(example_hips3d_dataset, label='HiPS3D Data')

    app = Application()
    app.data_collection.append(hips_data)

    image = app.new_data_viewer(SimpleImageViewer)
    image.add_data(hips_data)

    yc, xc = _find_data_pixel(hips_data)
    px = hips_data.pixel_component_ids
    roi = RectangularROI(xmin=xc - 0.5, xmax=xc + 0.5, ymin=yc - 0.5, ymax=yc + 0.5)
    subset_state = RoiSubsetState(xatt=px[2], yatt=px[1], roi=roi)
    app.data_collection.new_subset_group(label='roi', subset_state=subset_state)

    # The bounding box should be tiny (a single spatial pixel), proving the
    # subset was located without scanning the full-resolution array.
    box = hips_data._bounding_box(subset_state, 40000000)
    assert box[1] == (yc, yc + 1)
    assert box[2] == (xc, xc + 1)

    viewer = app.new_data_viewer(SimpleProfileViewer)
    viewer.add_data(hips_data)
    viewer.figure.canvas.draw()

    for layer in viewer.layers:
        x, y = layer.state.profile
        assert len(x) == len(y)

    subset = app.data_collection.subset_groups[0].subsets[0]
    profile = hips_data.compute_statistic('mean', hips_data.main_components[0],
                                          axis=(1, 2),
                                          subset_state=subset.subset_state)
    assert len(profile) == hips_data.shape[0]
    assert np.isfinite(profile).any()


def test_hips3d_bounding_box_methods(example_hips3d_dataset):

    hips_data = HiPSData(example_hips3d_dataset, label='HiPS3D Data')
    yc, xc = _find_data_pixel(hips_data)
    px = hips_data.pixel_component_ids

    # A large ROI is found by the coarse grid search and refined.
    big = RoiSubsetState(xatt=px[2], yatt=px[1],
                         roi=RectangularROI(xmin=xc - 200, xmax=xc + 200,
                                            ymin=yc - 200, ymax=yc + 200))
    box = hips_data._bounding_box(big, 40000000)
    assert box[1][0] <= yc - 200 + 1 and box[1][1] >= yc + 200
    assert box[2][0] <= xc - 200 + 1 and box[2][1] >= xc + 200

    # A single-pixel ROI is too small for the coarse search but is located via
    # the subset centre.
    small = RoiSubsetState(xatt=px[2], yatt=px[1],
                           roi=RectangularROI(xmin=xc - 0.5, xmax=xc + 0.5,
                                              ymin=yc - 0.5, ymax=yc + 0.5))
    assert hips_data._bounding_box(small, 40000000)[1] == (yc, yc + 1)

    # A subset with no centre still works via the chunked fallback.
    inequality = (px[1] > yc - 0.5) & (px[1] < yc + 0.5) & \
                 (px[2] > xc - 0.5) & (px[2] < xc + 0.5)
    box = hips_data._bounding_box(inequality, 40000000)
    assert box[1] == (yc, yc + 1) and box[2] == (xc, xc + 1)

    # An empty subset returns None.
    empty = RoiSubsetState(xatt=px[2], yatt=px[1],
                           roi=RectangularROI(xmin=-10, xmax=-5, ymin=-10, ymax=-5))
    assert hips_data._bounding_box(empty, 40000000) is None


def test_hips3d_compute_statistic_shapes(example_hips3d_dataset):

    hips_data = HiPSData(example_hips3d_dataset, label='HiPS3D Data')

    yc, xc = _find_data_pixel(hips_data)
    px = hips_data.pixel_component_ids
    cid = hips_data.main_components[0]

    pixel = ((px[1] > yc - 0.5) & (px[1] < yc + 0.5) &
             (px[2] > xc - 0.5) & (px[2] < xc + 0.5))

    # Collapsing along different axes must always return a profile that matches
    # the full-resolution shape of the data along the remaining axis.
    assert hips_data.compute_statistic('mean', cid, axis=(1, 2),
                                       subset_state=pixel).shape == (hips_data.shape[0],)
    assert hips_data.compute_statistic('mean', cid, axis=(0, 2),
                                       subset_state=pixel).shape == (hips_data.shape[1],)

    # A scalar statistic over a subset returns a single value.
    assert np.ndim(hips_data.compute_statistic('mean', cid, subset_state=pixel)) == 0

    # A small subset stays at full resolution, a subset spanning the whole cube
    # falls back to a coarser level so the load stays bounded.
    order = len(hips_data._dask_arrays) - 1
    small_level, _ = hips_data._select_level(
        hips_data._bounding_box(pixel, 40000000), 40000000)
    assert small_level == order

    everything = px[1] >= 0
    big_level, _ = hips_data._select_level(
        hips_data._bounding_box(everything, 40000000), 40000000)
    assert big_level < order


def test_hips3d_histogram(example_hips3d_dataset):

    hips_data = HiPSData(example_hips3d_dataset, label='HiPS3D Data')

    app = Application()
    app.data_collection.append(hips_data)

    image = app.new_data_viewer(SimpleImageViewer)
    image.add_data(hips_data)

    yc, xc = _find_data_pixel(hips_data)
    px = hips_data.pixel_component_ids
    subset_state = ((px[1] > yc - 0.5) & (px[1] < yc + 0.5) &
                    (px[2] > xc - 0.5) & (px[2] < xc + 0.5))
    app.data_collection.new_subset_group(label='pixel', subset_state=subset_state)

    viewer = app.new_data_viewer(SimpleHistogramViewer)
    viewer.add_data(hips_data)

    viewer.figure.canvas.draw()

    n_bin = viewer.state.hist_n_bin
    for layer in viewer.layers:
        _edges, values = layer.state.histogram
        assert len(values) == n_bin
        assert np.all(np.isfinite(values))

    # The data layer histogram should contain counts, and the subset histogram
    # should be a strict subset of those counts.
    data_total = np.nansum(viewer.layers[0].state.histogram[1])
    subset_total = np.nansum(viewer.layers[1].state.histogram[1])
    assert data_total > 0
    assert 0 < subset_total <= data_total


def test_hips3d_compute_histogram(example_hips3d_dataset):

    hips_data = HiPSData(example_hips3d_dataset, label='HiPS3D Data')
    cid = hips_data.main_components[0]

    vmin = hips_data.compute_statistic('minimum', cid)
    vmax = hips_data.compute_statistic('maximum', cid)

    # Histogram over the whole dataset.
    hist = hips_data.compute_histogram([cid], range=[(vmin, vmax)], bins=[10],
                                       log=[False])
    assert hist.shape == (10,)
    assert hist.sum() > 0

    # Histogram over a single-pixel subset is computed at full resolution, so
    # the (un-scaled) counts equal the number of finite values at that pixel.
    yc, xc = _find_data_pixel(hips_data)
    px = hips_data.pixel_component_ids
    subset_state = ((px[1] > yc - 0.5) & (px[1] < yc + 0.5) &
                    (px[2] > xc - 0.5) & (px[2] < xc + 0.5))
    hist = hips_data.compute_histogram([cid], range=[(vmin, vmax)], bins=[10],
                                       log=[False], subset_state=subset_state)
    assert hist.sum() > 0
    assert hist.sum() <= hips_data.shape[0]
