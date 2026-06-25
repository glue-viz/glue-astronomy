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
from glue.viewers.image.pixel_selection_subset_state import PixelSubsetState
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


@pytest.fixture(scope="session")
def example_hips3d_deep_dataset(tmp_path_factory):

    # A HiPS3D with several levels, so that the spectral axis is downsampled
    # (by a non power-of-two factor) between levels - unlike the shallow
    # fixture above where it stays the same size.

    array = np.arange(64 * 120 * 120).reshape((64, 120, 120)).astype(float)

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = 'RA---TAN', 'DEC--TAN', 'FREQ'
    wcs.wcs.crval = 20, 40, 1e9
    wcs.wcs.cdelt = -0.05, 0.05, 1e8
    wcs.wcs.crpix = 60, 60, 1

    hips_directory = tmp_path_factory.mktemp('hips') / 'hips3d_deep'

    reproject_to_hips((array, wcs), output_directory=hips_directory,
                       coord_system_out='equatorial',
                       reproject_function=reproject_interp, level=2,
                       tile_size=64,
                       tile_depth=4)

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
    # Locate a full-resolution spatial pixel that contains data: find the
    # footprint on the (small) coarsest level, then scan the corresponding
    # full-resolution tile for a finite pixel.
    coarse = np.asarray(hips_data._dask_arrays[0])
    cys, cxs = np.where(np.isfinite(coarse).any(axis=0))
    assert len(cys) > 0
    middle = len(cys) // 2
    step = hips_data._array.chunksize
    y0 = (int(cys[middle] * hips_data.shape[1] / coarse.shape[1]) // step[1]) * step[1]
    x0 = (int(cxs[middle] * hips_data.shape[2] / coarse.shape[2]) // step[2]) * step[2]
    block = np.asarray(hips_data._array[0:hips_data.shape[0],
                                        y0:y0 + step[1], x0:x0 + step[2]])
    ys, xs = np.where(np.isfinite(block).any(axis=0))
    assert len(ys) > 0
    middle = len(ys) // 2
    return y0 + int(ys[middle]), x0 + int(xs[middle])


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


def test_hips3d_profile_coarse_spectral_range(example_hips3d_deep_dataset):

    # When a large subset forces the profile to be computed from a coarser
    # level, the spectral axis is downsampled by a non power-of-two factor, so
    # the coarse profile must be mapped back to full resolution via the spectral
    # WCS. Otherwise it would be shifted and cover the wrong spectral range.

    hips_data = HiPSData(example_hips3d_deep_dataset, label='HiPS3D Deep')

    # The spectral axis really is downsampled between levels here.
    assert hips_data._dask_arrays[0].shape[0] < hips_data.shape[0]

    px = hips_data.pixel_component_ids
    coarse = np.asarray(hips_data._dask_arrays[0])
    cy, cx = np.where(np.isfinite(coarse).any(axis=0))
    fy = hips_data.shape[1] / coarse.shape[1]
    fx = hips_data.shape[2] / coarse.shape[2]
    y0, y1 = int(cy.min() * fy), int(cy.max() * fy)
    x0, x1 = int(cx.min() * fx), int(cx.max() * fx)
    big = (px[1] >= y0) & (px[1] <= y1) & (px[2] >= x0) & (px[2] <= x1)

    cid = hips_data.main_components[0]
    full = hips_data.compute_statistic('mean', cid, axis=(1, 2),
                                       subset_state=big, max_load=10 ** 12)
    coarse_profile = hips_data.compute_statistic('mean', cid, axis=(1, 2),
                                                 subset_state=big, max_load=200000)

    full_finite = np.where(np.isfinite(full))[0]
    coarse_finite = np.where(np.isfinite(coarse_profile))[0]

    # Both profiles span the full-resolution spectral length and cover the same
    # spectral range (to within a channel of the nearest-neighbour mapping).
    assert len(full) == len(coarse_profile) == hips_data.shape[0]
    assert abs(int(full_finite.min()) - int(coarse_finite.min())) <= 2
    assert abs(int(full_finite.max()) - int(coarse_finite.max())) <= 2


def test_hips3d_wcs_override(example_hips3d_dataset):

    # The third axis of a HiPS3D is stored as a spectral axis, but a user may be
    # using it to represent something else (e.g. distance). wcs_override lets
    # them relabel it; the original WCS is kept internally for the
    # multi-resolution mapping.

    def to_distance(wcs):
        wcs.wcs.ctype[2] = 'DIST'
        wcs.wcs.cunit[2] = 'kpc'
        wcs.wcs.crpix[2] = 1
        wcs.wcs.crval[2] = 0
        wcs.wcs.cdelt[2] = 10
        wcs.wcs.set()
        return wcs

    hips_data = HiPSData(example_hips3d_dataset, label='HiPS3D Data',
                         wcs_override=to_distance)

    # The public coords reflect the override...
    assert hips_data.coords.world_axis_units[2] == 'kpc'
    assert hips_data.coords.wcs.ctype[2] == 'DIST'
    assert [c.label for c in hips_data.world_component_ids][0] == 'Dist'

    # ...but the WCS used internally for the multi-resolution mapping is intact.
    assert hips_data._wcs.wcs.ctype[2] == 'FREQ-LOG'

    # And statistics still work (they do not depend on the public coords).
    yc, xc = _find_data_pixel(hips_data)
    px = hips_data.pixel_component_ids
    subset_state = ((px[1] > yc - 0.5) & (px[1] < yc + 0.5) &
                    (px[2] > xc - 0.5) & (px[2] < xc + 0.5))
    profile = hips_data.compute_statistic('mean', hips_data.main_components[0],
                                          axis=(1, 2), subset_state=subset_state)
    assert len(profile) == hips_data.shape[0]

    # The default (no override) keeps the spectral WCS as the public coords.
    default = HiPSData(example_hips3d_dataset, label='default')
    assert default.coords.wcs.ctype[2] == 'FREQ-LOG'


def test_hips3d_pixel_subset_state(example_hips3d_dataset):

    # The single-pixel selection tool produces a PixelSubsetState (a slice-based
    # subset). Its bounding box must be read directly off the slices, and its
    # mask evaluated without ever allocating a full-resolution array (which
    # SliceSubsetState.to_mask does for an array-style view).

    hips_data = HiPSData(example_hips3d_dataset, label='HiPS3D Data')
    yc, xc = _find_data_pixel(hips_data)

    slices = [slice(None)] * hips_data.ndim
    slices[1] = slice(yc, yc + 1)
    slices[2] = slice(xc, xc + 1)
    subset_state = PixelSubsetState(hips_data, slices)

    # Bounding box comes straight from the slices.
    box = hips_data._bounding_box(subset_state, 40000000)
    assert box[0] == (0, hips_data.shape[0])
    assert box[1] == (yc, yc + 1)
    assert box[2] == (xc, xc + 1)

    profile = hips_data.compute_statistic('mean', hips_data.main_components[0],
                                          axis=(1, 2), subset_state=subset_state)
    assert len(profile) == hips_data.shape[0]
    assert np.isfinite(profile).any()


def test_hips3d_pixel_subset_single_tile(example_hips3d_deep_dataset):

    # Regression test: a single-pixel selection on a HiPS where one full
    # resolution tile is larger than the load budget. The pixel spans a single
    # spatial tile, so it must still be read at full resolution (giving the
    # exact spectrum), not coarsened to a level where the slice mask would miss
    # the pixel entirely and the profile would come out empty.

    hips_data = HiPSData(example_hips3d_deep_dataset, label='HiPS3D Deep')
    yc, xc = _find_data_pixel(hips_data)

    slices = [slice(None)] * hips_data.ndim
    slices[1] = slice(yc, yc + 1)
    slices[2] = slice(xc, xc + 1)
    subset_state = PixelSubsetState(hips_data, slices)

    box = hips_data._bounding_box(subset_state, 40000000)
    order = len(hips_data._dask_arrays) - 1

    # A tile is bigger than this tiny budget, but the single tile must still be
    # read at full resolution rather than coarsened.
    tile_volume = int(np.prod(hips_data._array.chunksize))
    tiny = tile_volume // 2
    level, level_box = hips_data._select_level(box, tiny)
    assert level == order

    # The mask selects the pixel across all spectral channels.
    mask = hips_data._level_mask(subset_state, level, level_box)
    assert mask.sum() == hips_data.shape[0]

    # Even forced onto the coarsest level, the overlap-based selection picks the
    # cell containing the pixel rather than missing it (which used to give an
    # empty mask, and hence an empty profile).
    coarse = hips_data._dask_arrays[0]
    coarse_box = []
    for axis in range(hips_data.ndim):
        factor = hips_data.shape[axis] / coarse.shape[axis]
        step = coarse.chunksize[axis]
        clo = (int(box[axis][0] / factor) // step) * step
        chi = min(coarse.shape[axis], clo + step)
        coarse_box.append((clo, chi))
    assert hips_data._level_mask(subset_state, 0, coarse_box).sum() >= 1

    # End to end: the profile is full length and not empty, even with a tiny
    # budget that would otherwise force coarsening.
    profile = hips_data.compute_statistic('mean', hips_data.main_components[0],
                                          axis=(1, 2), subset_state=subset_state,
                                          max_load=tiny)
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
