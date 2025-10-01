import pytest

import numpy as np
from astropy.wcs import WCS
from glue_astronomy.data.hips import HiPSData
from glue.tests.visual.helpers import visual_test
from glue.viewers.image.viewer import SimpleImageViewer
from glue.core.application_base import Application
from echo import delay_callback

try:
    from reproject import reproject_interp
    from reproject.hips import reproject_to_hips
except ImportError:
    REPROJECT_INSTALLED = False
else:
    REPROJECT_INSTALLED = True


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


@pytest.mark.skipif("not REPROJECT_INSTALLED")
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
