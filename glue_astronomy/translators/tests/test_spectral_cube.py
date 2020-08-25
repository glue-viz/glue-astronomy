import os

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from spectral_cube import SpectralCube

from astropy import units as u
from astropy.wcs import WCS
from astropy.tests.helper import assert_quantity_allclose

from glue.core import Data, DataCollection
from glue.core.component import Component


@pytest.fixture
def spectral_cube_wcs():
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'VELO-LSR']
    wcs.wcs.set()
    return wcs


def test_to_spectral_cube(spectral_cube_wcs):

    data = Data(label='spectral_cube', coords=spectral_cube_wcs)
    values = np.random.random((4, 5, 3))
    data.add_component(Component(values, units='Jy'), 'x')

    spec = data.get_object(SpectralCube, attribute=data.id['x'])

    assert_quantity_allclose(spec.spectral_axis, [1, 2, 3, 4] * u.m / u.s)
    assert_quantity_allclose(spec.filled_data[...], values * u.Jy)

    data.add_subset(data.id['x'] > 0.5, label='bright')

    spec_subset = data.get_subset_object(cls=SpectralCube, subset_id=0,
                                         attribute=data.id['x'])

    assert_quantity_allclose(spec_subset.spectral_axis, [1, 2, 3, 4] * u.m / u.s)
    expected = values.copy()
    expected[expected <= 0.5] = np.nan
    assert_quantity_allclose(spec_subset.filled_data[...], expected * u.Jy)
    assert_equal(spec_subset.mask.include(), values > 0.5)


def test_to_spectral_cube_unitless(spectral_cube_wcs):

    data = Data(label='spectral_cube', coords=spectral_cube_wcs)
    values = np.random.random((4, 5, 3))
    data.add_component(Component(values), 'x')

    spec = data.get_object(SpectralCube, attribute=data.id['x'])

    assert_quantity_allclose(spec.spectral_axis, [1, 2, 3, 4] * u.m / u.s)
    assert_quantity_allclose(spec.filled_data[...], values * u.one)


def test_to_spectral_cube_invalid_ndim():

    data = Data(label='not-a-spectral-cube')
    data.add_component(Component(np.array([3.4, 2.3, -1.1, 0.3]), units='Jy'), 'x')

    with pytest.raises(ValueError) as exc:
        data.get_object(SpectralCube, attribute=data.id['x'])
    assert exc.value.args[0] == ('Data object should have 3 or 4 dimensions in order '
                                 'to be converted to a SpectralCube object.')


def test_to_spectral_cube_missing_wcs():

    data = Data(label='not-a-spectral-cube')
    values = np.random.random((4, 5, 3))
    data.add_component(Component(values, units='Jy'), 'x')

    with pytest.raises(TypeError) as exc:
        data.get_object(SpectralCube, attribute=data.id['x'])
    assert exc.value.args[0] == ('data.coords should be an instance of BaseLowLevelWCS.')


def test_to_spectral_cube_invalid_wcs():

    wcs = WCS(naxis=3)

    data = Data(label='not-a-spectral-cube', coords=wcs)
    values = np.random.random((4, 5, 3))
    data.add_component(Component(values, units='Jy'), 'x')

    with pytest.raises(ValueError) as exc:
        data.get_object(SpectralCube, attribute=data.id['x'])
    assert exc.value.args[0] == ('No celestial axes found in WCS')

    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', '']

    data = Data(label='not-a-spectral-cube', coords=wcs)
    values = np.random.random((4, 5, 3))
    data.add_component(Component(values, units='Jy'), 'x')

    with pytest.raises(ValueError) as exc:
        data.get_object(SpectralCube, attribute=data.id['x'])
    assert exc.value.args[0] == ('No spectral axes found in WCS')


def test_to_spectral_cube_default_attribute(spectral_cube_wcs):

    data = Data(label='spectral_cube', coords=spectral_cube_wcs)
    values = np.random.random((4, 5, 3))

    with pytest.raises(ValueError) as exc:
        data.get_object(SpectralCube)
    assert exc.value.args[0] == 'Data object has no attributes.'

    data.add_component(Component(values, units='Jy'), 'x')

    spec = data.get_object(SpectralCube)
    assert_quantity_allclose(spec.filled_data[...], values * u.Jy)

    data.add_component(Component(values, units='Jy'), 'y')

    with pytest.raises(ValueError) as exc:
        data.get_object(SpectralCube)
    assert exc.value.args[0] == ('Data object has more than one attribute, so '
                                 'you will need to specify which one to use as '
                                 'the flux for the spectral cube using the attribute= '
                                 'keyword argument.')


def test_from_spectral_cube(spectral_cube_wcs):

    values = np.random.random((4, 5, 3))

    spec = SpectralCube(values * u.Jy, wcs=spectral_cube_wcs)

    data_collection = DataCollection()

    data_collection['spectral-cube'] = spec

    data = data_collection['spectral-cube']

    assert isinstance(data, Data)
    assert len(data.main_components) == 1
    assert data.main_components[0].label == 'flux'
    assert_allclose(data['flux'], values)
    component = data.get_component('flux')
    assert component.units == 'Jy'

    # Check round-tripping
    spec_new = data.get_object(attribute='flux')
    assert isinstance(spec_new, SpectralCube)
    assert_quantity_allclose(spec_new.spectral_axis, [1, 2, 3, 4] * u.m / u.s)
    assert_quantity_allclose(spec_new.filled_data[...], values * u.Jy)


def test_spectral_cube_io():
    # Make sure that when we use the spectral cube I/O from glue-astronomy,
    # glue knows to automatically give a SpectralCube
    from glue_astronomy.io.spectral_cube.spectral_cube import read_spectral_cube
    data = read_spectral_cube(os.path.join(os.path.dirname(__file__), '..', '..',
                                           'io', 'spectral_cube', 'tests',
                                           'data', 'cube_3d.image'))
    assert isinstance(data.get_object(), SpectralCube)


def test_spectral_cube_io_4d():
    # As above, when original data was 4D with a 1-element Stokes axis
    from glue_astronomy.io.spectral_cube.spectral_cube import read_spectral_cube
    data = read_spectral_cube(os.path.join(os.path.dirname(__file__), '..', '..',
                                           'io', 'spectral_cube', 'tests',
                                           'data', 'cube_4d.fits'))
    assert isinstance(data.get_object(), SpectralCube)


def test_spectral_cube_io_4d_fullstokes():
    # As above, when original data was 4D with a 1-element Stokes axis
    from glue_astronomy.io.spectral_cube.spectral_cube import read_spectral_cube
    data = read_spectral_cube(os.path.join(os.path.dirname(__file__), '..', '..',
                                           'io', 'spectral_cube', 'tests',
                                           'data', 'cube_4d_fullstokes.fits'))
    assert isinstance(data.get_object(attribute='STOKES Q'), SpectralCube)


def test_fits_io_4d():
    # This time using the built-in glue FITS reader which returns a 4D dataset
    from glue.core.data_factories.fits import fits_reader
    data = fits_reader(os.path.join(os.path.dirname(__file__), '..', '..',
                                    'io', 'spectral_cube', 'tests',
                                    'data', 'cube_4d.fits'))[0]
    sc = data.get_object(cls=SpectralCube)
    assert isinstance(sc, SpectralCube)
    assert sc.shape == (2, 3, 4)


def test_fits_io_4d_fullstokes():
    # And if there are multiple Stokes axes
    from glue.core.data_factories.fits import fits_reader
    data = fits_reader(os.path.join(os.path.dirname(__file__), '..', '..',
                                    'io', 'spectral_cube', 'tests',
                                    'data', 'cube_4d_fullstokes.fits'))[0]
    sc = data.get_object(cls=SpectralCube)
    assert isinstance(sc, SpectralCube)
    assert sc.shape == (2, 3, 4)


def test_meta_round_trip():
    data = np.array([[[0, 1, 2, 3, 4]]])
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'VELO-HEL']
    meta = {'BUNIT': 'Jy/beam',
            'some_variable': 10}

    cube = SpectralCube(data, wcs=wcs, meta=meta)

    data_collection = DataCollection()

    data_collection['spec_cube'] = cube

    # Test to see if meta exists in glue data object
    glue_data = data_collection['spec_cube']
    assert isinstance(glue_data, Data)
    assert len(glue_data.meta) == 2
    assert glue_data.meta['BUNIT'] == 'Jy/beam'
    assert glue_data.meta['some_variable'] == 10

    # Test to see if meta is included in translated spectral cube instance
    sc_data = data_collection['spec_cube'].get_object()
    assert isinstance(sc_data, SpectralCube)
    assert len(sc_data.meta) == 2
    assert sc_data.meta['BUNIT'] == 'Jy/beam'
    assert sc_data.meta['some_variable'] == 10
