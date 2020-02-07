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


def test_to_spectrum1d_unitless(spectral_cube_wcs):

    data = Data(label='spectral_cube', coords=spectral_cube_wcs)
    values = np.random.random((4, 5, 3))
    data.add_component(Component(values), 'x')

    spec = data.get_object(SpectralCube, attribute=data.id['x'])

    assert_quantity_allclose(spec.spectral_axis, [1, 2, 3, 4] * u.m / u.s)
    assert_quantity_allclose(spec.filled_data[...], values * u.one)


def test_to_spectrum1d_invalid_ndim():

    data = Data(label='not-a-spectral-cube')
    data.add_component(Component(np.array([3.4, 2.3, -1.1, 0.3]), units='Jy'), 'x')

    with pytest.raises(ValueError) as exc:
        data.get_object(SpectralCube, attribute=data.id['x'])
    assert exc.value.args[0] == ('Data object should have 3 dimensions in order '
                                 'to be converted to a SpectralCube object.')


def test_to_spectrum1d_missing_wcs():

    data = Data(label='not-a-spectral-cube')
    values = np.random.random((4, 5, 3))
    data.add_component(Component(values, units='Jy'), 'x')

    with pytest.raises(TypeError) as exc:
        data.get_object(SpectralCube, attribute=data.id['x'])
    assert exc.value.args[0] == ('data.coords should be an instance of BaseLowLevelWCS.')


def test_to_spectrum1d_invalid_wcs():

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


def test_from_spectrum1d(spectral_cube_wcs):

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
