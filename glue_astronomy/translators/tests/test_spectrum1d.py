import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from specutils import Spectrum1D

from astropy import units as u
from astropy.wcs import WCS
from astropy.tests.helper import assert_quantity_allclose
from astropy.nddata import VarianceUncertainty

from glue.core import Data, DataCollection
from glue.core.component import Component

from glue_astronomy.spectral_coordinates import SpectralCoordinates


def test_to_spectrum1d():

    # Set up simple spectral WCS
    wcs = WCS(naxis=1)
    wcs.wcs.ctype = ['VELO-LSR']
    wcs.wcs.set()

    data = Data(label='spectrum', coords=wcs)
    data.add_component(Component(np.array([3.4, 2.3, -1.1, 0.3]), units='Jy'), 'x')

    spec = data.get_object(Spectrum1D, attribute=data.id['x'])

    assert_quantity_allclose(spec.spectral_axis, [1, 2, 3, 4] * u.m / u.s)
    assert_quantity_allclose(spec.flux, [3.4, 2.3, -1.1, 0.3] * u.Jy)

    data.add_subset(data.id['x'] > 1, label='bright')

    spec_subset = data.get_subset_object(cls=Spectrum1D, subset_id=0,
                                         attribute=data.id['x'])

    assert_quantity_allclose(spec_subset.spectral_axis, [1, 2, 3, 4] * u.m / u.s)
    assert_quantity_allclose(spec_subset.flux, [3.4, 2.3, -1.1, 0.3] * u.Jy)
    assert_equal(spec_subset.mask, [0, 0, 1, 1])


def test_to_spectrum1d_unitless():

    # Set up simple spectral WCS
    wcs = WCS(naxis=1)
    wcs.wcs.ctype = ['VELO-LSR']
    wcs.wcs.set()

    data = Data(label='spectrum', coords=wcs)
    data.add_component(Component(np.array([3.4, 2.3, -1.1, 0.3])), 'x')

    spec = data.get_object(Spectrum1D, attribute=data.id['x'])

    assert_quantity_allclose(spec.spectral_axis, [1, 2, 3, 4] * u.m / u.s)
    assert_quantity_allclose(spec.flux, [3.4, 2.3, -1.1, 0.3] * u.one)


def test_to_spectrum1d_invalid():

    data = Data(label='not-a-spectrum')
    data.add_component(Component(np.array([3.4, 2.3, -1.1, 0.3]), units='Jy'), 'x')

    with pytest.raises(TypeError) as exc:
        data.get_object(Spectrum1D, attribute=data.id['x'])
    assert exc.value.args[0] == ('data.coords should be an instance of WCS '
                                 'or SpectralCoordinates')


def test_to_spectrum1d_from_3d_cube():

    # Set up simple spectral WCS
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'VELO-LSR']
    wcs.wcs.set()

    data = Data(label='spectral-cube', coords=wcs)
    data.add_component(Component(np.ones((3, 4, 5)), units='Jy'), 'x')

    spec = data.get_object(Spectrum1D, attribute=data.id['x'], statistic='sum')

    assert_quantity_allclose(spec.spectral_axis, [1, 2, 3] * u.m / u.s)
    assert_quantity_allclose(spec.flux, [20, 20, 20] * u.Jy)


def test_to_spectrum1d_with_spectral_coordinates():

    coords = SpectralCoordinates([1, 4, 10] * u.micron)

    data = Data(label='spectrum1d', coords=coords)
    data.add_component(Component(np.array([3, 4, 5]), units='Jy'), 'x')

    assert_allclose(data.coords.pixel_to_world_values([0, 0.5, 1, 1.5, 2]),
                    [[1, 2.5, 4, 7, 10]])

    spec = data.get_object(Spectrum1D, attribute=data.id['x'])
    assert_quantity_allclose(spec.spectral_axis, [1, 4, 10] * u.micron)
    assert_quantity_allclose(spec.flux, [3, 4, 5] * u.Jy)


def test_to_spectrum1d_default_attribute():

    coords = SpectralCoordinates([1, 4, 10] * u.micron)

    data = Data(label='spectrum1d', coords=coords)

    with pytest.raises(ValueError) as exc:
        data.get_object(Spectrum1D)
    assert exc.value.args[0] == 'Data object has no attributes.'

    data.add_component(Component(np.array([3, 4, 5]), units='Jy'), 'x')

    spec = data.get_object(Spectrum1D)
    assert_quantity_allclose(spec.flux, [3, 4, 5] * u.Jy)

    data.add_component(Component(np.array([3, 4, 5]), units='Jy'), 'y')

    with pytest.raises(ValueError) as exc:
        data.get_object(Spectrum1D)
    assert exc.value.args[0] == ('Data object has more than one attribute, so '
                                 'you will need to specify which one to use as '
                                 'the flux for the spectrum using the attribute= '
                                 'keyword argument.')


@pytest.mark.parametrize('mode', ('wcs', 'lookup'))
def test_from_spectrum1d(mode):

    if mode == 'wcs':
        wcs = WCS(naxis=1)
        wcs.wcs.ctype = ['FREQ']
        wcs.wcs.set()
        kwargs = {'wcs': wcs}
    else:

        kwargs = {'spectral_axis': [1, 2, 3, 4] * u.Hz}

    spec = Spectrum1D([2, 3, 4, 5] * u.Jy,
                      uncertainty=VarianceUncertainty(
                          [0.1, 0.1, 0.1, 0.1] * u.Jy**2),
                      mask=[False, False, False, False],
                      **kwargs)

    data_collection = DataCollection()

    data_collection['spectrum'] = spec

    data = data_collection['spectrum']

    assert isinstance(data, Data)
    assert len(data.main_components) == 3
    assert data.main_components[0].label == 'flux'
    assert_allclose(data['flux'], [2, 3, 4, 5])
    component = data.get_component('flux')
    assert component.units == 'Jy'

    # Check uncertainty parsing within glue data object
    assert data.main_components[1].label == 'uncertainty'
    assert_allclose(data['uncertainty'], [0.1, 0.1, 0.1, 0.1])
    component = data.get_component('uncertainty')
    assert component.units == 'Jy2'

    # Check round-tripping via single attribute reference
    spec_new = data.get_object(attribute='flux')
    assert isinstance(spec_new, Spectrum1D)
    assert_quantity_allclose(spec_new.spectral_axis, [1, 2, 3, 4] * u.Hz)
    assert_quantity_allclose(spec_new.flux, [2, 3, 4, 5] * u.Jy)
    assert spec_new.uncertainty is None

    # Check complete round-tripping, including uncertainties
    spec_new = data.get_object()
    assert isinstance(spec_new, Spectrum1D)
    assert_quantity_allclose(spec_new.spectral_axis, [1, 2, 3, 4] * u.Hz)
    assert_quantity_allclose(spec_new.flux, [2, 3, 4, 5] * u.Jy)
    assert spec_new.uncertainty is not None
    assert_quantity_allclose(spec_new.uncertainty.quantity, [0.1, 0.1, 0.1, 0.1] * u.Jy**2)
