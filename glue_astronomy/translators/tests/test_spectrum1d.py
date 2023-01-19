import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from specutils import Spectrum1D

from astropy import units as u
from astropy.wcs import WCS
from astropy.tests.helper import assert_quantity_allclose
from astropy.nddata import VarianceUncertainty
from astropy.coordinates import SpectralCoord
from astropy.utils.exceptions import AstropyUserWarning

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
                    [1, 2.5, 4, 7, 10])

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


@pytest.mark.filterwarnings('ignore:Input WCS indicates that the spectral axis is not last')
@pytest.mark.parametrize('mode', ('wcs1d', 'wcs3d', 'lookup'))
def test_from_spectrum1d(mode):

    if mode == 'wcs3d':
        # This test is intended to be run with the version of Spectrum1D based
        # on NDCube 2.0
        pytest.importorskip("ndcube", minversion="1.99")

        # Set up simple spatial+spectral WCS
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'FREQ']
        wcs.wcs.set()
        flux = np.ones((4, 4, 5))*u.Unit('Jy')
        uncertainty = VarianceUncertainty(np.square(flux*0.1))
        mask = np.zeros((4, 4, 5))
        kwargs = {'wcs': wcs, 'uncertainty': uncertainty, 'mask': mask}
    else:
        flux = [2, 3, 4, 5] * u.Jy
        uncertainty = VarianceUncertainty([0.1, 0.1, 0.1, 0.1] * u.Jy**2)
        mask = [False, False, False, False]
        if mode == 'wcs1d':
            wcs = WCS(naxis=1)
            wcs.wcs.ctype = ['FREQ']
            wcs.wcs.set()
            kwargs = {'wcs': wcs, 'uncertainty': uncertainty, 'mask': mask}
        else:
            kwargs = {'spectral_axis': [1, 2, 3, 4] * u.Hz,
                      'uncertainty': uncertainty, 'mask': mask}

    spec = Spectrum1D(flux, **kwargs)

    data_collection = DataCollection()

    data_collection['spectrum'] = spec

    data = data_collection['spectrum']

    assert isinstance(data, Data)
    assert len(data.main_components) == 3
    assert data.main_components[0].label == 'flux'
    assert_allclose(data['flux'], spec.flux.value)
    component = data.get_component('flux')
    assert component.units == 'Jy'

    # Check uncertainty parsing within glue data object
    assert data.main_components[1].label == 'uncertainty'
    assert_allclose(data['uncertainty'], spec.uncertainty.array)
    component = data.get_component('uncertainty')
    assert component.units == 'Jy2'

    # Check round-tripping via single attribute reference
    spec_new = data.get_object(attribute='flux', statistic=None)
    assert isinstance(spec_new, Spectrum1D)
    assert_quantity_allclose(spec_new.spectral_axis, [1, 2, 3, 4] * u.Hz)
    if mode == 'wcs3d':
        assert_quantity_allclose(spec_new.flux, np.ones((5, 4, 4))*u.Unit('Jy'))
    else:
        assert_quantity_allclose(spec_new.flux, [2, 3, 4, 5] * u.Jy)
    assert spec_new.uncertainty is None

    # Check complete round-tripping, including uncertainties
    spec_new = data.get_object(statistic=None)
    assert isinstance(spec_new, Spectrum1D)
    assert_quantity_allclose(spec_new.spectral_axis, [1, 2, 3, 4] * u.Hz)
    if mode == 'wcs3d':
        assert_quantity_allclose(spec_new.flux, np.ones((5, 4, 4))*u.Unit('Jy'))
        assert spec_new.uncertainty is not None
        assert_quantity_allclose(spec_new.uncertainty.quantity,
                                 np.ones((5, 4, 4))*0.01*u.Jy**2)
    else:
        assert_quantity_allclose(spec_new.flux, [2, 3, 4, 5] * u.Jy)
        assert spec_new.uncertainty is not None
        assert_quantity_allclose(spec_new.uncertainty.quantity, [0.1, 0.1, 0.1, 0.1] * u.Jy**2)


@pytest.mark.parametrize('spec_ndim', (2, 3))
def test_spectrum1d_2d_data(spec_ndim):

    # This test makes sure that 2D spectra represented as Spectrum1D round-trip
    # Note that Spectrum1D will typically have a 1D spectral WCS even if the
    # data is N-dimensional, so we need to pad the WCS before passing it to
    # glue and un-pad it when translating back.

    # We test both the case where the WCS is 2D and the case where it is 1D

    wcs = WCS(naxis=1)
    wcs.wcs.ctype = ['FREQ']
    wcs.wcs.cdelt = [10]
    wcs.wcs.set()

    if spec_ndim == 2:
        flux = np.ones((3, 2)) * u.Unit('Jy')
    elif spec_ndim == 3:
        flux = np.ones((3, 3, 2)) * u.Unit('Jy')

    spec = Spectrum1D(flux, wcs=wcs, meta={'instrument': 'spamcam'})

    assert spec.data.ndim == spec_ndim
    assert spec.wcs.naxis == 1

    data_collection = DataCollection()

    data_collection['spectrum'] = spec

    data = data_collection['spectrum']

    assert isinstance(data, Data)
    assert len(data.main_components) == 1
    assert data.main_components[0].label == 'flux'
    assert_allclose(data['flux'], flux.value)
    assert data.get_component('flux').units == 'Jy'

    assert data.coords.pixel_n_dim == spec_ndim
    assert data.coords.world_n_dim == spec_ndim
    assert len(data.pixel_component_ids) == spec_ndim
    assert len(data.world_component_ids) == spec_ndim

    if spec_ndim == 2:
        assert data.coordinate_components[0].label == 'Pixel Axis 0 [y]'
        assert data.coordinate_components[1].label == 'Pixel Axis 1 [x]'
        assert data.coordinate_components[2].label == 'Offset'
        assert data.coordinate_components[3].label == 'Frequency'
        assert data.coords.pixel_axis_names == ('', 'spatial')

        assert_equal(data['Offset'], [[0, 0], [1, 1], [2, 2]])
        assert_equal(data['Frequency'], [[10, 20], [10, 20], [10, 20]])

        s, o = data.coords.pixel_to_world(1, 2)
        assert isinstance(s, SpectralCoord)

        # Check round-tripping of coordinates
        with pytest.warns(AstropyUserWarning, match='No observer defined on WCS'):
            px, py = data.coords.world_to_pixel(s, o)
        assert_allclose(px, 1)
        assert_allclose(py, 2)

    elif spec_ndim == 3:
        assert data.coordinate_components[0].label == 'Pixel Axis 0 [z]'
        assert data.coordinate_components[1].label == 'Pixel Axis 1 [y]'
        assert data.coordinate_components[2].label == 'Pixel Axis 2 [x]'
        assert data.coordinate_components[3].label == 'Offset1'
        assert data.coordinate_components[4].label == 'Offset0'
        assert data.coordinate_components[5].label == 'Frequency'
        assert data.coords.pixel_axis_names == ('', 'spatial0', 'spatial1')

        assert_equal(data['Offset1'], [[[0, 0], [0, 0], [0, 0]],
                                       [[1, 1], [1, 1], [1, 1]],
                                       [[2, 2], [2, 2], [2, 2]]])
        assert_equal(data['Offset0'], [[[0, 0], [1, 1], [2, 2]],
                                       [[0, 0], [1, 1], [2, 2]],
                                       [[0, 0], [1, 1], [2, 2]]])
        assert_equal(data['Frequency'], [[[10, 20], [10, 20], [10, 20]],
                                         [[10, 20], [10, 20], [10, 20]],
                                         [[10, 20], [10, 20], [10, 20]]])

        s, o1, o2 = data.coords.pixel_to_world(1, 2, 0)
        assert isinstance(s, SpectralCoord)

        # Check round-tripping of coordinates
        with pytest.warns(AstropyUserWarning, match='No observer defined on WCS'):
            px, py, pz = data.coords.world_to_pixel(s, o1, o2)
        assert_allclose(px, 1)
        assert_allclose(py, 2)
        assert_allclose(pz, 0)

    assert data.coords.world_axis_units == ('Hz', *(None,)*(spec_ndim-1))
    assert data.coords.world_axis_physical_types == ['em.freq', *(None,)*(spec_ndim-1)]

    # Check round-tripping of translation
    spec_new = data.get_object(statistic=None)
    assert isinstance(spec_new, Spectrum1D)

    # The WCS object should be the same
    assert spec_new.wcs.pixel_n_dim == 1
    assert spec_new.wcs.world_n_dim == 1
    assert spec_new.wcs is spec.wcs

    # The metadata should still be present
    assert spec_new.meta['instrument'] == 'spamcam'
