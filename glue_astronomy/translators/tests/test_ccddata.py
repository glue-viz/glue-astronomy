import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from astropy import units as u
from astropy.nddata import CCDData
from astropy.wcs import WCS
from astropy.tests.helper import assert_quantity_allclose

from glue.core import Data, DataCollection
from glue.core.component import Component
from glue.core.coordinates import WCSCoordinates

WCS_CELESTIAL = WCS(naxis=2)
WCS_CELESTIAL.wcs.ctype = ['RA---TAN', 'DEC--TAN']
WCS_CELESTIAL.wcs.set()


@pytest.mark.parametrize('with_wcs', (False, True))
def test_to_ccddata(with_wcs):

    if with_wcs:
        coords = WCSCoordinates(wcs=WCS_CELESTIAL)
    else:
        coords = None

    data = Data(label='image', coords=coords)
    data.add_component(Component(np.array([[3.4, 2.3], [-1.1, 0.3]]), units='Jy'), 'x')

    image = data.get_object(CCDData, attribute=data.id['x'])

    assert image.wcs is (WCS_CELESTIAL if with_wcs else None)
    assert_allclose(image.data, [[3.4, 2.3], [-1.1, 0.3]])
    assert image.unit is u.Jy

    data.add_subset(data.id['x'] > 1, label='bright')

    image_subset = data.get_subset_object(cls=CCDData, subset_id=0,
                                          attribute=data.id['x'])

    assert image_subset.wcs is (WCS_CELESTIAL if with_wcs else None)
    assert_allclose(image_subset.data, [[3.4, 2.3], [np.nan, np.nan]])
    assert image_subset.unit is u.Jy
    assert_equal(image_subset.mask, [[1, 1], [0, 0]])


def test_to_ccddata_unitless():

    coords = WCSCoordinates(wcs=WCS_CELESTIAL)
    data = Data(label='image', coords=coords)
    data.add_component(Component(np.array([[3.4, 2.3], [-1.1, 0.3]])), 'x')

    image = data.get_object(CCDData, attribute=data.id['x'])

    assert_allclose(image.data, [[3.4, 2.3], [-1.1, 0.3]])
    assert image.unit is u.one


def test_to_ccddata_invalid():

    data = Data(label='not-an-image')
    data.add_component(Component(np.array([3.4, 2.3, -1.1, 0.3]), units='Jy'), 'x')

    with pytest.raises(ValueError) as exc:
        data.get_object(CCDData, attribute=data.id['x'])
    assert exc.value.args[0] == 'Only 2-dimensional datasets can be converted to CCDData'


def test_to_ccddata_default_attribute():

    coords = WCSCoordinates(wcs=WCS_CELESTIAL)
    data = Data(label='image', coords=coords)

    with pytest.raises(ValueError) as exc:
        data.get_object(CCDData)
    assert exc.value.args[0] == 'Data object has no attributes.'

    data.add_component(Component(np.array([[3, 4], [5, 6]]), units='Jy'), 'x')

    image = data.get_object(CCDData)
    assert_allclose(image.data, [[3, 4], [5, 6]])
    assert image.unit is u.Jy

    data.add_component(Component(np.array([[3, 4], [5, 6]]), units='Jy'), 'y')

    with pytest.raises(ValueError) as exc:
        data.get_object(CCDData)
    assert exc.value.args[0] == ('Data object has more than one attribute, so '
                                 'you will need to specify which one to use as '
                                 'the flux for the spectrum using the attribute= '
                                 'keyword argument.')


@pytest.mark.parametrize('with_wcs', (False, True))
def test_from_ccddata(with_wcs):

    if with_wcs:
        wcs = WCS_CELESTIAL
    else:
        wcs = None

    spec = CCDData([[2, 3], [4, 5]] * u.Jy, wcs=wcs)

    data_collection = DataCollection()

    data_collection['image'] = spec

    data = data_collection['image']

    assert isinstance(data, Data)
    assert len(data.main_components) == 1
    assert data.main_components[0].label == 'data'
    assert_allclose(data['data'], [[2, 3], [4, 5]])
    component = data.get_component('data')
    assert component.units == 'Jy'

    # Check round-tripping
    image_new = data.get_object(attribute='data')
    assert isinstance(image_new, CCDData)
    assert image_new.wcs is (WCS_CELESTIAL if with_wcs else None)
    assert_allclose(image_new.data, [[2, 3], [4, 5]])
    assert image_new.unit is u.Jy
