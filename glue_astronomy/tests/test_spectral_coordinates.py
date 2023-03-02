import pytest
import numpy as np
from astropy import units as u
from numpy.testing import assert_allclose
from glue_astronomy.spectral_coordinates import SpectralCoordinates


def test_basic():

    sc = SpectralCoordinates([10, 20, 30] * u.Hz)

    assert_allclose(sc.pixel_to_world_values(0), 10)
    assert_allclose(sc.pixel_to_world_values([0, 1]), [10, 20])
    assert_allclose(sc.pixel_to_world_values([-0.5, 0, 0.5, 1, 10]),
                    [np.nan, 10, 15, 20, np.nan])

    assert_allclose(sc.world_to_pixel_values(10), 0)
    assert_allclose(sc.world_to_pixel_values([10, 15, 20]), [0, 0.5, 1.0])

    assert sc.world_axis_units == ('Hz',)


def test_invalid_init():

    with pytest.raises(TypeError, match='values should be a Quantity instance'):
        SpectralCoordinates([10, 20, 30])


def test_invalid_conversion():

    sc = SpectralCoordinates([10, 20, 30] * u.Hz)

    with pytest.raises(ValueError, match='SpectralCoordinates is a 1-d coordinate class'):
        sc.pixel_to_world_values(1, 2)

    with pytest.raises(ValueError, match='SpectralCoordinates is a 1-d coordinate class'):
        sc.world_to_pixel_values(1, 2)
