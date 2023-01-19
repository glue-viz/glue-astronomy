import numpy as np

from astropy.units import Quantity
from glue.core.coordinates import Coordinates

__all__ = ['SpectralCoordinates']


class SpectralCoordinates(Coordinates):
    """
    This is a sub-class of Coordinates that is intended for 1-d spectral axes
    given by a :class:`~astropy.units.Quantity` array.
    """

    def __init__(self, values):
        if not isinstance(values, Quantity):
            raise TypeError('values should be a Quantity instance')
        self._index = np.arange(len(values))
        self._values = values
        super().__init__(n_dim=1)

    @property
    def spectral_axis(self):
        """
        Returns
        -------
        """
        return self._values

    def world_to_pixel_values(self, *world):
        """
        Parameters
        ----------
        world
        Returns
        -------
        """
        if len(world) > 1:
            raise ValueError('SpectralCoordinates is a 1-d coordinate class '
                             'and only accepts a single scalar or array to convert')
        return np.interp(world[0], self._values.value, self._index,
                         left=np.nan, right=np.nan)

    def pixel_to_world_values(self, *pixel):
        """
        Parameters
        ----------
        pixel
        Returns
        -------
        """
        if len(pixel) > 1:
            raise ValueError('SpectralCoordinates is a 1-d coordinate class '
                             'and only accepts a single scalar or array to convert')
        return np.interp(pixel[0], self._index, self._values.value,
                         left=np.nan, right=np.nan)
