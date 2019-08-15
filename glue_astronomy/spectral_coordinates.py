import numpy as np

from glue.core.coordinates import Coordinates

__all__ = ['SpectralCoordinates']


class SpectralCoordinates(Coordinates):
    """
    This is a sub-class of Coordinates that is intended for 1-d spectral axes
    given by a :class:`~astropy.units.Quantity` array.
    """

    def __init__(self, values):
        self._index = np.arange(len(values))
        self._values = values

    @property
    def spectral_axis(self):
        """
        Returns
        -------
        """
        return self._values

    def world2pixel(self, *world):
        """
        Parameters
        ----------
        world
        Returns
        -------
        """
        return tuple(np.interp(world, self._values.value, self._index,
                               left=np.nan, right=np.nan))

    def pixel2world(self, *pixel):
        """
        Parameters
        ----------
        pixel
        Returns
        -------
        """
        return tuple(np.interp(pixel, self._index, self._values.value,
                               left=np.nan, right=np.nan))

    def dependent_axes(self, axis):
        """
        Parameters
        ----------
        axis
        Returns
        -------
        """
        return (axis,)