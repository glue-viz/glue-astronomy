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
        return tuple(np.interp(world, self._values.value, self._index,
                               left=np.nan, right=np.nan))

    def pixel_to_world_values(self, *pixel):
        """
        Parameters
        ----------
        pixel
        Returns
        -------
        """
        return tuple(np.interp(pixel, self._index, self._values.value,
                               left=np.nan, right=np.nan))
