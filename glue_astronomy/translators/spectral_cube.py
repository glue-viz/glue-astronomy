import numpy as np

from glue.config import data_translator
from glue.core import Data, Subset
from glue.core.coordinates import WCSCoordinates

from astropy import units as u

from spectral_cube import SpectralCube, BooleanArrayMask


@data_translator(SpectralCube)
class SpectralCubeHandler:

    def to_data(self, obj):
        coords = WCSCoordinates(wcs=obj.wcs)
        data = Data(coords=coords)
        data['flux'] = obj.filled_data[...]
        data.get_component('flux').units = str(obj.unit)
        data.meta.update(obj.meta)
        return data

    def to_object(self, data_or_subset, attribute=None):
        """
        Convert a glue Data object to a SpectralCube object.

        Parameters
        ----------
        data_or_subset : `glue.core.data.Data` or `glue.core.subset.Subset`
            The data to convert to a SpectralCube object
        attribute : `glue.core.component_id.ComponentID`
            The attribute to use for the SpectralCube data
        """

        if data_or_subset.ndim > 0 and data_or_subset.ndim != 3:
            raise ValueError('Data object should have 3 dimensions in order to '
                             'be converted to a SpectralCube object.')

        if isinstance(data_or_subset, Subset):
            data = data_or_subset.data
            subset_state = data_or_subset.subset_state
        else:
            data = data_or_subset
            subset_state = None

        if isinstance(data.coords, WCSCoordinates):
            wcs = data.coords.wcs
        else:
            raise TypeError('data.coords should be an instance of WCSCoordinates.')

        if isinstance(attribute, str):
            attribute = data.id[attribute]
        elif len(data.main_components) == 0:
            raise ValueError('Data object has no attributes.')
        elif attribute is None:
            if len(data.main_components) == 1:
                attribute = data.main_components[0]
            else:
                raise ValueError("Data object has more than one attribute, so "
                                 "you will need to specify which one to use as "
                                 "the flux for the spectral cube using the "
                                 "attribute= keyword argument.")

        component = data.get_component(attribute)

        values = data.get_data(attribute)
        if subset_state is None:
            mask = None
        else:
            mask = data.get_mask(subset_state=subset_state)
            values = values.copy()
            values[~mask] = np.nan
            mask = BooleanArrayMask(mask, wcs=wcs)

        values = values * u.Unit(component.units)

        return SpectralCube(values, mask=mask, wcs=wcs)
