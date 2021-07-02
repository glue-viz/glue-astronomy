from astropy.wcs import WCS
from astropy.nddata import CCDData
from astropy import units as u

from glue.config import data_translator
from glue.core import Data, Subset
from glue.core.coordinates import Coordinates


@data_translator(CCDData)
class CCDDataHandler:

    def to_data(self, obj):
        data = Data(coords=obj.wcs)
        data['data'] = obj.data
        data.get_component('data').units = str(obj.unit)
        data.meta.update(obj.meta)
        return data

    def to_object(self, data_or_subset, attribute=None):
        """
        Convert a glue Data object to a CCDData object.

        Parameters
        ----------
        data_or_subset : `glue.core.data.Data` or `glue.core.subset.Subset`
            The data to convert to a Spectrum1D object
        attribute : `glue.core.component_id.ComponentID`
            The attribute to use for the Spectrum1D data
        """

        if isinstance(data_or_subset, Subset):
            data = data_or_subset.data
            subset_state = data_or_subset.subset_state
        else:
            data = data_or_subset
            subset_state = None

        if isinstance(data.coords, WCS):
            wcs = data.coords
        elif type(data.coords) is Coordinates or data.coords is None:
            wcs = None
        else:
            raise TypeError('data.coords should be an instance of Coordinates or WCS')

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
                                 "the flux for the spectrum using the "
                                 "attribute= keyword argument.")

        component = data.get_component(attribute)

        if data.ndim != 2:
            raise ValueError("Only 2-dimensional datasets can be converted to CCDData")

        values = data.get_data(attribute)

        if subset_state is None:
            mask = None
        else:
            mask = data.get_mask(subset_state=subset_state)
            values = values.copy()
            # Flip mask to match astropy.ndddata formalism
            mask = ~mask

        values = values * u.Unit(component.units)

        return CCDData(values, mask=mask, wcs=wcs, meta=data.meta)
