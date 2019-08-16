import numpy as np

from glue.config import data_translator
from glue.core import Data, Subset
from glue.core.coordinates import WCSCoordinates

from astropy import units as u
from astropy.wcs import WCSSUB_SPECTRAL

from glue_astronomy.spectral_coordinates import SpectralCoordinates

from specutils import Spectrum1D


@data_translator(Spectrum1D)
class Specutils1DHandler:

    def to_data(self, obj):
        coords = SpectralCoordinates(obj.spectral_axis)
        data = Data(coords=coords)
        data['flux'] = obj.flux
        data.get_component('flux').units = str(obj.unit)
        data.meta.update(obj.meta)
        return data

    def to_object(self, data_or_subset, attribute=None, statistic='mean'):
        """
        Convert a glue Data object to a Spectrum1D object.

        Parameters
        ----------
        data_or_subset : `glue.core.data.Data` or `glue.core.subset.Subset`
            The data to convert to a Spectrum1D object
        attribute : `glue.core.component_id.ComponentID`
            The attribute to use for the Spectrum1D data
        statistic : {'minimum', 'maximum', 'mean', 'median', 'sum', 'percentile'}
            The statistic to use to collapse the dataset
        """

        if isinstance(data_or_subset, Subset):
            data = data_or_subset.data
            subset_state = data_or_subset.subset_state
        else:
            data = data_or_subset
            subset_state = None

        if isinstance(data.coords, WCSCoordinates):

            # Find spectral axis
            spec_axis = data.coords.wcs.naxis - 1 - data.coords.wcs.wcs.spec

            # Find non-spectral axes
            axes = tuple(i for i in range(data.ndim) if i != spec_axis)

            kwargs = {'wcs': data.coords.wcs.sub([WCSSUB_SPECTRAL])}

        elif isinstance(data.coords, SpectralCoordinates):

            kwargs = {'spectral_axis': data.coords.spectral_axis}

        else:

            raise TypeError('data.coords should be an instance of WCSCoordinates or SpectralCoordinates')

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

        # Collapse values to profile
        if data.ndim > 1:
            # Get units and attach to value
            values = data.compute_statistic(statistic, attribute, axis=axes,
                                            subset_state=subset_state)
            mask = None
        else:
            values = data.get_data(attribute)
            if subset_state is None:
                mask = None
            else:
                mask = data.get_mask(subset_state=subset_state)
                values = values.copy()
                values[~mask] = np.nan

        values = values * u.Unit(component.units)

        return Spectrum1D(values, mask=mask, **kwargs)
