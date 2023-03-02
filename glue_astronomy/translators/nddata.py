from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseHighLevelWCS
from astropy.nddata import CCDData, NDData, NDDataArray
from astropy.nddata.nduncertainty import StdDevUncertainty
from astropy import units as u

from glue.config import data_translator
from glue.core import Data, Subset
from glue.core.coordinates import Coordinates

from .spectrum1d import SpectralCoordinates, UNCERT_REF


def _get_attribute(attribute, data):
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
    return attribute


def _get_value_and_mask(subset_state, data, values):
    if subset_state is None:
        mask = None
    else:
        mask = data.get_mask(subset_state=subset_state)
        values = values.copy()
        # Flip mask to match astropy.ndddata formalism
        mask = ~mask
    return values, mask


def _get_data_and_subset_state(data_or_subset):
    if isinstance(data_or_subset, Subset):
        data = data_or_subset.data
        subset_state = data_or_subset.subset_state
    else:
        data = data_or_subset
        subset_state = None
    return data, subset_state


@data_translator(NDDataArray)
class NDDataArrayHandler:

    def to_data(self, obj):
        data = Data(coords=obj.wcs)
        data['data'] = obj.data
        data.get_component('data').units = str(obj.unit)
        if obj.uncertainty is not None:
            uncert = obj.uncertainty.represent_as(StdDevUncertainty)
            data['uncertainty'] = uncert.array
        data.meta.update(obj.meta)
        return data

    def to_object(self, data_or_subset, attribute=None):
        """
        Convert a glue Data object to a NDDataArray object.

        Parameters
        ----------
        data_or_subset : `glue.core.data.Data` or `glue.core.subset.Subset`
            The data to convert to a NDDataArray object
        attribute : `glue.core.component_id.ComponentID`
            The attribute to use for the NDDataArray data
        """

        data, subset_state = _get_data_and_subset_state(data_or_subset)

        if isinstance(data.coords, (WCS, BaseHighLevelWCS, SpectralCoordinates)):
            wcs = data.coords
        elif type(data.coords) is Coordinates or data.coords is None:
            wcs = None
        else:
            raise TypeError('data.coords should be an instance of Coordinates or WCS')

        component_labels = [d.label for d in data.component_ids()]
        if attribute is None:
            for desired_label in ('data', 'flux'):
                if desired_label in component_labels:
                    attribute = desired_label
                    break

        attribute = _get_attribute(attribute, data)
        component = data.get_component(attribute)
        values = data.get_data(attribute)
        values, mask = _get_value_and_mask(subset_state, data, values)

        if 'uncertainty' in component_labels:
            uncert_cls = UNCERT_REF[
                data.meta.get('uncertainty_type', 'std')
            ]
            uncertainty = uncert_cls(
                data.get_component('uncertainty').data
            ).represent_as(StdDevUncertainty)
        else:
            uncertainty = None

        result = NDDataArray(
            values,
            unit=component.units,
            mask=mask,
            wcs=wcs,
            meta=data.meta,
            uncertainty=uncertainty
        )

        return result


@data_translator(CCDData)
class CCDDataHandler(NDDataArrayHandler):

    def to_object(self, data_or_subset, attribute=None):
        """
        Convert a glue Data object to a CCDData object.

        Parameters
        ----------
        data_or_subset : `glue.core.data.Data` or `glue.core.subset.Subset`
            The data to convert to a CCDData object
        attribute : `glue.core.component_id.ComponentID`
            The attribute to use for the CCDData data
        """

        data, subset_state = _get_data_and_subset_state(data_or_subset)

        if isinstance(data.coords, WCS):
            has_fitswcs = True
            wcs = data.coords
        elif isinstance(data.coords, BaseHighLevelWCS):
            has_fitswcs = False
            wcs = data.coords
        elif type(data.coords) is Coordinates or data.coords is None:
            has_fitswcs = True  # For backward compatibility
            wcs = None
        else:
            raise TypeError('data.coords should be an instance of Coordinates or WCS')

        attribute = _get_attribute(attribute, data)
        component = data.get_component(attribute)

        if data.ndim != 2:
            raise ValueError("Only 2-dimensional datasets can be converted to CCDData")

        values = data.get_data(attribute)
        values, mask = _get_value_and_mask(subset_state, data, values)
        values = u.Quantity(values, unit=component.units)

        if has_fitswcs:
            result = CCDData(values, mask=mask, wcs=wcs, meta=data.meta)
        else:
            # https://github.com/astropy/astropy/issues/11727
            result = NDData(values, mask=mask, wcs=wcs, meta=data.meta)

        return result


@data_translator(StdDevUncertainty)
class StdDevUncertaintyHandler:

    def to_data(self, obj):
        data = Data()
        data['data'] = obj.array
        data.get_component('data').units = str(obj.unit)
        return data

    def to_object(self, data_or_subset, attribute=None):
        """
        Convert a glue Data object to a StdDevUncertainty object.

        Parameters
        ----------
        data_or_subset : `glue.core.data.Data` or `glue.core.subset.Subset`
            The data to convert to a StdDevUncertainty object
        attribute : `glue.core.component_id.ComponentID`
            The attribute to use for the StdDevUncertainty data
        """

        if isinstance(data_or_subset, Subset):
            data = data_or_subset.data
        else:
            data = data_or_subset

        attribute = _get_attribute(attribute, data)
        component = data.get_component(attribute)
        values = data.get_data(attribute)

        result = StdDevUncertainty(values, unit=component.units)

        return result
