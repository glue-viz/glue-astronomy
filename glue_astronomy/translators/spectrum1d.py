import numpy as np

from glue.config import data_translator
from glue.core import Data, Subset

from astropy.wcs import WCS
from astropy import units as u
from astropy.wcs import WCSSUB_SPECTRAL
from astropy.nddata import StdDevUncertainty, InverseVariance, VarianceUncertainty
from astropy.wcs.wcsapi.wrappers.base import BaseWCSWrapper
from astropy.wcs.wcsapi import HighLevelWCSMixin

from gwcs import WCS as GWCS

from glue_astronomy.spectral_coordinates import SpectralCoordinates

from specutils import Spectrum1D

UNCERT_REF = {'std': StdDevUncertainty,
              'var': VarianceUncertainty,
              'ivar': InverseVariance}


class PaddedSpectrumWCS(BaseWCSWrapper, HighLevelWCSMixin):

    # Spectrum1D can use a 1D spectral WCS even for n-dimensional
    # datasets while glue always needs the dimensionality to match,
    # so this class pads the WCS so that it is n-dimensional.

    # NOTE: for now this only handles padding the WCS into 2D WCS. Rather than
    # generalize this we can just remove this class and use CompoundLowLevelWCS
    # from NDCube once it is in a released version.

    def __init__(self, wcs):
        self.spectral_wcs = wcs

    @property
    def pixel_n_dim(self):
        return 2

    @property
    def world_n_dim(self):
        return 2

    @property
    def world_axis_physical_types(self):
        return [None, self.spectral_wcs.world_axis_physical_types[0]]

    @property
    def world_axis_units(self):
        return (None, self.spectral_wcs.world_axis_units[0])

    def pixel_to_world_values(self, *pixel_arrays):
        world_arrays = [pixel_arrays[0],
                        self.spectral_wcs.pixel_to_world_values(pixel_arrays[1])]
        return tuple(world_arrays)

    def world_to_pixel_values(self, *world_arrays):
        pixel_arrays = [world_arrays[0],
                        self.spectral_wcs.world_to_pixel_values(world_arrays[0])]
        return tuple(pixel_arrays)

    @property
    def world_axis_object_components(self):
        return [
            ('spatial', 'value', 'value'),
            self.spectral_wcs.world_axis_object_components[0]
        ]

    @property
    def world_axis_object_classes(self):
        spectral_key = self.spectral_wcs.world_axis_object_components[0][0]
        return {
            'spatial': (u.Quantity, (), {'unit': u.pixel}),
            spectral_key: self.spectral_wcs.world_axis_object_classes[spectral_key]
        }

    @property
    def pixel_shape(self):
        return None

    @property
    def pixel_bounds(self):
        return None

    @property
    def pixel_axis_names(self):
        return tuple(['spatial', self.spectral_wcs.pixel_axis_names[0]])

    @property
    def world_axis_names(self):
        return tuple(['spatial', self.spectral_wcs.world_axis_names[0]])

    @property
    def axis_correlation_matrix(self):
        return np.identity(2).astype('bool')

    @property
    def serialized_classes(self):
        return False


@data_translator(Spectrum1D)
class Specutils1DHandler:

    def to_data(self, obj):

        # Glue expects spectral axis first for cubes (opposite of specutils).
        # Swap the spectral axis to first here. to_object doesn't need this because
        # Spectrum1D does it automatically on initialization.
        if len(obj.flux.shape) == 3:
            data = Data(coords=obj.wcs.swapaxes(-1, 0))
            data['flux'] = np.swapaxes(obj.flux, -1, 0)
            data.get_component('flux').units = str(obj.unit)
        else:
            if obj.flux.ndim == 2 and obj.wcs.world_n_dim == 1:
                data = Data(coords=PaddedSpectrumWCS(obj.wcs))
            else:
                data = Data(coords=obj.wcs)
            data['flux'] = obj.flux
            data.get_component('flux').units = str(obj.unit)

        # Include uncertainties if they exist
        if obj.uncertainty is not None:
            if len(obj.flux.shape) == 3:
                data['uncertainty'] = np.swapaxes(obj.uncertainty.quantity, -1, 0)
            else:
                data['uncertainty'] = obj.uncertainty.quantity
            data.get_component('uncertainty').units = str(obj.uncertainty.unit)
            data.meta.update({'uncertainty_type': obj.uncertainty.uncertainty_type})

        # Include mask if it exists
        if obj.mask is not None:
            if len(obj.flux.shape) == 3:
                data['mask'] = np.swapaxes(obj.mask, -1, 0)
            else:
                data['mask'] = obj.mask

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

        if isinstance(data.coords, WCS):

            # Find spectral axis
            spec_axis = data.coords.naxis - 1 - data.coords.wcs.spec

            # Find non-spectral axes
            axes = tuple(i for i in range(data.ndim) if i != spec_axis)

            if statistic is not None:
                kwargs = {'wcs': data.coords.sub([WCSSUB_SPECTRAL])}
            else:
                kwargs = {'wcs': data.coords}

        elif isinstance(data.coords, PaddedSpectrumWCS):

            kwargs = {'wcs': data.coords.spectral_wcs}

        elif isinstance(data.coords, SpectralCoordinates):

            kwargs = {'spectral_axis': data.coords.spectral_axis}

        else:
            raise TypeError('data.coords should be an instance of WCS '
                            'or SpectralCoordinates')

        if isinstance(attribute, str):
            attribute = data.id[attribute]
        elif len(data.main_components) == 0:
            raise ValueError('Data object has no attributes.')
        elif attribute is None:
            if len(data.main_components) == 1:
                attribute = data.main_components[0]
            # If no specific attribute is defined, attempt to retrieve
            #  both the flux and uncertainties
            elif any([x.label in ('flux', 'uncertainty') for x in data.components]):
                attribute = [data.find_component_id('flux'),
                             data.find_component_id('uncertainty')]
            else:
                raise ValueError("Data object has more than one attribute, so "
                                 "you will need to specify which one to use as "
                                 "the flux for the spectrum using the "
                                 "attribute= keyword argument.")

        def parse_attributes(attributes):
            data_kwargs = {}

            for attribute in attributes:
                component = data.get_component(attribute)

                # Get mask if there is one defined, or if this is a subset
                if subset_state is None:
                    mask = None
                else:
                    mask = data.get_mask(subset_state=subset_state)
                    mask = ~mask

                # Collapse values and mask to profile
                if data.ndim > 1 and statistic is not None:
                    # Get units and attach to value
                    values = data.compute_statistic(statistic, attribute, axis=axes,
                                                    subset_state=subset_state)
                    if mask is not None:
                        collapse_axes = tuple([x for x in range(1, data.ndim)])
                        mask = np.all(mask, collapse_axes)
                else:
                    values = data.get_data(attribute)

                attribute_label = attribute.label

                if attribute_label not in ('flux', 'uncertainty'):
                    attribute_label = 'flux'

                values = values * u.Unit(component.units)

                # If the attribute is uncertainty, we must coerce it to a
                #  specific uncertainty type. If no value exists in the glue
                #  object meta dictionary, use standard deviation.
                if attribute_label == 'uncertainty':
                    values = UNCERT_REF[
                        data.meta.get('uncertainty_type', 'std')](values)

                data_kwargs.update({attribute_label: values,
                                   'mask': mask})

            return data_kwargs

        data_kwargs = parse_attributes(
            [attribute] if not hasattr(attribute, '__len__') else attribute)

        return Spectrum1D(**data_kwargs, **kwargs)
