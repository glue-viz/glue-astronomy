import numpy as np

from glue.config import data_translator
from glue.core import Data, Subset

from gwcs import WCS as GWCS
from gwcs.coordinate_frames import CoordinateFrame

from astropy.wcs import WCS
from astropy import units as u
from astropy.wcs import WCSSUB_SPECTRAL
from astropy.nddata import StdDevUncertainty, InverseVariance, VarianceUncertainty
from astropy.wcs.wcsapi import BaseHighLevelWCS
from astropy.modeling import models

from ndcube.wcs.wrappers import CompoundLowLevelWCS

from glue_astronomy.spectral_coordinates import SpectralCoordinates

from specutils import Spectrum1D

UNCERT_REF = {'std': StdDevUncertainty,
              'var': VarianceUncertainty,
              'ivar': InverseVariance}


UCD_TO_SPECTRAL_NAME = {'em.freq': 'Frequency',
                        'em.energy': 'Energy',
                        'em.wavenumber': 'Wavenumber',
                        'em.wl': 'Wavelength',
                        'spect.dopplerVeloc.radio': 'Velocity',
                        'spect.dopplerVeloc.opt': 'Velocity',
                        'spect.dopplerVeloc': 'Velocity',
                        'src.redshift': 'Redshift',
                        'custom:spect.doplerVeloc.beta': 'Beta'}


class PaddedSpectrumWCS(CompoundLowLevelWCS):

    def __init__(self, spectral_wcs, n_extra_axes):
        self.spectral_wcs = spectral_wcs
        frame1 = CoordinateFrame(n_extra_axes, ['SPATIAL']*n_extra_axes,
                                 np.arange(n_extra_axes), unit=[u.pix]*n_extra_axes,
                                 name="Dummy1")
        frame2 = CoordinateFrame(n_extra_axes, ['SPATIAL']*n_extra_axes,
                                 np.arange(n_extra_axes), unit=[u.pix]*n_extra_axes,
                                 name="Dummy2")
        frame2frame = models.Multiply(1)
        if n_extra_axes > 1:
            for i in range(n_extra_axes-1):
                frame2frame = frame2frame & models.Multiply(1)

        pad_wcs = GWCS([(frame1, frame2frame), (frame2, None)])
        super().__init__(pad_wcs, spectral_wcs)


@data_translator(Spectrum1D)
class Specutils1DHandler:

    def to_data(self, obj):

        # Glue expects spectral axis first for cubes (opposite of specutils).
        # Swap the spectral axis to first here. to_object doesn't need this because
        # Spectrum1D does it automatically on initialization.
        if obj.flux.ndim > 1:
            # It's possible to have a 3D Spectrum1D with only a spectral axis defined
            # rather than a full WCS, in which case we need to pad the WCS to match
            # the dimensionality of the flux array.
            if obj.wcs.world_n_dim == obj.flux.ndim:
                data = Data(coords=obj.wcs.swapaxes(-1, 0))
            else:
                n_extra = obj.flux.ndim - obj.wcs.world_n_dim
                data = Data(coords=PaddedSpectrumWCS(obj.wcs, n_extra))
            data['flux'] = np.swapaxes(obj.flux, -1, 0)
            data.get_component('flux').units = str(obj.unit)
        else:
            if obj.flux.ndim == 1 and obj.wcs.world_n_dim == 1 and isinstance(obj.wcs, GWCS):
                data = Data(coords=SpectralCoordinates(obj.spectral_axis))
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
            if len(obj.flux.shape) > 1:
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

        if data.ndim < 2 and statistic is not None:
            statistic = None

        manual_swap = None

        if statistic is None and isinstance(data.coords, PaddedSpectrumWCS):
            kwargs = {'wcs': data.coords.spectral_wcs}
            if data.ndim > 1:
                manual_swap = True
        elif statistic is None and isinstance(data.coords, BaseHighLevelWCS):
            kwargs = {'wcs': data.coords}

        elif statistic is not None:

            if isinstance(data.coords, PaddedSpectrumWCS):
                spec_axis = 0
                axes = tuple(range(1, data.ndim))
                kwargs = {'wcs': data.coords.spectral_wcs}
            elif isinstance(data.coords, WCS):

                # Find spectral axis
                spec_axis = data.coords.naxis - 1 - data.coords.wcs.spec

                # Find non-spectral axes
                axes = tuple(i for i in range(data.ndim) if i != spec_axis)

                kwargs = {'wcs': data.coords.sub([WCSSUB_SPECTRAL])}

            else:
                raise ValueError('Can only use statistic= if the Data object has a FITS WCS')

        elif isinstance(data.coords, SpectralCoordinates):

            kwargs = {'spectral_axis': data.coords.spectral_axis}

        else:
            raise TypeError('data.coords should be an instance of WCS '
                            'or SpectralCoordinates')

        # Copy over metadata
        kwargs['meta'] = data.meta.copy()

        if isinstance(attribute, str):
            attribute = data.id[attribute]
        elif len(data.main_components) == 0:
            raise ValueError('Data object has no attributes.')
        elif attribute is None:
            if len(data.main_components) == 1:
                attribute = data.main_components[0]
            # If no specific attribute is defined, attempt to retrieve
            # the flux and uncertainty, if available
            elif any([x.label in ('flux', 'uncertainty') for x in data.components]):
                attribute = [data.find_component_id(x)
                             for x in ('flux', 'uncertainty')
                             if data.find_component_id(x) is not None]
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
                    if manual_swap:
                        # In this case we need to move the spectral axis back to last
                        values = np.swapaxes(values, -1, 0)

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
