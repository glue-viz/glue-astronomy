from astropy.wcs.wcsapi import BaseLowLevelWCS

from glue.config import data_translator
from glue.core import Data, Subset

from astropy import units as u
from astropy.wcs import WCSSUB_STOKES, WCS

from spectral_cube import BooleanArrayMask
from spectral_cube.spectral_cube import (BaseSpectralCube, SpectralCube,
                                         VaryingResolutionSpectralCube)
from spectral_cube.dask_spectral_cube import DaskSpectralCube, DaskVaryingResolutionSpectralCube


@data_translator(BaseSpectralCube)
class SpectralCubeHandler:

    def to_data(self, obj):
        data = Data(coords=obj.wcs)
        data['flux'] = obj.filled_data[...]
        data.get_component('flux').units = str(obj.unit)
        data.meta.update(obj.meta)
        return data

    def to_object(self, data_or_subset, attribute=None, cls=SpectralCube):
        """
        Convert a glue Data object to a SpectralCube object.

        Parameters
        ----------
        data_or_subset : `glue.core.data.Data` or `glue.core.subset.Subset`
            The data to convert to a SpectralCube object
        attribute : `glue.core.component_id.ComponentID`
            The attribute to use for the SpectralCube data
        """

        if data_or_subset.ndim > 0 and data_or_subset.ndim != 3 and data_or_subset.ndim != 4:
            raise ValueError('Data object should have 3 or 4 dimensions in order to '
                             'be converted to a SpectralCube object.')

        if isinstance(data_or_subset, Subset):
            data = data_or_subset.data
            subset_state = data_or_subset.subset_state
        else:
            data = data_or_subset
            subset_state = None

        if isinstance(data.coords, BaseLowLevelWCS):
            wcs = data.coords
        else:
            raise TypeError('data.coords should be an instance of BaseLowLevelWCS.')

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
            mask = BooleanArrayMask(mask, wcs=wcs)

        values = u.Quantity(values, unit=component.units)

        # Drop Stokes axis if there is one for FITS WCS
        if isinstance(wcs, WCS) and wcs.sub([WCSSUB_STOKES]).naxis > 0:
            types = [axistype['coordinate_type'] for axistype in wcs.get_axis_types()]
            # For now we only ever get the first stokes element. We could in
            # principle allow this to be customized, or return a StokesSpectralCube
            slc = tuple([0 if tp == 'stokes' else slice(None) for tp in types])
            subkeep = tuple([index + 1 for index, tp in enumerate(types) if tp != 'stokes'])
            values = values[slc[::-1]]
            wcs = wcs.sub(subkeep)

        return cls(values, mask=mask, wcs=wcs, meta=data.meta)


data_translator(SpectralCube)(SpectralCubeHandler)
data_translator(DaskSpectralCube)(SpectralCubeHandler)


@data_translator(VaryingResolutionSpectralCube)
class VaryingResolutionSpectralCubeHandler(SpectralCubeHandler):

    def to_data(self, obj):
        data = super().to_data(obj)
        data.meta['beams'] = obj.beams
        return data

    def to_object(self, data_or_subset, attribute=None, cls=VaryingResolutionSpectralCube):
        return super().to_object(data_or_subset, attribute=attribute, cls=cls,
                                 beams=data_or_subset.meta['beams'])


data_translator(DaskVaryingResolutionSpectralCube)(VaryingResolutionSpectralCubeHandler)
