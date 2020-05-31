import os

from spectral_cube import SpectralCube, StokesSpectralCube

from glue.core import Data
from glue.config import data_factory, qglue_parser
from glue.core.data_factories.fits import is_fits
from glue.core.coordinates import coordinates_from_wcs

__all__ = ['read_spectral_cube', 'parse_spectral_cube']


def identify_file_format(filename):
    if os.path.isdir(filename):
        if os.path.exists(os.path.join(filename, 'table.f0')):
            return 'casa_image'
        else:
            return None
    else:
        if is_fits(filename):
            return 'fits'
        else:
            return None


def is_spectral_cube(filename, **kwargs):
    """
    Check that the file is a 3D or 4D FITS spectral cube
    """

    file_format = identify_file_format(filename)

    if file_format is None:
        return False

    try:
        StokesSpectralCube.read(filename, format=file_format)
    except Exception:
        return False
    else:
        return True


def spectral_cube_to_data(cube, label=None):

    if isinstance(cube, SpectralCube):
        cube = StokesSpectralCube({'I': cube})

    result = Data(label=label)
    result.coords = coordinates_from_wcs(cube.wcs)

    for component in cube.components:
        data = getattr(cube, component)._data
        result.add_component(data, label='STOKES {0}'.format(component))

    result._preferred_translation = SpectralCube

    return result


@data_factory(label='Spectral Cube', identifier=is_spectral_cube)
def read_spectral_cube(filename, **kwargs):
    """
    Read in a FITS spectral cube. If multiple Stokes components are present,
    these are split into separate glue components.
    """
    cube = StokesSpectralCube.read(filename,
                                   format=identify_file_format(filename))
    return spectral_cube_to_data(cube)


@qglue_parser((SpectralCube, StokesSpectralCube))
def parse_spectral_cube(cube, label):
    return [spectral_cube_to_data(cube, label=label)]
