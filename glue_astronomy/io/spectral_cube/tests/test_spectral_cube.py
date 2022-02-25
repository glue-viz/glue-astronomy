import numpy as np
from astropy.utils.data import get_pkg_data_filename, get_pkg_data_path
from glue.qglue import parse_data
from spectral_cube import SpectralCube

from glue_astronomy.io.spectral_cube.spectral_cube import is_spectral_cube, read_spectral_cube


def test_identifier_fits():
    assert is_spectral_cube(get_pkg_data_filename('data/cube_3d.fits'))


def test_identifier_casa():
    assert is_spectral_cube(get_pkg_data_path('data/cube_3d.image'))


def test_reader_fits():
    data = read_spectral_cube(get_pkg_data_filename('data/cube_3d.fits'))
    assert isinstance(data['STOKES I'], np.ndarray)
    assert data.shape == (2, 3, 4)


def test_reader_fits_4d():
    data = read_spectral_cube(get_pkg_data_filename('data/cube_4d.fits'))
    assert isinstance(data['STOKES I'], np.ndarray)
    assert data.shape == (2, 3, 4)


def test_reader_fits_4d_fullstokes():
    data = read_spectral_cube(get_pkg_data_filename('data/cube_4d_fullstokes.fits'))
    assert isinstance(data['STOKES I'], np.ndarray)
    assert isinstance(data['STOKES Q'], np.ndarray)
    assert isinstance(data['STOKES U'], np.ndarray)
    assert isinstance(data['STOKES V'], np.ndarray)
    assert data.shape == (2, 3, 4)


def test_reader_casa():
    from dask import array as dask_array
    data = read_spectral_cube(get_pkg_data_path('data/cube_3d.image'))
    assert isinstance(data['STOKES I'], dask_array.Array)
    assert data.shape == (2, 3, 4)


def test_qglue():
    cube = SpectralCube.read(get_pkg_data_filename('data/cube_3d.fits'))
    data = parse_data(cube, 'x')[0]
    assert data.label == 'x'
    assert isinstance(data['flux'], np.ndarray)
    assert data.shape == (2, 3, 4)
