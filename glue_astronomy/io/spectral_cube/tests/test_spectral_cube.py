import os
import pytest
from glue.qglue import parse_data

DATA = os.path.join(os.path.dirname(__file__), 'data')


def test_identifier_fits():
    from ..spectral_cube import is_spectral_cube
    assert is_spectral_cube(os.path.join(DATA, 'cube_3d.fits'))


def test_identifier_casa():
    pytest.importorskip('casatools')
    from ..spectral_cube import is_spectral_cube
    assert is_spectral_cube(os.path.join(DATA, 'cube_3d.image'))


def test_reader_fits():
    from ..spectral_cube import read_spectral_cube
    data = read_spectral_cube(os.path.join(DATA, 'cube_3d.fits'))
    data['STOKES I']
    assert data.shape == (2, 3, 4)


def test_reader_fits_4d():
    from ..spectral_cube import read_spectral_cube
    data = read_spectral_cube(os.path.join(DATA, 'cube_4d.fits'))
    data['STOKES I']
    assert data.shape == (2, 3, 4)


def test_reader_fits_4d_fullstokes():
    from ..spectral_cube import read_spectral_cube
    data = read_spectral_cube(os.path.join(DATA, 'cube_4d_fullstokes.fits'))
    data['STOKES I']
    data['STOKES Q']
    data['STOKES U']
    data['STOKES V']
    assert data.shape == (2, 3, 4)


def test_reader_casa():
    pytest.importorskip('casatools')
    from ..spectral_cube import read_spectral_cube
    data = read_spectral_cube(os.path.join(DATA, 'cube_3d.image'))
    data['STOKES I']
    assert data.shape == (2, 3, 4)


def test_qglue():
    from spectral_cube import SpectralCube
    cube = SpectralCube.read(os.path.join(DATA, 'cube_3d.fits'))
    data = parse_data(cube, 'x')[0]
    assert data.label == 'x'
    data['flux']
    assert data.shape == (2, 3, 4)
