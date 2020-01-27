Astronomy-specific data loaders
===============================

The **glue-astronomy** plugin packages provides some astronomy-specific
data loaders, which are described below.

Spectral cubes
--------------

By default, glue can load FITS files, including spectral cubes, but it
does not treat spectral cube any differently to other datasets. The
**glue-astronomy** plugin adds the option of loading in spectral cube
files using the `spectral-cube <https://spectral-cube.readthedocs.io/en/latest/>`_
package, which does several things differently to the default FITS loader:

* It re-orders the axes to always be as if the cube had been in the
  longitude, latitude, spectral order to start with.

* It identifies and corrects certain non-standard WCS headers to use the
  recommended keywords.

* It splits cubes with Stokes components into several cubes, one for
  each component. This means that in glue, a cube with a Stokes axis
  would end up as a 3-d dataset with multiple attributes, as opposed to
  a 4-d dataset with a single attribute.

In addition, the loader provided here is able to read in spectral cubes in
other formats supported by spectral-cube, such as for example the CASA
``.image`` format.

To make use of this data loader, click on **Import Data** in glue and
select **Spectral Cube** for the type in the window that pops up, then
select the file (or directory for ``.image`` datasets) you want to load.
