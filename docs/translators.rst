Translation between glue and Astropy data objects
=================================================

.. warning:: The functionality described below is not yet available in a released
             version of glue.

Glue v0.16 and later includes functionality that make it easy to work with data
container classes other than the default glue :class:`~glue.core.data.Data`
class, and the **glue-astronomy** plugin adds the ability to use native Astropy
data classes. At this time, the following data classes are supported:

* :class:`~specutils.Spectrum1D` for spectra (from the `specutils
  <https://specutils.readthedocs.io>`_ package)
* :class:`~astropy.nddata.CCDData` for CCD images (from the `astropy
  <https://docs.astropy.org>`_ core package)

* :class:`~spectral_cube.SpectralCube` for spectra (from the `spectral-cube
  <https://spectral-cube.readthedocs.io>`_ package)

* :class:`~astropy.nddata.CCDData` for spectra (from the `astropy.nddata
  <https://docs.astropy.org/en/stable/nddata/>`_ sub-package)

* :class `~specreduce.tracing.Trace` for 2D spectral traces (from the `specreduce
  <https://specreduce.readthedocs.io>`_ package)

Working with these classes is described in `Datasets and subsets`_ below. In
addition, this plugin defines ways for glue selections to be translated to
Astropy `regions <https://astropy-regions.readthedocs.io>`_, as described in
`Selection information`_.

Datasets and subsets
--------------------

We will now take a look at how to use this functionality using
:class:`~specutils.Spectrum1D` as an example, but the workflow is identical for
the other data classes. To start off, we can create a
:class:`~specutils.Spectrum1D` object to use as an example:

.. testsetup::

    >>> from glue.core import DataCollection
    >>> dc = DataCollection()

.. doctest::

    >>> from astropy import units as u
    >>> from specutils import Spectrum1D
    >>> spec = Spectrum1D([0.2, 0.3, 2.2, 0.3] * u.Jy,
    ...                   spectral_axis=[1, 2, 3, 4] * u.micron)

You can add this spectrum to the glue data collection (which we refer to as
``dc`` here) as follows:

    >>> dc['myspectrum'] = spec

The spectrum will be auto-converted to a glue data object. Note that accessing
this object in the data collection will return a glue data object:

    >>> dc['myspectrum']
    Data (label: myspectrum)

To get back a :class:`~specutils.Spectrum1D` object, you can do::

    >>> dc['myspectrum'].get_object()
    <Spectrum1D(flux=<Quantity [0.2, 0.3, 2.2, 0.3] Jy>, spectral_axis=<Quantity [1., 2., 3., 4.] micron>)>

If you are working with glue data objects that were not initially created from
:class:`~specutils.Spectrum1D`, you can still convert them to this class by
explicitly specifying it with the ``cls=`` argument::

    >>> dc['myspectrum'].get_object(cls=Spectrum1D)
    <Spectrum1D(flux=<Quantity [0.2, 0.3, 2.2, 0.3] Jy>, spectral_axis=<Quantity [1., 2., 3., 4.] micron>)>

It is also possible to convert subsets created in glue to e.g.
:class:`~specutils.Spectrum1D` objects. Subsets are usually created by selecting
values in viewers, but for the purposes of this example, we can create a
simple subset programmatically (see LINK for more details on how to do this)::

    >>> dc.new_subset_group(subset_state=dc['myspectrum'].id['flux'] > 1,
    ...                     label='Signal')  # doctest: +IGNORE_OUTPUT

Now that the subset exists, we can extract the subset for the spectrum using::

    >>> spec_subset = dc['myspectrum'].get_subset_object()

The result is a :class:`~specutils.Spectrum1D` object that has the mask set to
indicate values that are part of the subset, and has flux values set to NaN
outside of the subset::

    >>> spec_subset
    <Spectrum1D(flux=<Quantity [nan, nan, 2.2, nan] Jy>, spectral_axis=<Quantity [1., 2., 3., 4.] micron>)>
    >>> spec_subset.mask
    array([False, False,  True, False])

.. TODO: need to make sure the __repr__ for NDData objects includes the mask

Note that the :meth:`~glue.core.data.BaseData.get_subset_object` method is used
to get a data object with the subset of values from a given glue subset - if
instead you are interested in getting a representation of the selection (in
the above case it would be the idea that the selection is 'flux > 1' rather
than the actual values that match that selection), you should take a look
at the `Selection information`_ section.

Selection information
---------------------

As seen in the previous section, we can convert glue data objects and subsets
from/to Astropy data container classes such as :class:`~specutils.Spectrum1D`.
However, in some cases you may want to access the abstract selection information
rather than the actual data values that are in a subset. The Astropy project
includes a package called `regions <https://astropy-regions.readthedocs.io>`_
that provides a way to represent regions of interet, and the **glue-astronomy**
plugin makes it easy to convert selections from glue to Astropy regions.

To illustrate this, we start from a :class:`~astropy.nddata.CCDData` object and
use the infrastructure shown in `Datasets and subsets`_ to add this to a glue
data collection:

.. testsetup::

    >>> from glue.core import DataCollection
    >>> dc = DataCollection()

.. doctest::

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from astropy.nddata import CCDData
    >>> image = CCDData(np.random.random((128, 128)) * u.Jy)
    >>> dc['myimage'] = image

Let's now assume that you define a rectangular selection graphically. We can
also do this programmatically but it is more complicated::

    >>> from glue.core.roi import RectangularROI
    >>> from glue.core.subset import RoiSubsetState
    >>> subset_state = RoiSubsetState(dc['myimage'].pixel_component_ids[1],
    ...                               dc['myimage'].pixel_component_ids[0],
    ...                               RectangularROI(1, 3.5, -0.2, 3.3))
    >>> dc.new_subset_group(subset_state=subset_state, label='Rectangular selection')  # doctest: +IGNORE_OUTPUT

We can then use the :meth:`~glue.core.data.BaseData.get_selection_definition`
method to retrieve the selection as an Astropy
:class:`~regions.RectanglePixelRegion` object::

    >>> dc['myimage'].get_selection_definition(format='astropy-regions')  # doctest: +FLOAT_CMP
    <RectanglePixelRegion(center=PixCoord(x=2.25, y=1.55), width=2.5, height=3.5, angle=0.0 deg)>

If multiple selections/subsets are present, you can specify which one to
retrieve either by index::

    >>> dc['myimage'].get_selection_definition(format='astropy-regions',
    ...                                        subset_id=0)  # doctest: +FLOAT_CMP
    <RectanglePixelRegion(center=PixCoord(x=2.25, y=1.55), width=2.5, height=3.5, angle=0.0 deg)>

or by name::

    >>> dc['myimage'].get_selection_definition(format='astropy-regions',
    ...                                        subset_id='Rectangular selection')  # doctest: +FLOAT_CMP
    <RectanglePixelRegion(center=PixCoord(x=2.25, y=1.55), width=2.5, height=3.5, angle=0.0 deg)>

Note that not all selections in glue can necessarily be represented by Astropy
regions - for example, if we define a subset based on the flux values in the
image::

    >>> dc.new_subset_group(subset_state=dc['myimage'].id['data'] > 0.5,
    ...                     label='Flux-based selection')  # doctest: +IGNORE_OUTPUT

this selection cannot be translated to an Astropy region::

    >>> dc['myimage'].get_selection_definition(format='astropy-regions',
    ...                                        subset_id='Flux-based selection')
    Traceback (most recent call last):
    ...
    NotImplementedError: Subset states of type InequalitySubsetState are not supported
