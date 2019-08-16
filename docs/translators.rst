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


