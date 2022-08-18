0.5.0 (2022-08-18)
------------------

- Updated ``AstropyRegions`` translator to export ``roi.theta`` angle
  (supported as of ``glue`` 1.5.0). [#73]

- Added support to import and export ``specreduce`` ``Trace`` objects. [#72]

0.4.0 (2022-04-07)
------------------

- Updated ``Spectrum1D`` translator to generate dummy WCS when needed for any
  dimensionality, and to preserve specutils axis order when translating
  to Glue ``Data``. [#68]

0.3.3 (2022-03-22)
------------------

- Fixed translation to ``regions.EllipsePixelRegion``. Previous translation
  was passing in radii as full height/width of the ellipse. [#67]

- Fixed compatibility of CCDData translator with GWCS. [#58]

0.3.2 (2021-09-14)
------------------

- Fixed round-tripping of metadata in Spectrum1D. [#48]

0.3.1 (2021-09-09)
------------------

- Fixed coordinate conversion for 2D spectra. [#47]

0.3 (2021-09-07)
----------------

- Improvements to the ``Spectrum1D`` to glue ``Data`` translator, in particular
  for >1-d datasets. [#36, #40, #41, #44, #45]

0.2 (2021-07-05)
----------------

- Add support for converting ``EllipticalROI`` to ``EllipsePixelRegion``. [#32]

0.1 (2020-09-17)
----------------

- Initial release
