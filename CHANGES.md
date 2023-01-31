# Full changelog

## v0.6.1 - 2023-01-31

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### Bug Fixes

- Accept `component.units=None` in translators input for new glue-core unit support by @dhomeier in https://github.com/glue-viz/glue-astronomy/pull/84

#### Other Changes

- Require glue-core >= 1.6.1, dropped Python 3.7 support by @pllim in https://github.com/glue-viz/glue-astronomy/pull/83

**Full Changelog**: https://github.com/glue-viz/glue-astronomy/compare/v0.6.0...v0.6.1

## v0.6.0 - 2023-01-20

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### New Features

- Added translators for `NDDataArray` and `StdDevUncertainty` by @bmorris3 in https://github.com/glue-viz/glue-astronomy/pull/81

#### Bug Fixes

- Fixed return type and shape for pixel <-> world conversions in `SpectralCoordinates` by @astrofrog in https://github.com/glue-viz/glue-astronomy/pull/82

### New Contributors

- @bmorris3 made their first contribution in https://github.com/glue-viz/glue-astronomy/pull/81

**Full Changelog**: https://github.com/glue-viz/glue-astronomy/compare/v0.5.1...v0.6.0

## [0.5.1](https://github.com/glue-viz/glue-astronomy/compare/v0.5.0...v0.5.1) - 2022-09-26

### What's Changed

#### New Features

- Added basic support for importing and exporting a wider range of
- `SpectralCube` classes. in https://github.com/glue-viz/glue-astronomy/pull/54

#### Bug Fixes

- Fixed unit parsing for `Specutils1DHandler.to_data` so it no longer
- drops the flux unit in some cases. in https://github.com/glue-viz/glue-astronomy/pull/78

## [0.5.0](https://github.com/glue-viz/glue-astronomy/compare/v0.4.0...v0.5.0) - 2022-08-18

### What's Changed

#### New Features

- Updated `AstropyRegions` translator to export `roi.theta` angle
- (supported as of `glue` 1.5.0). in https://github.com/glue-viz/glue-astronomy/pull/73
- 
- Added support to import and export `specreduce` `Trace` objects. in https://github.com/glue-viz/glue-astronomy/pull/72
- 

## [0.4.0](https://github.com/glue-viz/glue-astronomy/compare/v0.3.3...v0.4.0) - 2022-04-07

### What's Changed

#### New Features

- Updated `Spectrum1D` translator to generate dummy WCS when needed for any
- dimensionality, and to preserve specutils axis order when translating
- to Glue `Data`. in https://github.com/glue-viz/glue-astronomy/pull/68

## [0.3.3](https://github.com/glue-viz/glue-astronomy/compare/v0.3.2...v0.3.3) - 2022-03-22

#### Bug Fixes

- Fixed translation to `regions.EllipsePixelRegion`. Previous translation
- was passing in radii as full height/width of the ellipse. in https://github.com/glue-viz/glue-astronomy/pull/67
- 
- Fixed compatibility of CCDData translator with GWCS. in https://github.com/glue-viz/glue-astronomy/pull/58
- 

## [0.3.2](https://github.com/glue-viz/glue-astronomy/compare/v0.3.1...v0.3.2) - 2021-09-14

#### Bug Fixes

- Fixed round-tripping of metadata in Spectrum1D. in https://github.com/glue-viz/glue-astronomy/pull/48

## [0.3.1](https://github.com/glue-viz/glue-astronomy/compare/v0.3...v0.3.1) - 2021-09-09

#### Bug Fixes

- Fixed coordinate conversion for 2D spectra. in https://github.com/glue-viz/glue-astronomy/pull/47

## [0.3](https://github.com/glue-viz/glue-astronomy/compare/v0.2...v0.3) - 2021-09-07

### What's Changed

#### New Features

- Improvements to the `Spectrum1D` to glue `Data` translator, in particular
- for >1-d datasets. [#36, #40, #41, #44, #45]

## [0.2](https://github.com/glue-viz/glue-astronomy/compare/v0.1...v0.2) - 2021-07-05

### What's Changed

#### New Features

- Add support for converting `EllipticalROI` to `EllipsePixelRegion`. in https://github.com/glue-viz/glue-astronomy/pull/32

## [0.1](https://github.com/glue-viz/glue-astronomy/releases/tag/v0.1) - 2020-09-17

- Initial release
