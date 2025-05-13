# Full changelog

## v0.11.0 - 2025-05-13

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### Bug Fixes

* Loosen check on coordinate class for NDData translator to accept subclasses by @bmorris3 in https://github.com/glue-viz/glue-astronomy/pull/101

#### Documentation

* Document and check WCS usage for `to_sky` option in subset-to-region translator; add tests by @cshanahan1 in https://github.com/glue-viz/glue-astronomy/pull/97

#### Other Changes

* Don't set PLAT in GitHub Actions config by @astrofrog in https://github.com/glue-viz/glue-astronomy/pull/70
* Fix compatibility with future glue-core changes by @astrofrog in https://github.com/glue-viz/glue-astronomy/pull/95
* Update CI matrix and fix issues by @astrofrog in https://github.com/glue-viz/glue-astronomy/pull/104

### New Contributors

* @cshanahan1 made their first contribution in https://github.com/glue-viz/glue-astronomy/pull/97

**Full Changelog**: https://github.com/glue-viz/glue-astronomy/compare/v0.10.0...v0.11.0

## v0.10.0 - 2023-06-16

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### New Features

- Support sky regions and allow direct call to translator function by @pllim in https://github.com/glue-viz/glue-astronomy/pull/93

**Full Changelog**: https://github.com/glue-viz/glue-astronomy/compare/v0.9.0...v0.10.0

## v0.9.0 - 2023-06-01

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### New Features

- Support new CircularAnnulusROI by @pllim in https://github.com/glue-viz/glue-astronomy/pull/92

**Full Changelog**: https://github.com/glue-viz/glue-astronomy/compare/v0.8.0...v0.9.0

## v0.8.0 - 2023-05-11

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### New Features

- Added translator for `CircleAnnulusPixelRegion` by @pllim in https://github.com/glue-viz/glue-astronomy/pull/90

#### Bug Fixes

- Use `CircularROI.center` to avoid deprecation warnings with glue_core >= 1.10 by @dhomeier in https://github.com/glue-viz/glue-astronomy/pull/91

**Full Changelog**: https://github.com/glue-viz/glue-astronomy/compare/v0.7.0...v0.8.0

## v0.7.0 - 2023-03-02

<!-- Release notes generated using configuration in .github/release.yml at main -->
### What's Changed

#### New Features

- Adding support for uncertainty extraction to NDDataArray by @bmorris3 in https://github.com/glue-viz/glue-astronomy/pull/86

#### Bug Fixes

- Fix world_axis_units for SpectralCoordinates by @astrofrog in https://github.com/glue-viz/glue-astronomy/pull/87

**Full Changelog**: https://github.com/glue-viz/glue-astronomy/compare/v0.6.1...v0.7.0

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
