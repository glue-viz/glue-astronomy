import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from regions import RectanglePixelRegion, PolygonPixelRegion, CirclePixelRegion, PointPixelRegion

from glue.core import Data, DataCollection
from glue.core.roi import RectangularROI, PolygonalROI, CircularROI, PointROI, XRangeROI, YRangeROI
from glue.core.subset import RoiSubsetState, RangeSubsetState


class TestAstropyRegions:

    def setup_method(self, method):
        self.data = Data(flux=np.ones((128, 128)))
        self.dc = DataCollection([self.data])

    def test_rectangular_roi(self):

        subset_state = RoiSubsetState(self.data.pixel_component_ids[1],
                                      self.data.pixel_component_ids[0],
                                      RectangularROI(1, 3.5, -0.2, 3.3))

        self.dc.new_subset_group(subset_state=subset_state, label='rectangular')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, RectanglePixelRegion)

        assert_allclose(reg.center.x, 2.25)
        assert_allclose(reg.center.y, 1.55)
        assert_allclose(reg.width, 2.5)
        assert_allclose(reg.height, 3.5)

    def test_polygonal_roi(self):

        xv = [1.3,2,3,1.5,0.5]
        yv = [10,20.20,30,25,17.17]

        subset_state = RoiSubsetState(self.data.pixel_component_ids[1],
                                      self.data.pixel_component_ids[0],
                                      PolygonalROI(xv, yv))

        self.dc.new_subset_group(subset_state=subset_state, label='polygon')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, PolygonPixelRegion)

        assert_array_equal(reg.vertices.x, xv)
        assert_array_equal(reg.vertices.y, yv)

    def test_circular_roi(self):

        subset_state = RoiSubsetState(self.data.pixel_component_ids[1],
                                      self.data.pixel_component_ids[0],
                                      CircularROI(1, 3.5, 0.75))

        self.dc.new_subset_group(subset_state=subset_state, label='circular')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, CirclePixelRegion)

        assert_equal(reg.center.x, 1)
        assert_equal(reg.center.y, 3.5)
        assert_equal(reg.radius, 0.75)

    def test_point_roi(self):

        subset_state = RoiSubsetState(self.data.pixel_component_ids[1],
                                      self.data.pixel_component_ids[0],
                                      PointROI(2.64, 5.4))

        self.dc.new_subset_group(subset_state=subset_state, label='point')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, PointPixelRegion)

        assert_equal(reg.center.x, 2.64)
        assert_equal(reg.center.y, 5.4)

    def test_xregion_roi(self):

        subset_state = RoiSubsetState(self.data.pixel_component_ids[1],
                                      self.data.pixel_component_ids[0],
                                      XRangeROI(1, 3.5))

        self.dc.new_subset_group(subset_state=subset_state, label='xrangeroi')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, RectanglePixelRegion)

        assert_allclose(reg.center.x, 2.25)
        assert_allclose(reg.center.y, 64)
        assert_allclose(reg.width, 2.5)
        assert_allclose(reg.height, 128)

    def test_yregion_roi(self):

        subset_state = RoiSubsetState(self.data.pixel_component_ids[1],
                                      self.data.pixel_component_ids[0],
                                      YRangeROI(10, 22.2))

        self.dc.new_subset_group(subset_state=subset_state, label='xrangeroi')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, RectanglePixelRegion)

        assert_allclose(reg.center.x, 64)
        assert_allclose(reg.center.y, 16.1)
        assert_allclose(reg.width, 128)
        assert_allclose(reg.height, 12.2)

    def test_horiz_range_subset(self):
        subset_state = RangeSubsetState(26,27.5,self.data.pixel_component_ids[1])

        self.dc.new_subset_group(subset_state=subset_state, label='horizrange')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, RectanglePixelRegion)

        assert_allclose(reg.center.x, 26.75)
        assert_allclose(reg.center.y, 64)
        assert_allclose(reg.width, 1.5)
        assert_allclose(reg.height, 128)

    def test_vert_range_subset(self):
        subset_state = RangeSubsetState(105.5,107.7,self.data.pixel_component_ids[0])

        self.dc.new_subset_group(subset_state=subset_state, label='vertrange')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, RectanglePixelRegion)

        assert_allclose(reg.center.x, 64)
        assert_allclose(reg.center.y, 106.6)
        assert_allclose(reg.width, 128)
        assert_allclose(reg.height, 2.2)

    def test_invalid_range_subset(self):
        subset_state = RangeSubsetState(0,1, self.data.main_components[0])

        self.dc.new_subset_group(subset_state=subset_state, label='invalidrange')

        with pytest.raises(ValueError) as exc:
            self.data.get_selection_definition(format='astropy-regions')
        assert exc.value.args[0] == 'range subset state att should be either x or y pixel coordinate'


    def test_subset_id(self):

        subset_state = RoiSubsetState(self.data.pixel_component_ids[1],
                                      self.data.pixel_component_ids[0],
                                      RectangularROI(1, 3.5, -0.2, 3.3))

        self.dc.new_subset_group(subset_state=subset_state, label='rectangular')

        for subset_id in [None, 0, 'rectangular']:
            reg = self.data.get_selection_definition(format='astropy-regions',
                                                     subset_id=subset_id)
            assert isinstance(reg, RectanglePixelRegion)
            assert_allclose(reg.center.x, 2.25)
            assert_allclose(reg.center.y, 1.55)
            assert_allclose(reg.width, 2.5)
            assert_allclose(reg.height, 3.5)

        with pytest.raises(ValueError) as exc:
            self.data.get_selection_definition(format='astropy-regions',
                                               subset_id='circular')
        assert exc.value.args[0] == "No subset found with the label 'circular'"

    def test_unsupported(self):
        self.dc.new_subset_group(subset_state=self.data.id['flux'] > 0.5,
                                 label='Flux-based selection')
        with pytest.raises(NotImplementedError) as exc:
            self.data.get_selection_definition(format='astropy-regions',
                                               subset_id='Flux-based selection')
        assert exc.value.args[0] == 'Subset states of type InequalitySubsetState are not supported'
