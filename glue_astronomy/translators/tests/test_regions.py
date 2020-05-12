import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from regions import RectanglePixelRegion, PolygonPixelRegion, CirclePixelRegion,\
                    PointPixelRegion, CompoundPixelRegion, PixCoord

from glue.core import Data, DataCollection
from glue.core.roi import RectangularROI, PolygonalROI, CircularROI,\
                          PointROI, XRangeROI, YRangeROI, AbstractMplRoi

from glue.core.subset import RoiSubsetState, RangeSubsetState, OrState,\
                             AndState, XorState, MultiOrState, MultiRangeSubsetState

from glue.viewers.image.pixel_selection_subset_state import PixelSubsetState


class TestAstropyRegions:

    def setup_method(self, method):
        self.data = Data(flux=np.ones((128, 256)))
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

        xv = [1.3, 2, 3, 1.5, 0.5]
        yv = [10, 20.20, 30, 25, 17.17]

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

        assert_allclose(reg.center.x, 128)
        assert_allclose(reg.center.y, 16.1)
        assert_allclose(reg.width, 256)
        assert_allclose(reg.height, 12.2)

    def test_mpl_roi(self):
        rect_roi = RectangularROI(1, 3.5, -0.2, 3.3)
        subset_state = RoiSubsetState(self.data.pixel_component_ids[1],
                                      self.data.pixel_component_ids[0],
                                      AbstractMplRoi(None, rect_roi))
        self.dc.new_subset_group(subset_state=subset_state, label='mpl_rectangular')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, RectanglePixelRegion)

        assert_allclose(reg.center.x, 2.25)
        assert_allclose(reg.center.y, 1.55)
        assert_allclose(reg.width, 2.5)
        assert_allclose(reg.height, 3.5)

    def test_horiz_range_subset(self):
        subset_state = RangeSubsetState(26, 27.5, self.data.pixel_component_ids[1])

        self.dc.new_subset_group(subset_state=subset_state, label='horizrange')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, RectanglePixelRegion)

        assert_allclose(reg.center.x, 26.75)
        assert_allclose(reg.center.y, 64)
        assert_allclose(reg.width, 1.5)
        assert_allclose(reg.height, 128)

    def test_vert_range_subset(self):
        subset_state = RangeSubsetState(105.5, 107.7, self.data.pixel_component_ids[0])

        self.dc.new_subset_group(subset_state=subset_state, label='vertrange')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, RectanglePixelRegion)

        assert_allclose(reg.center.x, 128)
        assert_allclose(reg.center.y, 106.6)
        assert_allclose(reg.width, 256)
        assert_allclose(reg.height, 2.2)

    def test_invalid_range_subset(self):
        subset_state = RangeSubsetState(0, 1, self.data.main_components[0])

        self.dc.new_subset_group(subset_state=subset_state, label='invalidrange')

        with pytest.raises(ValueError) as exc:
            self.data.get_selection_definition(format='astropy-regions')
        expect_message = 'Range subset state att should be either x or y pixel coordinate'
        assert exc.value.args[0] == expect_message

    def test_horiz_multirange(self):
        subset_state = MultiRangeSubsetState([(26, 27.5), (28, 29)],
                                             self.data.pixel_component_ids[1])

        self.dc.new_subset_group(subset_state=subset_state, label='horizmultirange')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, CompoundPixelRegion)
        assert reg.contains(PixCoord(26.4, 54.6))
        assert reg.contains(PixCoord(28.26, 75.5))
        assert not reg.contains(PixCoord(27.75, 34))

        rect1 = reg.region1
        assert isinstance(rect1, RectanglePixelRegion)
        assert_allclose(rect1.center.x, 26.75)
        assert_allclose(rect1.center.y, 64)
        assert_allclose(rect1.width, 1.5)
        assert_allclose(rect1.height, 128)

        rect2 = reg.region2
        assert isinstance(rect2, RectanglePixelRegion)
        assert_allclose(rect2.center.x, 28.5)
        assert_allclose(rect2.center.y, 64)
        assert_allclose(rect2.width, 1)
        assert_allclose(rect2.height, 128)

    def test_vert_multirange(self):
        subset_state = MultiRangeSubsetState([(30, 33.5), (21, 27)],
                                             self.data.pixel_component_ids[0])

        self.dc.new_subset_group(subset_state=subset_state, label='horizmultirange')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, CompoundPixelRegion)
        assert reg.contains(PixCoord(145, 31.2))
        assert reg.contains(PixCoord(32, 24.6))
        assert not reg.contains(PixCoord(128, 29.2))

        rect1 = reg.region1
        assert isinstance(rect1, RectanglePixelRegion)
        assert_allclose(rect1.center.x, 128)
        assert_allclose(rect1.center.y, 31.75)
        assert_allclose(rect1.width, 256)
        assert_allclose(rect1.height, 3.5)

        rect2 = reg.region2
        assert isinstance(rect2, RectanglePixelRegion)
        assert_allclose(rect2.center.x, 128)
        assert_allclose(rect2.center.y, 24)
        assert_allclose(rect2.width, 256)
        assert_allclose(rect2.height, 6)

    def test_invalid_multiranges(self):
        wrong_att = MultiRangeSubsetState([(30, 33.5), (21, 27)], self.data.main_components[0])
        empty = MultiRangeSubsetState([], self.data.pixel_component_ids[0])
        self.dc.new_subset_group(subset_state=wrong_att, label='wrong_att')
        self.dc.new_subset_group(subset_state=empty, label='empty')

        with pytest.raises(ValueError) as exc:
            self.data.get_selection_definition(subset_id='wrong_att', format='astropy-regions')
        expect_message = 'Multirange subset state att should be either x or y pixel coordinate'
        assert exc.value.args[0] == expect_message

        with pytest.raises(ValueError) as exc:
            self.data.get_selection_definition(subset_id='empty', format='astropy-regions')
        assert exc.value.args[0] == 'Multirange subset state should contain at least one range'

    def test_pixel_subset(self):
        slices = [slice(15, 16, None), slice(130, 131, None)]  # Correspond to pixel (130,15)
        subset_state = PixelSubsetState(self.data, slices)
        self.dc.new_subset_group(subset_state=subset_state, label='pixel_subset')
        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, PointPixelRegion)
        assert_equal(reg.center.x, 130)
        assert_equal(reg.center.y, 15)

    def test_and_region(self):
        subset_state1 = RoiSubsetState(self.data.pixel_component_ids[1],
                                       self.data.pixel_component_ids[0],
                                       RectangularROI(1, 5, 2, 6))
        subset_state2 = RoiSubsetState(self.data.pixel_component_ids[1],
                                       self.data.pixel_component_ids[0],
                                       CircularROI(4.75, 5.75, 0.5))
        and_subset_state = AndState(subset_state1, subset_state2)
        self.dc.new_subset_group(subset_state=and_subset_state, label='andstate')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, CompoundPixelRegion)
        assert isinstance(reg.region1, RectanglePixelRegion)
        assert isinstance(reg.region2, CirclePixelRegion)

        assert reg.contains(PixCoord(4.5, 5.5))
        assert not reg.contains(PixCoord(3, 4))
        assert not reg.contains(PixCoord(5.1, 6.1))
        assert not reg.contains(PixCoord(11, 12))

    def test_or_region(self):
        subset_state1 = RoiSubsetState(self.data.pixel_component_ids[1],
                                       self.data.pixel_component_ids[0],
                                       RectangularROI(1, 5, 2, 6))
        subset_state2 = RoiSubsetState(self.data.pixel_component_ids[1],
                                       self.data.pixel_component_ids[0],
                                       CircularROI(4.75, 5.75, 0.5))
        or_subset_state = OrState(subset_state1, subset_state2)
        self.dc.new_subset_group(subset_state=or_subset_state, label='orstate')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, CompoundPixelRegion)
        assert isinstance(reg.region1, RectanglePixelRegion)
        assert isinstance(reg.region2, CirclePixelRegion)

        assert reg.contains(PixCoord(4.5, 5.5))
        assert reg.contains(PixCoord(3, 4))
        assert reg.contains(PixCoord(5.1, 6.1))
        assert not reg.contains(PixCoord(11, 12))

    def test_xor_region(self):
        subset_state1 = RoiSubsetState(self.data.pixel_component_ids[1],
                                       self.data.pixel_component_ids[0],
                                       RectangularROI(1, 5, 2, 6))
        subset_state2 = RoiSubsetState(self.data.pixel_component_ids[1],
                                       self.data.pixel_component_ids[0],
                                       CircularROI(4.75, 5.75, 0.5))
        xor_subset_state = XorState(subset_state1, subset_state2)
        self.dc.new_subset_group(subset_state=xor_subset_state, label='xorstate')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, CompoundPixelRegion)
        assert isinstance(reg.region1, RectanglePixelRegion)
        assert isinstance(reg.region2, CirclePixelRegion)

        assert not reg.contains(PixCoord(4.5, 5.5))
        assert reg.contains(PixCoord(3, 4))
        assert reg.contains(PixCoord(5.1, 6.1))
        assert not reg.contains(PixCoord(11, 12))

    def test_multior_region(self):
        rects = [(1, 2, 3, 4),
                 (1.5, 2.5, 3.5, 4.5),
                 (2, 3, 4, 5)]
        states = [RoiSubsetState(self.data.pixel_component_ids[1],
                                 self.data.pixel_component_ids[0],
                                 RectangularROI(*rect)) for rect in rects]

        multior_subset_state = MultiOrState(states)
        self.dc.new_subset_group(subset_state=multior_subset_state, label='multiorstate')

        reg = self.data.get_selection_definition(format='astropy-regions')

        assert isinstance(reg, CompoundPixelRegion)
        assert isinstance(reg.region1, CompoundPixelRegion)
        assert isinstance(reg.region2, RectanglePixelRegion)
        assert isinstance(reg.region1.region1, RectanglePixelRegion)
        assert isinstance(reg.region1.region2, RectanglePixelRegion)

        assert reg.contains(PixCoord(1.25, 3.25))
        assert reg.contains(PixCoord(1.75, 3.75))
        assert reg.contains(PixCoord(2.25, 3.75))
        assert reg.contains(PixCoord(2.25, 4.25))
        assert reg.contains(PixCoord(2.75, 4.75))
        assert not reg.contains(PixCoord(5, 7))

    # The following test fails because the logical operations should now work?

    @pytest.mark.xfail
    def test_invalid_combos(self):
        good_subset = RoiSubsetState(self.data.pixel_component_ids[1],
                                     self.data.pixel_component_ids[0],
                                     RectangularROI(1, 5, 2, 6))
        bad_subset = RoiSubsetState(self.data.pixel_component_ids[1],
                                    self.data.main_components[0],
                                    CircularROI(4.75, 5.75, 0.5))
        and_sub = AndState(good_subset, bad_subset)
        or_sub = OrState(good_subset, bad_subset)
        xor_sub = XorState(good_subset, bad_subset)
        multior = MultiOrState([good_subset, bad_subset])
        self.dc.new_subset_group(subset_state=and_sub, label='and')
        self.dc.new_subset_group(subset_state=or_sub, label='or')
        self.dc.new_subset_group(subset_state=xor_sub, label='xor')
        self.dc.new_subset_group(subset_state=multior, label='multior')

        expected_error = 'Subset state yatt should be y pixel coordinate'

        with pytest.raises(ValueError) as exc:
            self.data.get_selection_definition(subset_id='and', format='astropy-regions')
        assert exc.value.args[0] == expected_error

        with pytest.raises(ValueError) as exc:
            self.data.get_selection_definition(subset_id='or', format='astropy-regions')
        assert exc.value.args[0] == expected_error

        with pytest.raises(ValueError) as exc:
            self.data.get_selection_definition(subset_id='xor', format='astropy-regions')
        assert exc.value.args[0] == expected_error

        with pytest.raises(ValueError) as exc:
            self.data.get_selection_definition(subset_id='multior', format='astropy-regions')
        assert exc.value.args[0] == expected_error

    def test_reordered_pixel_components(self):
        self.data._pixel_component_ids = self.data._pixel_component_ids[::-1]
        range_state = RangeSubsetState(105.5, 107.7, self.data.pixel_component_ids[1])

        self.dc.new_subset_group(subset_state=range_state, label='reordered_range')
        rect_state = RoiSubsetState(self.data.pixel_component_ids[0],
                                    self.data.pixel_component_ids[1],
                                    RectangularROI(1, 3.5, -0.2, 3.3))
        self.dc.new_subset_group(subset_state=rect_state, label='reordered_rectangular')

        range_region = self.data.get_selection_definition(subset_id='reordered_range',
                                                          format='astropy-regions')
        rect_region = self.data.get_selection_definition(subset_id='reordered_rectangular',
                                                         format='astropy-regions')

        assert isinstance(range_region, RectanglePixelRegion)

        assert_allclose(range_region.center.x, 128)
        assert_allclose(range_region.center.y, 106.6)
        assert_allclose(range_region.width, 256)
        assert_allclose(range_region.height, 2.2)

        assert isinstance(rect_region, RectanglePixelRegion)

        assert_allclose(rect_region.center.x, 2.25)
        assert_allclose(rect_region.center.y, 1.55)
        assert_allclose(rect_region.width, 2.5)
        assert_allclose(rect_region.height, 3.5)

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
