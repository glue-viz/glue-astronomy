import pytest
import numpy as np
from numpy.testing import assert_allclose

from regions import RectanglePixelRegion

from glue.core import Data, DataCollection
from glue.core.roi import RectangularROI
from glue.core.subset import RoiSubsetState


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