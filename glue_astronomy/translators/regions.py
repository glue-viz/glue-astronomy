from glue.config import subset_state_translator
from glue.core.subset import (RoiSubsetState, RangeSubsetState, OrState, AndState,
                              XorState, MultiOrState, Subset, MultiRangeSubsetState, InvertState)
from glue.core.roi import (RectangularROI, PolygonalROI, CircularROI, PointROI,
                           RangeROI, AbstractMplRoi, EllipticalROI)
from glue.viewers.image.pixel_selection_subset_state import PixelSubsetState
from glue import __version__ as glue_version

from astropy import units as u
from packaging.version import Version
from regions import (RectanglePixelRegion, PolygonPixelRegion, CirclePixelRegion,
                     PointPixelRegion, PixCoord, EllipsePixelRegion,
                     AnnulusPixelRegion, CircleAnnulusPixelRegion)

__all__ = ["range_to_rect", "AstropyRegionsHandler"]

GLUE_LT_1_10 = Version(glue_version) < Version('1.10')
GLUE_LT_1_10_1 = Version(glue_version) < Version('1.10.1.dev')  # remove .dev after it is released


def range_to_rect(data, ori, low, high):
    """
    Converts a 1D range on a 2D glue Data set into an astropy regions RectangularPixelRegion.

    The region covers the entirety of the data along the axis not specified by the `ori` parameter.

    Parameters
    ----------
    data : `glue.core.data.Data`
        The 2D glue Data object on which the range subset is defined.
    ori: 'x' or 'y'
        Specifies if the range limits are for the x-axis or y-axis respectively.
    low: `float`
        The lower limit of the range.
    high: `float`
        The upper limit of the range.
    """
    if data.ndim != 2:
        raise NotImplementedError("Can only handle 2-d datasets")
    if ori == 'x':
        ymin = 0
        ymax = data.shape[0]
        xmin = low
        xmax = high
    else:
        xmin = 0
        xmax = data.shape[1]
        ymin = low
        ymax = high
    xcen = 0.5 * (xmin + xmax)
    ycen = 0.5 * (ymin + ymax)
    width = xmax - xmin
    height = ymax - ymin
    return RectanglePixelRegion(PixCoord(xcen, ycen), width, height)


def _is_annulus(subset_state):
    # There is a new way to make annulus in newer glue.
    if not GLUE_LT_1_10_1:
        from glue.core.roi import CircularAnnulusROI
        res1 = (isinstance(subset_state, RoiSubsetState) and
                isinstance(subset_state.roi, CircularAnnulusROI))
    else:
        res1 = False

    # subset_state.state1 = outer circle
    # subset_state.state2 = inner circle
    # subset_state.state2 is inverted, so we need its state1
    if not res1:
        res2 = (hasattr(subset_state, 'state1') and
                isinstance(subset_state.state1, RoiSubsetState) and
                isinstance(subset_state.state1.roi, CircularROI) and
                isinstance(subset_state.state2, InvertState) and
                isinstance(subset_state.state2.state1, RoiSubsetState) and
                isinstance(subset_state.state2.state1.roi, CircularROI) and
                (subset_state.state1.roi.xc == subset_state.state2.state1.roi.xc) and
                (subset_state.state1.roi.yc == subset_state.state2.state1.roi.yc) and
                (subset_state.state1.roi.radius > subset_state.state2.state1.roi.radius))
    else:
        res2 = False

    return res1 or res2


# Put this here because there is nowhere else to put it.
# https://github.com/glue-viz/glue/issues/2390
def _annulus_to_subset_state(reg, data):
    """AnnulusPixelRegion to glue subset state."""
    if not isinstance(reg, AnnulusPixelRegion):  # pragma: no cover
        raise ValueError(f"{reg} is not an AnnulusPixelRegion instance")
    # TODO: Add ellipse and rectangle annulus support.
    if not isinstance(reg, CircleAnnulusPixelRegion):  # pragma: no cover
        raise NotImplementedError(f"{reg} not supported")

    xcen = reg.center.x
    ycen = reg.center.y

    # There is a new way to make annulus in newer glue.
    if not GLUE_LT_1_10_1:
        from glue.core.roi import CircularAnnulusROI
        sbst = RoiSubsetState(data.pixel_component_ids[1], data.pixel_component_ids[0],
                              CircularAnnulusROI(xc=xcen, yc=ycen,
                                                 inner_radius=reg.inner_radius,
                                                 outer_radius=reg.outer_radius))
    else:
        state1 = RoiSubsetState(data.pixel_component_ids[1], data.pixel_component_ids[0],
                                CircularROI(xcen, ycen, reg.outer_radius))
        state2 = RoiSubsetState(data.pixel_component_ids[1], data.pixel_component_ids[0],
                                CircularROI(xcen, ycen, reg.inner_radius))
        sbst = AndState(state1, ~state2)

    return sbst


@subset_state_translator('astropy-regions')
class AstropyRegionsHandler:

    def to_object(self, subset):
        """
        Convert a glue Subset object to a astropy regions Region object.

        Parameters
        ----------
        subset : `glue.core.subset.Subset`
            The subset to convert to a Region object
        """
        data = subset.data

        if data.pixel_component_ids[0].axis == 0:
            x_pix_att = data.pixel_component_ids[1]
            y_pix_att = data.pixel_component_ids[0]
        else:
            x_pix_att = data.pixel_component_ids[0]
            y_pix_att = data.pixel_component_ids[1]

        subset_state = subset.subset_state

        if isinstance(subset_state, RoiSubsetState):

            roi = subset_state.roi
            angle = getattr(roi, 'theta', 0) * u.radian
            if isinstance(roi, RectangularROI):
                xcen = 0.5 * (roi.xmin + roi.xmax)
                ycen = 0.5 * (roi.ymin + roi.ymax)
                width = roi.xmax - roi.xmin
                height = roi.ymax - roi.ymin
                return RectanglePixelRegion(PixCoord(xcen, ycen), width, height, angle=angle)
            elif isinstance(roi, PolygonalROI):
                return PolygonPixelRegion(PixCoord(roi.vx, roi.vy))
            elif isinstance(roi, CircularROI):
                xcen, ycen = roi.get_center() if GLUE_LT_1_10 else roi.center()
                return CirclePixelRegion(PixCoord(xcen, ycen), roi.get_radius())
            elif isinstance(roi, EllipticalROI):
                return EllipsePixelRegion(
                    PixCoord(roi.xc, roi.yc), roi.radius_x * 2, roi.radius_y * 2, angle=angle)
            elif isinstance(roi, PointROI):
                return PointPixelRegion(PixCoord(*roi.center()))
            elif isinstance(roi, RangeROI):
                return range_to_rect(data, roi.ori, roi.min, roi.max)

            elif isinstance(roi, AbstractMplRoi):
                temp_sub = Subset(data)
                temp_sub.subset_state = RoiSubsetState(x_pix_att, y_pix_att, roi.roi())
                try:
                    return self.to_object(temp_sub)
                except NotImplementedError:
                    raise NotImplementedError("ROIs of type {0} are not yet supported"
                                              .format(roi.__class__.__name__))

            # There is a new way to make annulus in newer glue.
            elif not GLUE_LT_1_10_1:
                from glue.core.roi import CircularAnnulusROI
                if isinstance(roi, CircularAnnulusROI):
                    return CircleAnnulusPixelRegion(
                        center=PixCoord(x=roi.xc, y=roi.yc),
                        inner_radius=roi.inner_radius,
                        outer_radius=roi.outer_radius)
                else:
                    raise NotImplementedError("ROIs of type {0} are not yet supported"
                                              .format(roi.__class__.__name__))

            else:
                raise NotImplementedError("ROIs of type {0} are not yet supported"
                                          .format(roi.__class__.__name__))

        elif isinstance(subset_state, RangeSubsetState):
            if subset_state.att == x_pix_att:
                return range_to_rect(data, 'x', subset_state.lo, subset_state.hi)
            elif subset_state.att == y_pix_att:
                return range_to_rect(data, 'y', subset_state.lo, subset_state.hi)
            else:
                raise ValueError('Range subset state att should be either x or y pixel coordinate')

        elif isinstance(subset_state, MultiRangeSubsetState):
            if subset_state.att == x_pix_att:
                ori = 'x'
            elif subset_state.att == y_pix_att:
                ori = 'y'
            else:
                message = 'Multirange subset state att should be either x or y pixel coordinate'
                raise ValueError(message)
            if len(subset_state.pairs) == 0:
                message = 'Multirange subset state should contain at least one range'
                raise ValueError(message)
            region = range_to_rect(data, ori, subset_state.pairs[0][0], subset_state.pairs[0][1])
            for pair in subset_state.pairs[1:]:
                region = region | range_to_rect(data, ori, pair[0], pair[1])
            return region

        elif isinstance(subset_state, PixelSubsetState):
            return PointPixelRegion(PixCoord(*subset_state.get_xy(data, 1, 0)))

        elif isinstance(subset_state, AndState):
            if _is_annulus(subset_state):
                return CircleAnnulusPixelRegion(
                    center=PixCoord(x=subset_state.state1.roi.xc, y=subset_state.state1.roi.yc),
                    inner_radius=subset_state.state2.state1.roi.radius,
                    outer_radius=subset_state.state1.roi.radius)
            else:
                temp_sub1 = Subset(data=data)
                temp_sub1.subset_state = subset_state.state1
                temp_sub2 = Subset(data=data)
                temp_sub2.subset_state = subset_state.state2
                return self.to_object(temp_sub1) & self.to_object(temp_sub2)

        elif isinstance(subset_state, OrState):
            temp_sub1 = Subset(data=data)
            temp_sub1.subset_state = subset_state.state1
            temp_sub2 = Subset(data=data)
            temp_sub2.subset_state = subset_state.state2
            return self.to_object(temp_sub1) | self.to_object(temp_sub2)

        elif isinstance(subset_state, XorState):
            temp_sub1 = Subset(data=data)
            temp_sub1.subset_state = subset_state.state1
            temp_sub2 = Subset(data=data)
            temp_sub2.subset_state = subset_state.state2
            return self.to_object(temp_sub1) ^ self.to_object(temp_sub2)

        elif isinstance(subset_state, MultiOrState):
            temp_sub = Subset(data=data)
            temp_sub.subset_state = subset_state.states[0]
            region = self.to_object(temp_sub)
            for state in subset_state.states[1:]:
                temp_sub.subset_state = state
                region = region | self.to_object(temp_sub)
            return region

        else:
            raise NotImplementedError("Subset states of type {0} are not supported"
                                      .format(subset_state.__class__.__name__))
