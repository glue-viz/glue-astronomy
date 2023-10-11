from glue.config import subset_state_translator
from glue.core.subset import (RoiSubsetState, RangeSubsetState, OrState, AndState,
                              XorState, MultiOrState, Subset, MultiRangeSubsetState, InvertState)
from glue.core.roi import (RectangularROI, PolygonalROI, CircularROI, PointROI,
                           RangeROI, AbstractMplRoi, EllipticalROI)
from glue.viewers.image.pixel_selection_subset_state import PixelSubsetState
from glue import __version__ as glue_version

from astropy import units as u
from astropy.wcs.wcsapi import BaseHighLevelWCS
from packaging.version import Version
from regions import (RectanglePixelRegion, PolygonPixelRegion, CirclePixelRegion,
                     PointPixelRegion, PixCoord, EllipsePixelRegion,
                     AnnulusPixelRegion, CircleAnnulusPixelRegion)

__all__ = ["range_to_rect", "roi_subset_state_to_region", "AstropyRegionsHandler"]

GLUE_LT_1_11 = Version(glue_version) < Version('1.11')


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


def roi_subset_state_to_region(subset_state, to_sky=False):
    """Translate the given ``RoiSubsetState`` containing ROI
    that is compatible with 2D spatial regions to proper
    ``regions`` shape.

    If ``to_sky`` is False, it will return the region in pixel coordinates.
    If ``to_sky=True``, it will return the region transformed to sky
    coordinates, per attached data WCS. Alternatively,  ``to_sky`` can be a WCS
    object, which will override any WCS on the input subset state data and the
    region will be returned in pixel coordinates.

    Parameters
    ----------
    subset_state : `~glue.core.subset.SubsetState`
        ROI subset state.
    to_sky: bool or WCS object
        If True, return region in celestial coordinates from attached data WCS.
        Optionally, if a WCS object - a world coordinate system (WCS)
        transformation that supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
        (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`) - is provided, then this will
        override the WCS attached to the subset data.
    """
    roi = subset_state.roi

    if isinstance(roi, (RectangularROI, EllipticalROI)):
        angle = roi.theta * u.radian

    if GLUE_LT_1_11 and isinstance(roi, (CircularROI, EllipticalROI)):
        reg_cen = PixCoord(*roi.get_center())
    elif isinstance(roi, PolygonalROI):
        reg_cen = PixCoord(roi.vx, roi.vy)
    else:
        reg_cen = PixCoord(*roi.center())

    if isinstance(roi, RectangularROI):
        reg = RectanglePixelRegion(reg_cen, roi.width(), roi.height(), angle=angle)
    elif isinstance(roi, PolygonalROI):
        reg = PolygonPixelRegion(reg_cen)
    elif isinstance(roi, CircularROI):
        reg = CirclePixelRegion(reg_cen, roi.radius)
    elif isinstance(roi, EllipticalROI):
        reg = EllipsePixelRegion(reg_cen, roi.radius_x * 2, roi.radius_y * 2, angle=angle)
    elif isinstance(roi, PointROI):
        reg = PointPixelRegion(reg_cen)
    elif not GLUE_LT_1_11:
        from glue.core.roi import CircularAnnulusROI
        if isinstance(roi, CircularAnnulusROI):
            reg = CircleAnnulusPixelRegion(
                center=reg_cen, inner_radius=roi.inner_radius, outer_radius=roi.outer_radius)
        else:
            raise NotImplementedError(f"ROIs of type {roi.__class__.__name__} are not yet supported")  # noqa: E501
    else:
        raise NotImplementedError(f"ROIs of type {roi.__class__.__name__} are not yet supported")

    if to_sky is True:
        if subset_state.xatt.parent.coords is None:
            raise ValueError('Subset parent does not have a WCS.')
        reg = reg.to_sky(subset_state.xatt.parent.coords)
    elif isinstance(to_sky, BaseHighLevelWCS):
        reg = reg.to_sky(to_sky)
    return reg


def _is_annulus(subset_state):
    # There is a new way to make annulus in newer glue.
    if not GLUE_LT_1_11:
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
    if not GLUE_LT_1_11:
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


def _get_xy_pix_att_from_subset(subset):
    data = subset.data
    if data.pixel_component_ids[0].axis == 0:
        x_pix_att = data.pixel_component_ids[1]
        y_pix_att = data.pixel_component_ids[0]
    else:
        x_pix_att = data.pixel_component_ids[0]
        y_pix_att = data.pixel_component_ids[1]
    return x_pix_att, y_pix_att


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
        subset_state = subset.subset_state

        if isinstance(subset_state, RoiSubsetState):
            roi = subset_state.roi

            if isinstance(roi, RangeROI):
                return range_to_rect(subset.data, roi.ori, roi.min, roi.max)

            elif isinstance(roi, AbstractMplRoi):
                temp_sub = Subset(subset.data)
                x_pix_att, y_pix_att = _get_xy_pix_att_from_subset(subset)
                temp_sub.subset_state = RoiSubsetState(x_pix_att, y_pix_att, roi.roi())
                try:
                    return self.to_object(temp_sub)
                except NotImplementedError:
                    raise NotImplementedError(
                        f"ROIs of type {roi.__class__.__name__} are not yet supported")

            else:
                return roi_subset_state_to_region(subset_state)

        elif isinstance(subset_state, RangeSubsetState):
            x_pix_att, y_pix_att = _get_xy_pix_att_from_subset(subset)
            if subset_state.att == x_pix_att:
                return range_to_rect(subset.data, 'x', subset_state.lo, subset_state.hi)
            elif subset_state.att == y_pix_att:
                return range_to_rect(subset.data, 'y', subset_state.lo, subset_state.hi)
            else:
                raise ValueError('Range subset state att should be either x or y pixel coordinate')

        elif isinstance(subset_state, MultiRangeSubsetState):
            x_pix_att, y_pix_att = _get_xy_pix_att_from_subset(subset)
            if subset_state.att == x_pix_att:
                ori = 'x'
            elif subset_state.att == y_pix_att:
                ori = 'y'
            else:
                raise ValueError('Multirange subset state att should be either x or y '
                                 'pixel coordinate')
            if len(subset_state.pairs) == 0:
                raise ValueError('Multirange subset state should contain at least one range')
            region = range_to_rect(
                subset.data, ori, subset_state.pairs[0][0], subset_state.pairs[0][1])
            for pair in subset_state.pairs[1:]:
                region = region | range_to_rect(subset.data, ori, pair[0], pair[1])
            return region

        elif isinstance(subset_state, PixelSubsetState):
            return PointPixelRegion(PixCoord(*subset_state.get_xy(subset.data, 1, 0)))

        elif isinstance(subset_state, AndState):
            if _is_annulus(subset_state):
                return CircleAnnulusPixelRegion(
                    center=PixCoord(x=subset_state.state1.roi.xc, y=subset_state.state1.roi.yc),
                    inner_radius=subset_state.state2.state1.roi.radius,
                    outer_radius=subset_state.state1.roi.radius)
            else:
                data = subset.data
                temp_sub1 = Subset(data=data)
                temp_sub1.subset_state = subset_state.state1
                temp_sub2 = Subset(data=data)
                temp_sub2.subset_state = subset_state.state2
                return self.to_object(temp_sub1) & self.to_object(temp_sub2)

        elif isinstance(subset_state, OrState):
            data = subset.data
            temp_sub1 = Subset(data=data)
            temp_sub1.subset_state = subset_state.state1
            temp_sub2 = Subset(data=data)
            temp_sub2.subset_state = subset_state.state2
            return self.to_object(temp_sub1) | self.to_object(temp_sub2)

        elif isinstance(subset_state, XorState):
            data = subset.data
            temp_sub1 = Subset(data=data)
            temp_sub1.subset_state = subset_state.state1
            temp_sub2 = Subset(data=data)
            temp_sub2.subset_state = subset_state.state2
            return self.to_object(temp_sub1) ^ self.to_object(temp_sub2)

        elif isinstance(subset_state, MultiOrState):
            temp_sub = Subset(data=subset.data)
            temp_sub.subset_state = subset_state.states[0]
            region = self.to_object(temp_sub)
            for state in subset_state.states[1:]:
                temp_sub.subset_state = state
                region = region | self.to_object(temp_sub)
            return region

        else:
            raise NotImplementedError(
                f"Subset states of type {subset_state.__class__.__name__} are not supported")
