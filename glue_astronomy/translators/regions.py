from glue.config import subset_state_translator
from glue.core.subset import RoiSubsetState, RangeSubsetState
from glue.core.roi import RectangularROI, PolygonalROI, CircularROI, PointROI, RangeROI

from regions import RectanglePixelRegion, PolygonPixelRegion, CirclePixelRegion, PointPixelRegion, PixCoord


@subset_state_translator('astropy-regions')
class AstropyRegionsHandler:

    def to_object(self, subset):

        data = subset.data
        x_pix_att = data.pixel_component_ids[1]
        y_pix_att  = data.pixel_component_ids[0]

        if data.ndim != 2:
            raise NotImplementedError("Can only handle 2-d datasets at this time")

        subset_state = subset.subset_state
        def range_to_rect(ori, low, high):
            if ori == 'x':
                ymin = 0
                ymax = data.shape[1]
                xmin = low
                xmax = high
            else:
                xmin = 0
                xmax = data.shape[0]
                ymin = low
                ymax = high
            xcen = 0.5 * (xmin + xmax)
            ycen = 0.5 * (ymin + ymax)
            width = xmax - xmin
            height = ymax - ymin
            return RectanglePixelRegion(PixCoord(xcen, ycen), width, height)

        if isinstance(subset_state, RoiSubsetState):

            if subset_state.xatt != x_pix_att:
                raise ValueError('subset state xatt should be x pixel coordinate')

            if subset_state.yatt != y_pix_att:
                raise ValueError('subset state yatt should be y pixel coordinate')

            roi = subset_state.roi
            if isinstance(roi, RectangularROI):
                xcen = 0.5 * (roi.xmin + roi.xmax)
                ycen = 0.5 * (roi.ymin + roi.ymax)
                width = roi.xmax - roi.xmin
                height = roi.ymax - roi.ymin
                return RectanglePixelRegion(PixCoord(xcen, ycen), width, height)
            elif isinstance(roi, PolygonalROI):
                return PolygonPixelRegion(PixCoord(roi.vx, roi.vy))
            elif isinstance(roi, CircularROI):
                return CirclePixelRegion(PixCoord(*roi.get_center()), roi.get_radius())
            elif isinstance(roi, PointROI):
                return PointPixelRegion(PixCoord(*roi.center()))
            elif isinstance(roi, RangeROI):
                return range_to_rect(roi.ori, roi.min, roi.max)

            else:
                raise NotImplementedError("ROIs of type {0} are not yet supported"
                                          .format(roi.__class__.__name__))

        elif isinstance(subset_state, RangeSubsetState):
            if subset_state.att == x_pix_att:
                return range_to_rect('x', subset_state.lo, subset_state.hi)
            elif subset_state.att == y_pix_att:
                return range_to_rect('y', subset_state.lo, subset_state.hi)
            else:
                raise ValueError('range subset state att should be either x or y pixel coordinate')

        else:
            raise NotImplementedError("Subset states of type {0} are not supported"
                                      .format(subset_state.__class__.__name__))
