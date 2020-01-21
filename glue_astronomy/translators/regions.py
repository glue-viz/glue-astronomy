from glue.config import subset_state_translator
from glue.core.subset import RoiSubsetState
from glue.core.roi import RectangularROI, PolygonalROI, CircularROI, PointROI

from regions import RectanglePixelRegion, PolygonPixelRegion, CirclePixelRegion, PointPixelRegion, PixCoord


@subset_state_translator('astropy-regions')
class AstropyRegionsHandler:

    def to_object(self, subset):

        data = subset.data

        if data.ndim != 2:
            raise NotImplementedError("Can only handle 2-d datasets at this time")

        subset_state = subset.subset_state

        if isinstance(subset_state, RoiSubsetState):

            if subset_state.xatt != data.pixel_component_ids[1]:
                raise ValueError('subset state xatt should be x pixel coordinate')

            if subset_state.yatt != data.pixel_component_ids[0]:
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
                return CirclePixelRegion(PixCoord(roi.get_center()), roi.get_radius())
            elif isinstance(roi, PointROI):
                return PointPixelRegion(PixCoord(roi.get_center()))
            else:
                raise NotImplementedError("ROIs of type {0} are not yet supported"
                                          .format(roi.__class__.__name__))

        else:
            raise NotImplementedError("Subset states of type {0} are not supported"
                                      .format(subset_state.__class__.__name__))
