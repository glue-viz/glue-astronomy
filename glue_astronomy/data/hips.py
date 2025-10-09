import numpy as np

from glue.core.component_id import ComponentID
from glue.core.data import BaseCartesianData
from glue.utils import compute_statistic
from glue.core.fixed_resolution_buffer import compute_fixed_resolution_buffer


class HiPSData(BaseCartesianData):

    def __init__(self, directory_or_url, *, label):
        from reproject.hips import hips_as_dask_array
        self._array, self._wcs = hips_as_dask_array(directory_or_url)
        self._dask_arrays = []
        # Determine order from array shape
        self._order = int(np.log2(self._array.shape[-1] / 5 / self._array.chunksize[-1]))
        for level in range(self._order):
            self._dask_arrays.append(hips_as_dask_array(directory_or_url, level=level)[0])
        self._dask_arrays.append(self._array)
        self.data_cid = ComponentID(label="values", parent=self)
        self._label = label
        self._nan = np.broadcast_to(np.nan, self._array.shape)
        super().__init__()

    @property
    def label(self):
        return self._label

    @property
    def coords(self):
        return self._wcs

    @property
    def shape(self):
        return self._array.shape

    @property
    def main_components(self):
        return [self.data_cid]

    def get_kind(self, cid):
        return "numerical"

    def get_data(self, cid, view=None):
        if cid is self.data_cid:
            if view is None:
                raise NotImplementedError("View must be specified for HiPS data")
            if isinstance(view, tuple):
                if len(view) == self._array.ndim:
                    indices = tuple(v.ravel() for v in view)
                    i, j = indices[-2], indices[-1]
                    # Only keep non-zero pixels for now
                    keep = (i > 0) & (j > 0)
                    if not np.any(keep):
                        return self._nan[view]
                    i = i[keep]
                    j = j[keep]
                    # Determine minimal separation between pixels. Pick any
                    # pixel and use it as a reference pixel, then find the
                    # minimum separation from any other pixel to that one.
                    iref, jref = i[0], j[0]
                    sep = np.hypot(i[1:] - iref, j[1:] - jref)
                    min_sep = np.min(sep[sep > 0])

                    # Now that we have min_sep, we can determine which level
                    # to use. If the minimum separation is larger than e.g.
                    # 2 we can use order - 1, and so on.
                    level = max(0, self._order - int(np.log2(min_sep)))
                    factor = 2 ** int(self._order - level)
                    view = tuple(v // factor for v in view)

                    return self._dask_arrays[level].vindex[view].compute()
                else:
                    raise ValueError(f"View must be a tuple of {self._array.ndim} arrays")
            raise NotImplementedError("View must be specified for HiPS data")
        return super().get_data(cid, view=view)

    def get_mask(self, subset_state, view=None):
        return subset_state.to_mask(self, view=view)

    def compute_fixed_resolution_buffer(self, *args, **kwargs):
        return compute_fixed_resolution_buffer(self, *args, **kwargs)

    def compute_statistic(
        self,
        statistic,
        cid,
        axis=None,
        finite=True,
        positive=False,
        subset_state=None,
        percentile=None,
        random_subset=None,
    ):
        data = self._dask_arrays[0].compute()
        return compute_statistic(
            statistic, data, axis=axis, percentile=percentile, finite=finite
        )

    def compute_histogram(
        self,
        cid,
        range=None,
        bins=None,
        log=False,
        subset_state=None,
        subset_group=None,
    ):

        raise NotImplementedError("Histogram computation not implemented for HiPS data")
