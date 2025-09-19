import numpy as np


from reproject.hips.dask_array import hips3d_as_dask, HiPS3DArray


from glue.core import DataCollection
from glue_qt.app.application import GlueApplication
from glue.core.component_id import ComponentID
from glue.core.data import BaseCartesianData
from glue.utils import compute_statistic
from glue.core.fixed_resolution_buffer import compute_fixed_resolution_buffer


class HiPSData(BaseCartesianData):

    def __init__(self, directory_or_url):
        self._array = HiPS3DArray(directory_or_url)
        self._dask_arrays = []
        for level in range(self._array._order + 1):
            self._dask_arrays.append(hips3d_as_dask(directory_or_url, level=level))
        self.data_cid = ComponentID(label="data", parent=self)
        super(HiPSData, self).__init__()

    @property
    def label(self):
        return "HiPS Data"

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
            else:
                if isinstance(view, tuple):
                    if len(view) == 3:
                        try:
                            i, j, k = view
                            i = i.ravel()
                            j = j.ravel()
                            k = k.ravel()
                            # Only keep non-zero pixels for now
                            keep = (j > 0) & (k > 0)
                            i = i[keep]
                            j = j[keep]
                            k = k[keep]
                            # Determine minimal separation between pixels. Pick any
                            # pixel and use it as a reference pixel, then find the
                            # minimum separation from any other pixel to that one.
                            jref, kref = j[0], k[0]
                            sep = np.hypot(j[1:] - jref, k[1:] - kref)
                            min_sep = np.min(sep[sep > 0])
                            # Now that we have min_sep, we can determine which level
                            # to use. If the minimum separation is larger than e.g.
                            # 2 we can use order - 1, and so on.
                            level = max(0, self._array._order - int(np.log2(min_sep)))
                            factor = 2 ** int(self._array._order - level)
                            inew, jnew, knew = view
                            inew = inew // factor
                            jnew = jnew // factor
                            knew = knew // factor
                            return self._dask_arrays[level].vindex[inew, jnew, knew].compute()
                        except Exception as e:
                            print("Exception in dask compute:", e)
                            import traceback

                            traceback.print_exc()
                            raise
                    else:
                        raise ValueError("View must be a tuple of three arrays")
                raise NotImplementedError("View must be specified for HiPS data")
        else:
            return super(HiPSData, self).get_data(cid, view=view)

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
