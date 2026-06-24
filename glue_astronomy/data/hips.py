import numpy as np

from glue.core.component_id import ComponentID
from glue.core.data import BaseCartesianData
from glue.utils import compute_statistic, unbroadcast
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

    def _bounding_box(self, subset_state):
        """
        Return the minimal full-resolution bounding box (a list of ``(lo, hi)``
        index pairs) that contains the given subset, or `None` if the subset is
        empty.
        """
        mask = subset_state.to_mask(self)
        unbroadcast_mask = unbroadcast(mask)
        if not np.any(unbroadcast_mask):
            return None
        box = []
        for idim in range(self.ndim):
            collapse_axes = tuple(i for i in range(self.ndim) if i != idim)
            valid = unbroadcast_mask.any(axis=collapse_axes)
            valid = np.broadcast_to(valid, mask.shape[idim:idim + 1])
            indices = np.where(valid)[0]
            box.append((int(indices[0]), int(indices[-1]) + 1))
        return box

    def _select_level(self, box, max_load):
        """
        Pick the finest HiPS level whose bounding box contains at most
        ``max_load`` pixels, and return ``(level, level_box)`` where
        ``level_box`` is the bounding box expressed in that level's pixel
        coordinates, aligned to the level's tile (chunk) boundaries.

        The volume (total number of pixels to load), rather than the size along
        any individual axis, is what we cap here: because the HiPS spatial and
        spectral resolutions are coupled, dropping to a coarser level shrinks
        every axis at once. A small subset (e.g. a single pixel) therefore stays
        at full resolution, while a very large subset falls back to a coarser
        level so that we never load an unreasonable amount of data.

        The box is aligned to tile boundaries because a HiPS tile is the unit of
        I/O - the underlying array can only be read a whole tile at a time.
        """
        order = len(self._dask_arrays) - 1
        for level in range(order, -1, -1):
            array = self._dask_arrays[level]
            shape = array.shape
            chunk = array.chunksize
            level_box = []
            volume = 1
            for axis in range(self.ndim):
                factor = self.shape[axis] / shape[axis]
                step = chunk[axis]
                lo = (int(np.floor(box[axis][0] / factor)) // step) * step
                hi = int(np.ceil(box[axis][1] / factor / step)) * step
                lo = max(0, lo)
                hi = min(shape[axis], hi)
                if hi <= lo:
                    hi = min(shape[axis], lo + step)
                level_box.append((lo, hi))
                volume *= hi - lo
            if volume <= max_load or level == 0:
                return level, level_box
        return 0, level_box

    def _level_mask(self, subset_state, level, level_box):
        """
        Evaluate the subset mask at the resolution of ``level``, aligned with
        the data returned for ``level_box``. The subset is sampled at the
        full-resolution pixel nearest the centre of each coarse cell, which
        avoids ever materialising a full-resolution mask.
        """
        array = self._dask_arrays[level]
        coords = []
        for axis in range(self.ndim):
            factor = self.shape[axis] / array.shape[axis]
            lo, hi = level_box[axis]
            full_index = np.floor((np.arange(lo, hi) + 0.5) * factor).astype(int)
            coords.append(np.clip(full_index, 0, self.shape[axis] - 1))
        grids = np.meshgrid(*coords, indexing='ij')
        return subset_state.to_mask(self, view=tuple(grids))

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
        max_load=40000000,
    ):

        # Global scalar statistics (e.g. the min/max used for colorbar limits)
        # do not depend on the array shape and do not need to be exact, so for
        # speed we compute them from the lowest-resolution level of the HiPS
        # hierarchy.
        if axis is None and subset_state is None:
            data = self._dask_arrays[0].compute()
            return compute_statistic(
                statistic, data, axis=None, percentile=percentile,
                finite=finite, positive=positive,
            )

        if isinstance(axis, tuple):
            collapse = axis
        elif axis is None:
            collapse = None
        else:
            collapse = (axis,)

        # Determine the region of interest and how much data it represents at
        # full resolution, then pick a level whose load is reasonable.
        if subset_state is not None:
            box = self._bounding_box(subset_state)
            if box is None:
                if collapse is None:
                    return np.nan
                shape = [self.shape[i] for i in range(self.ndim) if i not in collapse]
                return np.broadcast_to(np.nan, shape).copy()
        else:
            box = [(0, self.shape[i]) for i in range(self.ndim)]

        level, level_box = self._select_level(box, max_load)
        array = self._dask_arrays[level]

        data = np.asarray(array[tuple(slice(lo, hi) for lo, hi in level_box)])

        if subset_state is not None:
            mask = self._level_mask(subset_state, level, level_box)
        else:
            mask = None

        if collapse is None:
            return compute_statistic(
                statistic, data, mask=mask, axis=None,
                finite=finite, positive=positive, percentile=percentile,
            )

        result = compute_statistic(
            statistic, data, mask=mask, axis=collapse,
            finite=finite, positive=positive, percentile=percentile,
        )

        # The result is at the resolution of `level`, but the profile viewer
        # builds its x axis at full resolution, so we map the result back onto
        # the full-resolution shape along the non-collapsed axes (nearest
        # neighbour - exact when level is full resolution, blocky otherwise).
        remaining = [i for i in range(self.ndim) if i not in collapse]
        full_result = np.broadcast_to(
            np.nan, [self.shape[i] for i in remaining]).copy()
        gather = []
        scatter = []
        for ax in remaining:
            factor = self.shape[ax] / array.shape[ax]
            lo, hi = level_box[ax]
            full_range = np.arange(box[ax][0], box[ax][1])
            level_index = np.clip(np.floor(full_range / factor).astype(int), lo, hi - 1) - lo
            scatter.append(full_range)
            gather.append(level_index)
        full_result[np.ix_(*scatter)] = result[np.ix_(*gather)]
        return full_result

    def compute_histogram(
        self,
        cids,
        weights=None,
        range=None,
        bins=None,
        log=None,
        subset_state=None,
        random_subset=None,
        max_load=40000000,
    ):

        if len(cids) != 1:
            raise NotImplementedError("Only 1D histograms are supported for HiPS data")
        if weights is not None:
            raise NotImplementedError("Weights are not supported for HiPS data histograms")
        if cids[0] is not self.data_cid:
            raise NotImplementedError("Histograms are only supported for the data values")

        # Load the relevant data (and subset mask). Without a subset we use the
        # coarsest level for speed; with a subset we restrict to its bounding
        # box at a resolution chosen so the load stays bounded. As with
        # compute_statistic, the result is only approximate when a coarser level
        # is used, but the histogram shape is preserved.
        if subset_state is None:
            source = self._dask_arrays[0]
            data = source.compute()
            mask = None
        else:
            box = self._bounding_box(subset_state)
            if box is None:
                return np.zeros(bins[0], dtype=float)
            level, level_box = self._select_level(box, max_load)
            source = self._dask_arrays[level]
            data = np.asarray(source[tuple(slice(lo, hi) for lo, hi in level_box)])
            mask = self._level_mask(subset_state, level, level_box)

        if mask is None:
            values = data.ravel()
        else:
            values = data[mask]

        xmin, xmax = sorted(range[0])
        keep = np.isfinite(values) & (values >= xmin) & (values <= xmax)
        values = values[keep]

        if log is not None and log[0]:
            edges = np.logspace(np.log10(xmin), np.log10(xmax), bins[0] + 1)
        else:
            edges = np.linspace(xmin, xmax, bins[0] + 1)

        histogram = np.histogram(values, bins=edges)[0].astype(float)

        # Each loaded cell represents (self.size / source.size) full-resolution
        # pixels, so scale the counts to approximate the full-resolution
        # histogram. This is a no-op when the data was read at full resolution
        # (e.g. for a small subset).
        histogram *= self.size / source.size

        return histogram
