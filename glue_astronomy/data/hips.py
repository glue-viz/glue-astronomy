import numpy as np

from glue.core.component_id import ComponentID
from glue.core.data import BaseCartesianData
from glue.utils import compute_statistic, iterate_chunks
from glue.core.fixed_resolution_buffer import compute_fixed_resolution_buffer


class HiPSData(BaseCartesianData):

    def __init__(self, directory_or_url, *, label):
        from reproject.hips import hips_as_dask_array
        self._array, self._wcs = hips_as_dask_array(directory_or_url)
        self._dask_arrays = []
        # The WCS of each level is kept because the spectral axis is not
        # downsampled by a clean factor between levels, so mapping spectral
        # pixels between levels has to go via the WCS rather than the shape.
        self._level_wcs = []
        # Determine order from array shape
        self._order = int(np.log2(self._array.shape[-1] / 5 / self._array.chunksize[-1]))
        for level in range(self._order):
            arr, wcs = hips_as_dask_array(directory_or_url, level=level)
            self._dask_arrays.append(arr)
            self._level_wcs.append(wcs)
        self._dask_arrays.append(self._array)
        self._level_wcs.append(self._wcs)
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

    def _bounding_box(self, subset_state, max_load):
        """
        Return the minimal full-resolution bounding box (a list of ``(lo, hi)``
        index pairs) that contains the given subset, or `None` if the subset is
        empty.

        Evaluating the mask over the full-resolution array would use a
        prohibitive amount of memory for a large HiPS, so instead we locate the
        subset by evaluating the mask on a coarse grid and zooming in. If the
        subset is smaller than the coarse grid spacing it can be missed by that
        search, so we then try to seed the search from the subset centre (some
        subset states, e.g. ROI selections, expose one), and finally fall back
        to a memory-bounded chunked scan of the full-resolution array.
        """
        # Fast path: a slice-based subset (e.g. the single-pixel selection
        # tool, which creates a PixelSubsetState) encodes the selected region
        # directly as array slices, so we can read the bounding box off without
        # evaluating any mask at all.
        slices = self._slice_for_data(subset_state)
        if slices is not None:
            box = []
            for axis, slc in enumerate(slices):
                if slc.start is None:
                    box.append((0, self.shape[axis]))
                else:
                    lo = max(0, int(slc.start))
                    hi = int(slc.stop) if slc.stop is not None else lo + 1
                    hi = min(self.shape[axis], hi)
                    if hi <= lo:
                        hi = min(self.shape[axis], lo + 1)
                    box.append((lo, hi))
            return box

        search = min(max_load, 4_000_000)

        box = self._search_bounding_box(subset_state, search)
        if box is not None:
            return box

        center = self._subset_center(subset_state)
        if center:
            total = int(np.prod(self.shape))
            stride = max(1, int(np.ceil((total / search) ** (1.0 / self.ndim))))
            region = []
            for axis in range(self.ndim):
                if axis in center:
                    lo = max(0, int(np.floor(center[axis])) - stride)
                    hi = min(self.shape[axis], int(np.ceil(center[axis])) + stride + 1)
                    region.append((lo, hi))
                else:
                    region.append((0, self.shape[axis]))
            box = self._search_bounding_box(subset_state, search, region=region)
            if box is not None:
                return box

        return self._chunked_bounding_box(subset_state, max_load)

    def _search_bounding_box(self, subset_state, max_points, region=None):
        """
        Locate the subset by evaluating its mask on a grid with at most
        ``max_points`` points over ``region`` (the whole array by default) and
        zooming in until the grid is at full resolution. Returns the
        full-resolution bounding box, or `None` if nothing is selected on the
        grid (the subset may be empty, or smaller than the grid spacing).
        """
        if region is None:
            region = [(0, self.shape[axis]) for axis in range(self.ndim)]

        box = None
        for _ in range(64):
            sizes = [hi - lo for lo, hi in region]
            total = int(np.prod(sizes))
            if total <= max_points:
                stride = 1
            else:
                stride = int(np.ceil((total / max_points) ** (1.0 / self.ndim)))
            coords = [np.arange(lo, hi, stride) for lo, hi in region]
            grids = np.meshgrid(*coords, indexing='ij')
            mask = np.asarray(subset_state.to_mask(self, view=tuple(grids)))
            if not mask.any():
                return None
            new_box = []
            for axis in range(self.ndim):
                collapse = tuple(a for a in range(self.ndim) if a != axis)
                selected = np.where(mask.any(axis=collapse))[0]
                # Widen by the sampling stride to cover gaps between samples.
                lo = max(region[axis][0], int(coords[axis][selected[0]]) - (stride - 1))
                hi = min(region[axis][1], int(coords[axis][selected[-1]]) + stride)
                new_box.append((lo, hi))
            if stride == 1 or new_box == box:
                return new_box
            box = new_box
            region = new_box
        return box

    def _subset_center(self, subset_state):
        """
        Return a dict mapping pixel axis index to the subset centre coordinate
        along that axis, for subset states that expose a centre (e.g. ROI
        selections), or `None`.
        """
        center = getattr(subset_state, 'center', None)
        if not callable(center):
            return None
        try:
            values = center()
            atts = subset_state.attributes
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            return None
        if values is None or atts is None:
            return None
        pixel_axes = {cid: cid.axis for cid in self.pixel_component_ids}
        result = {}
        for att, value in zip(atts, np.atleast_1d(values), strict=False):
            if att in pixel_axes and value is not None and np.isfinite(value):
                result[pixel_axes[att]] = float(value)
        return result or None

    def _chunked_bounding_box(self, subset_state, max_points):
        """
        Find the subset bounding box by scanning the full-resolution array in
        chunks of at most ``max_points`` elements, so that the mask is never
        materialised for the whole array at once. Returns `None` if the subset
        is empty.
        """
        lo = list(self.shape)
        hi = [0] * self.ndim
        found = False
        for view in iterate_chunks(self.shape, n_max=max_points):
            mask = np.asarray(subset_state.to_mask(self, view=view))
            if not mask.any():
                continue
            found = True
            for axis in range(self.ndim):
                collapse = tuple(a for a in range(self.ndim) if a != axis)
                selected = np.where(mask.any(axis=collapse))[0]
                start = view[axis].start
                lo[axis] = min(lo[axis], start + int(selected[0]))
                hi[axis] = max(hi[axis], start + int(selected[-1]) + 1)
        if not found:
            return None
        return [(lo[axis], hi[axis]) for axis in range(self.ndim)]

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

    def _slice_for_data(self, subset_state):
        """
        If ``subset_state`` is a slice-based subset (e.g. a PixelSubsetState)
        defined directly in this data's pixel space with contiguous slices,
        return the list of slices, otherwise `None`. Such subsets are
        axis-separable and can be handled without evaluating their (potentially
        full-resolution) mask.
        """
        from glue.core.subset import SliceSubsetState
        if not isinstance(subset_state, SliceSubsetState):
            return None
        if subset_state.reference_data is not self:
            return None
        slices = list(subset_state.slices)
        if len(slices) != self.ndim:
            return None
        if any(slc.step not in (None, 1) for slc in slices):
            return None
        return slices

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

        # Slice-based subsets are axis-separable, so evaluate them directly from
        # the per-axis coordinates. This avoids SliceSubsetState.to_mask, which
        # allocates a full-resolution array when given an array-style view.
        slices = self._slice_for_data(subset_state)
        if slices is not None:
            mask = None
            for axis, slc in enumerate(slices):
                if slc.start is None:
                    continue
                stop = slc.stop if slc.stop is not None else slc.start + 1
                inside = (coords[axis] >= slc.start) & (coords[axis] < stop)
                shape = [1] * self.ndim
                shape[axis] = -1
                axis_mask = inside.reshape(shape)
                mask = axis_mask if mask is None else (mask & axis_mask)
            if mask is None:
                return None
            return np.broadcast_to(mask, tuple(len(c) for c in coords))

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
            box = self._bounding_box(subset_state, max_load)
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
            full_range = np.arange(box[ax][0], box[ax][1])
            scatter.append(full_range)
            gather.append(self._level_indices(ax, full_range, level, level_box))
        full_result[np.ix_(*scatter)] = result[np.ix_(*gather)]
        return full_result

    def _level_indices(self, axis, full_indices, level, level_box):
        """
        Map full-resolution pixel indices along ``axis`` to indices into the
        result computed at ``level`` over ``level_box``. The spatial axes
        downsample by a clean factor, but the spectral axis does not, so spectral
        pixels are mapped via the per-level WCS rather than the shape ratio.
        """
        lo, hi = level_box[axis]
        array = self._dask_arrays[level]
        spatial = (self.ndim - 2, self.ndim - 1)
        if axis not in spatial:
            try:
                world = self._wcs.spectral.pixel_to_world_values(full_indices)
                level_pixel = self._level_wcs[level].spectral.world_to_pixel_values(world)
                index = np.round(level_pixel).astype(int)
            except (AttributeError, ValueError, TypeError, IndexError):
                factor = self.shape[axis] / array.shape[axis]
                index = np.floor(full_indices / factor).astype(int)
        else:
            factor = self.shape[axis] / array.shape[axis]
            index = np.floor(full_indices / factor).astype(int)
        return np.clip(index, lo, hi - 1) - lo

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
            box = self._bounding_box(subset_state, max_load)
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
