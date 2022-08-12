import numpy as np

from glue.config import data_translator
from glue.core import Data, Subset

from specreduce.tracing import Trace, ArrayTrace


@data_translator(Trace)
class TraceHandler:

    def to_data(self, obj):
        """
        Convert a specreduce Trace object to a glue Data object

        Parameters
        ----------
        obj : `specreduce.tracing.Trace`
            The Trace object to convert
        """
        data = Data(trace=obj.trace)
        if hasattr(obj, 'meta'):
            data.meta.update(obj.meta)
        data.meta['Trace'] = obj
        return data

    def to_object(self, data, attribute=None):
        """
        Convert a glue Data object to a Trace object.

        Parameters
        ----------
        data : `glue.core.data.Data`
            The data to convert to a Trace object
        attribute : `glue.core.component_id.ComponentID`
            The attribute to use for the Trace data
        """

        if isinstance(data, Subset):
            raise NotImplementedError("Cannot convert subset to Trace object.")
        if not isinstance(data.meta.get('Trace'), Trace):
            raise TypeError("data is not a valid specreduce Trace object.")

        trace = data.meta['Trace']
        if not np.all(trace.trace[1] == data['trace']):
            trace = ArrayTrace(trace.image, data['trace'])

        return trace
