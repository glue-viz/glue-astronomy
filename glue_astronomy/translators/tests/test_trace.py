import numpy as np

from specreduce import tracing
from specreduce.utils.synth_data import make_2dspec_image

from glue.core import Data, DataCollection


def test_trace():

    image = make_2dspec_image()
    trace = tracing.FlatTrace(image, 5)

    data_collection = DataCollection()

    data_collection['dc_trace'] = trace
    data = data_collection['dc_trace']
    assert isinstance(data, Data)
    assert np.all(data['trace'] == trace.trace)

    trace_from_data = data.get_object()
    assert isinstance(trace_from_data, tracing.FlatTrace)

    # now edit the glue data object, this should now map to an ArrayTrace (instead of a FlatTrace)
    new_trace = np.ones_like(trace.trace)
    data.update_components({data.get_component('trace'): new_trace})
    trace_from_data = data.get_object()
    assert isinstance(trace_from_data, tracing.ArrayTrace)
    assert np.all(trace_from_data.trace == new_trace)
