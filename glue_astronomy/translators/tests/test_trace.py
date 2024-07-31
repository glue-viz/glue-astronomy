import numpy as np

from specreduce import tracing

# renamed in specreduce 1.4 and `add_noise` option added (requires photutils)
try:
    from specreduce.utils.synth_data import make_2d_trace_image
    trace_args = dict(add_noise=False)
except ImportError:
    from specreduce.utils.synth_data import make_2dspec_image as make_2d_trace_image
    trace_args = dict()

from glue.core import Data, DataCollection


def test_trace():

    image = make_2d_trace_image(**trace_args)
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
