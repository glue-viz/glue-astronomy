from specreduce import tracing
from specreduce.utils.synth_data import make_2dspec_image

from glue.core import Data, DataCollection


def test_trace():

    image = make_2dspec_image()
    trace = tracing.FlatTrace(image, 5)

    data_collection = DataCollection()

    data_collection['trace'] = trace
    data = data_collection['trace']
    assert isinstance(data, Data)
    assert data['y'] == trace.trace

    trace_from_data = data.get_object()
    assert isinstance(trace_from_data, tracing.FlatTrace)
