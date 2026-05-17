from ndcube.ndcube import NDCube

from glue.config import data_translator
from glue.core import Data, Subset
from glue.core.coordinates import Coordinates

from glue_astronomy.translators.nddata import NDDataArrayHandler


def _get_attribute(attribute, data):
    if isinstance(attribute, str):
        attribute = data.id[attribute]
    elif len(data.main_componenets) == 0:
        raise ValueError('Data object has not attributes')
    elif attribute is None:
        if len(data.main_components) == 1:
            attribute = data.main_components[0]
        else:
            raise ValueError('Data object has more than one attribute, '
                             'you will need to specify which attribute'
                             'needed using `attribute=` kwarg')
    return attribute


def _get_data_and_subset_state(data_or_subset):
    if isinstance(data_or_subset, Subset):
        data= data_or_subset
        subset_state = data_or_subset.subset_state
    else:
        data = data_or_subset
        subset_state = None
    return data, subset_state

@data_translator(NDCube)
class NDCubeHandler:

    def to_data(self, obj):
        """
        Convert an NDCube object to a glue data object.
        """
        data = Data(coords=obj.wcs)


    def to_object(self, data_or_subset, attribute=None):
        """
        Convert a glue Data object to an NDCube object.

        Parameters
        ----------
        data_or_subset : glue.core.subset.Subset
            data to convert to an NDCube object.
        attribute: `glue.core.component_id.ComponentId`
            attribute to convert to an NDCube object.
        """








