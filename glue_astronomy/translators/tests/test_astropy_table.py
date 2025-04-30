import numpy as np
import pytest

from astropy.table import Table, QTable
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from glue.core import Data, DataCollection


@pytest.mark.parametrize('table_class', (Table, QTable))
def test_table(table_class):

    data_collection = DataCollection()

    ra = [9.417, 9.422]
    dec = [-33.702, -33.714]
    flux = [1,1]
    label = ['test1', 'test2']
    columns=['label', 'ra', 'dec', 'flux']
    table = table_class(data=[label, ra, dec, flux], names=columns)

    data_collection['test table'] = table
    data = data_collection['test table']
    assert isinstance(data, Data)

    table_from_data = data.get_object(cls=table_class)
    assert isinstance(table_from_data, table_class)

    for column in columns:
        print(table[column])
        print(table_from_data[column])
        if column == 'label':
            assert np.all(table[column] == table[column])
        else:
            assert_quantity_allclose(table[column].value, table_from_data[column].value)
