from astropy.table import Table, QTable
from astropy import units as u

from glue.config import data_translator
from glue.core import Data, Component, Subset


@data_translator(Table)
class AstropyTableHandler:

    def to_data(self, table):
        data = Data()

        data.meta = table.meta

        # Loop through columns and make component list
        for column_name in table.columns:
            c = table[column_name]
            u = table[column_name].unit

            if hasattr(c, 'mask'):
                # fill array for now
                try:
                    c = c.filled(fill_value=np.nan)
                except (ValueError, TypeError):  # assigning nan to integer dtype
                    c = c.filled(fill_value=-1)

            data.add_component(Component.autotyped(c, units=u), column_name)

        return data

    def to_object(self, data):

        if isinstance(data, Subset):
            mask = data.to_mask()
            data = data.data
        else:
            mask = None

        table = Table()
        for cid in data.main_components + data.derived_components:

            values = data[cid]
            unit = data.get_component(cid.label).units

            if mask is not None:
                values = values[mask]

            if unit not in (None, ''):
                values *= u.Unit(unit)

            table[cid.label] = values

        return table

@data_translator(QTable)
class AstropyQTableHandler(AstropyTableHandler):

    def to_object(self, data):
        if isinstance(data, Subset):
            mask = data.to_mask()
            data = data.data
        else:
            mask = None

        table = QTable()
        for cid in data.main_components + data.derived_components:

            values = data[cid]
            unit = data.get_component(cid.label).units

            if mask is not None:
                values = values[mask]

            if unit not in (None, ''):
                values *= u.Unit(unit)

            table[cid.label] = values

        return table