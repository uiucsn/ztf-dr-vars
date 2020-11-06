import logging
import re
from contextlib import contextmanager

import numpy as np
from clickhouse_driver import Client


NP_TYPE_TO_CH = {
    np.object_: 'String',
    np.str_: 'String',
    np.int64: 'Int64',
    np.float64: 'Float64',
}


CH_COLUMN_ILLEGAL_SYMBOLS = re.compile(r'[^a-z0-9]')


def astropy_column_to_ch(astropy_column):
    ch_type = NP_TYPE_TO_CH[astropy_column.dtype.type]
    if hasattr(astropy_column, 'mask'):
        ch_type = f'Nullable({ch_type})'
    name = astropy_column.name.lower()
    name = CH_COLUMN_ILLEGAL_SYMBOLS.sub('_', name)
    return name, ch_type


def fix_invalid(it, fix_value=None):
    for x in it:
        if np.ma.is_masked(x):
            yield fix_value
        else:
            yield x


CH_INSERT_BLOCK_SIZE = 1 << 12


def create_ch_table(name, data):
    logging.info(f'Creating ClickHouse table')

    db_name, _table_name = name.split('.')
    client = Client.from_url(
        f'clickhouse://default@sai.snad.space/{db_name}?insert_block_size={CH_INSERT_BLOCK_SIZE}',
    )

    columns = dict(map(astropy_column_to_ch, data.itercols()))
    columns_str = ',\n'.join(f'{name} {t}' for name, t in columns.items())

    result = client.execute(f'DROP TABLE IF EXISTS {name}')
    logging.debug(f'Result: {result}')

    result = client.execute(f'''
        CREATE TABLE {name}
        (
            {columns_str},
            h3index10 UInt64 MATERIALIZED geoToH3(radeg, dedeg, 10)
        )
        ENGINE = MergeTree()
        ORDER BY h3index10
    ''')
    logging.debug(f'Result: {result}')

    result = client.execute(
        f'INSERT INTO {name} ({", ".join(columns)}) VALUES',
        (dict(zip(columns, fix_invalid(values))) for values in data.iterrows()),
    )
    logging.debug(f'Result: {result}')


class JoinedVarType:
    def __init__(self, name, *types, description):
        self.name = name
        if not types:
            types = [name]
        self.types = frozenset(types)
        self.description = description

    def __contains__(self, item):
        return item in self.types

    def __repr__(self):
        return f'VarType {self.name} containing subtypes {", ".join(sorted(self.types))}'


def str_to_array(s, dtype=float):
    return np.fromstring(s[1:-1], sep=',', dtype=dtype)


@contextmanager
def numpy_print_options(**kwargs):
    current_options = np.get_printoptions()
    options = current_options.copy()
    options.update(kwargs)
    try:
        yield np.set_printoptions(**options)
    finally:
        np.set_printoptions(**current_options)
