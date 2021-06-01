import importlib

import pandas as pd

from ch_vars.data import hyperleda


def get_hyperleda():
    column_names = ['objname', 'j2000', 'modulus']
    with importlib.resources.open_binary(hyperleda, 'HyperLeda_a007_1620838730.txt') as fh:
        df = pd.read_csv(
            fh,
            skiprows=17,
            comment='#',
            delimiter=r'\s+',
            names=column_names,
            usecols=column_names,
            index_col='objname',
        )
    grouped_df = pd.DataFrame(
        [{'objname': objname, 'j2000': df['j2000'].iloc[0], 'modulus': df['modulus'].median()}
         for objname, df in df.groupby(by='objname', axis=0, as_index=True)],
    )
    grouped_df = grouped_df.set_index('objname')
    return grouped_df
