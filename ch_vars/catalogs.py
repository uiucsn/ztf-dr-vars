import dataclasses
import re
from typing import Callable, Optional, Union

import numpy as np

from ch_vars.vsx import VSX_TYPE_MAP


@dataclasses.dataclass(frozen=True)
class PeriodRange:
    min_period: float = 30. / 3600.
    max_period: float = np.inf

    @property
    def range(self):
        return self.min_period, self.max_period


@dataclasses.dataclass(frozen=True, eq=False)
class VarTypeColumn:
    name: str
    converter: Callable = str
    skip_value: Union[str, re.Pattern, None] = None
    map: Optional[Callable] = None

    def __repr__(self):
        return self.name


@dataclasses.dataclass(frozen=True, eq=False)
class TypePlotAll:
    threshold: dict = dataclasses.field(default_factory=lambda: {1: 50, 2: 50, 3: 50})
    as_is: bool = False
    folded: bool = False


@dataclasses.dataclass(frozen=True, eq=False)
class Catalog:
    filename: str
    id_column: str
    var_type_column: VarTypeColumn
    periodic_types: dict = dataclasses.field(default_factory=dict)
    types_plot_all: dict = dataclasses.field(default_factory=dict)


CATALOGS = {
    'sdss-vars': Catalog(
        filename='ztf-sdss-vars.csv.xz',
        id_column='sdss_name',
        var_type_column=VarTypeColumn('bhatti_vartype'),
    ),
    'sdss-candidates': Catalog(
        filename='ztf-sdss.csv.xz',
        id_column='sdss_name',
        var_type_column=VarTypeColumn('segue_tags'),
    ),
    'asassn': Catalog(
        filename='ztf-asassn.csv.xz',
        id_column='asassn_name',
        var_type_column=VarTypeColumn('asassn_type'),
        periodic_types={
            'M': PeriodRange(min_period=50),
            'RRAB': PeriodRange(min_period=0.05, max_period=5),
        }
    ),
    'vsx': Catalog(
        filename='ztf-vsx.csv.xz',
        id_column='vsx_oid',
        var_type_column=VarTypeColumn(
            'type',
            converter=lambda s: s.split('/')[0],
            skip_value=re.compile(r'[:|+]'),
            map=lambda x: VSX_TYPE_MAP[x],
        ),
        periodic_types={
            'Cepheid': PeriodRange(min_period=0.1, max_period=200),
            'Eclipsing': PeriodRange(min_period=0.01, max_period=2000),
            'Ellipsoidal': PeriodRange(min_period=0.01, max_period=1000),
            'Heatbeat': PeriodRange(min_period=0.5),
            'Mira': PeriodRange(min_period=50),
            'R': PeriodRange(max_period=10),
            'RS CVn': PeriodRange(min_period=10),
            'RR Lyr': PeriodRange(min_period=0.05, max_period=5),
            'ZZ Ceti': PeriodRange(max_period=0.1),
            'γ Dor': PeriodRange(min_period=0.1, max_period=10),
            'δ Sct': PeriodRange(max_period=5),
        },
        types_plot_all={
            'Cepheid': TypePlotAll(threshold={1: 50, 2: 50, 3: 20}, folded=True),
            'Ellipsoidal': TypePlotAll(threshold={1: 100, 2: 100, 3: 0}, folded=True),
            'R': TypePlotAll(threshold={1: 100, 2: 100, 3: 0}, folded=True),
            'RR Lyr': TypePlotAll(threshold={1: 50, 2: 50, 3: 50}, folded=True),
            'δ Sct': TypePlotAll(threshold={1: 50, 2: 50, 3: 40}, folded=True),
        },
    ),
}
