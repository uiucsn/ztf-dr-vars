import logging
import os
import re
import shutil
from contextlib import closing
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from astropy.io import ascii

from ch_vars.common import *


def download_from_ftp(url, directory):
    filename = os.path.basename(url)
    # CDS read driver requires filename to be consistent with ReadMe
    if filename.endswith('.gz'):
        filename, _ext = os.path.splitext(filename)
    path = os.path.join(directory, filename)
    logging.info(f'Downloading {url} to {path}')
    with closing(urlopen(url)) as r, open(path, 'wb') as fh:
        shutil.copyfileobj(r, fh)
    return path


def vsx_dat(directory):
    url = 'ftp://cdsarc.u-strasbg.fr/pub/cats/B/vsx/vsx.dat.gz'
    return download_from_ftp(url, directory)


def readme(directory):
    url = 'ftp://cdsarc.u-strasbg.fr/pub/cats/B/vsx/ReadMe'
    return download_from_ftp(url, directory)


def get_data():
    with TemporaryDirectory() as directory:
        logging.info('Reading VSX table')
        table = ascii.read(vsx_dat(directory), readme=readme(directory), format='cds')
    return table


def main():
    logging.basicConfig(level=logging.DEBUG)

    table = get_data()
    create_ch_table('vsx.vsx', table)


VSX_TYPES = [
    JoinedVarType(
        'Eclipsing',
        'E', 'EA', 'EB', 'EP', 'EW', 'EC', 'ED', 'ESD',
        description='''Eclipsing variables.'''
    ),
    JoinedVarType(
        'Other rotating',
        'ACV', 'FKCOM', 'LERI', 'PSR', 'SXARI', 'NSIN ELL',
        description='Other rotating variables',
    ),
    JoinedVarType(
        'BY Dra',
        'BY',
        description='BY Draconis-type variables, which are emission-line dwarfs of dKe-dMe spectral type showing quasi-periodic light changes with periods from a fraction of a day to 120 days and amplitudes from several hundredths to 0.5 mag. in V. The light variability is caused by axial rotation of a star with a variable degree of non-uniformity of the surface brightness (spots) and chromospheric activity. Some of these stars also show flares similar to those of UV Ceti stars, and in those cases they also belong to the latter type and are simultaneously considered eruptive variables.',
    ),
    JoinedVarType(
        'Ellipsoidal',
        'ELL',
        description='Rotating ellipsoidal variables. These are close binary systems with ellipsoidal components, which change combined brightnesses with periods equal to those of orbital motion because of changes in emitting areas toward an observer. Eclipsing binaries may also show ellipsoidal variability but the ELL objects listed in VSX are those showing no eclipses. Light amplitudes usually do not exceed 0.1 mag. in V but may reach 0.3 mag. in some cases. Examples: b Per'
    ),
    JoinedVarType(
        'Heartbeat',
        'HB',
        description='Heartbeat stars. A type of eccentric binary stars (e >0.2) whose light curves resemble a cardiogram. They are ellipsoidal variables that undergo extreme dynamic tidal forces. As the two stars pass through periastron, brightness variations occur as a consequence of tidal deformation and mutual irradiation. There may also be tidally induced pulsations present. The morphology of the photometric periastron variation (heartbeat) depends strongly on the eccentricity, inclination and argument of periastron. The amplitude of variations is very small, usually below 0.01 mag. but it may exceed 0.3 mag. in extreme cases.'
    ),
    JoinedVarType(
        'R',
        description='Close binary systems characterized by the presence of strong reflection (re-radiation) of the light of the hot star illuminating the surface of the cooler companion. Light curves are sinusoidal with the period equal to Porb, maximum brightness coinciding with the passage of the hot star in front of the companion. The eclipse may be absent. The range of light variation may reach 1 mag. in V. Example: KV Vel.',
    ),
    JoinedVarType(
        'Spotted',
        'ROT',
        description='''Spotted stars that weren't classified into a particular class. All the SPOTTED stars in the UNSW list and the very small amplitude spotted stars found by Kepler are included here. Also, some stars that don't fit the current subtypes due to their physical properties have been classified as such (brown dwarfs and white dwarfs with spots). It may be used as a subtype when a T Tauri star shows rotational variability (TTS/ROT, CTTS/ROT or WTTS/ROT).''',
    ),
    JoinedVarType(
        'RS CVn',
        'RS',
        description='''RS Canum Venaticorum-type binary systems. The primaries are usually giants from late F to late K spectral type. A significant property of these systems is the presence in their spectra of strong Ca II H and K emission lines of variable intensity, indicating increased chromospheric activity of the solar type. These systems are also characterized by the presence of radio and X-ray emission. Their light curves look like sine waves outside eclipses, with amplitudes and positions changing slowly with time. The presence of this wave (often called a distortion wave) is explained by differential rotation of the star, its surface being covered with groups of spots; the period of the rotation of a spot group is usually close to the period of orbital motion but still differs from it, which is the reason for the slow change (migration) of the phases of the distortion wave minimum and maximum in the mean light curve in the case of the eclipsing binaries (E/RS). The variability of the wave's amplitude (which may be up to 0.5 mag. in V) is explained by the existence of a long-period stellar activity cycle similar to the 11-year solar activity cycle, during which the number and total area of spots on the star's surface vary. Small amplitude flares are often observed too.''',
    ),
    JoinedVarType(
        'Other pulsating',
        'AHB1', 'ACYG', 'BCEP', 'BCEPS', 'BLAP', 'BXCIR', 'DWLYN', 'PPN', 'PVTEL', 'PVTELI', 'PVTELII', 'PVTELIII',
        'roAm', 'roAp', 'RV', 'RVA', 'RVB', 'SPB', 'V361HYA', 'V1093HER', 'LPV', 'PULS', 'PUL',
        description='''Other pulsating variables''',
    ),
    JoinedVarType(
        'Cepheid',
        'ACEP', 'CEP', 'CW', 'CWA', 'CWB', 'CWB(B)', 'DCEP', 'DCEP(B)', 'DCEPS', 'DCEPS(B)', 'CW-FO', 'CW-FU',
        'DCEP-FO', 'DCEP-FU',
        description='''Cepheids
        - ACEP. Anomalous Cepheids. Stars with periods characteristic of comparatively long-period RRAB variables (0.4 to 2 days), but considerably brighter by luminosity. They are more massive (1.3 to 2.2 solar masses) than RR Lyrae stars. They are metal-poor A and early F-type stars.
GCVS type BLBOO.
        - CEP. Cepheids. Radially pulsating, high luminosity (classes Ib-II) variables with periods in the range of 1-135 days and amplitudes from several hundredths to 2 mag. in V (in the B band, the amplitudes are greater). Spectral type at maximum light is F; at minimum, the types are G-K. The longer the period of light variation, the later is the spectral type. The maximum of the surface-layer expansion velocity almost coinciding with maximum light. There are several subtypes (see DCEP, DCEP(B), DCEPS, DCEPS(B), CWA, CWB and ACEP). Some DCEP and CW stars are quite often called Cepheids because it is often impossible to discriminate between them on the basis of the light curves for periods in the range 3 - 10 days. However, these are distinct groups of entirely different objects in different evolutionary stages. One of the significant spectral differences between W Virginis stars and Cepheids is the presence, during a certain phase interval, of hydrogen-line emission in the former and of Ca II H and K emission in the latter.
        - CW. Variables of the W Virginis type. These are pulsating variables of the galactic spherical component (old disk) population with periods of approximately 0.8 to 35 days and amplitudes from 0.3 to 1.2 mag. in V. They obey a period-luminosity relation different from that for δ Cep variables (see DCEP). For an equal period value, the W Vir variables are fainter than the δ Cep stars by 0.7 - 2 mag. The light curves of W Vir variables for some period intervals differ from those of δ Cep variables for corresponding periods either by amplitudes or by the presence of humps on their descending branches, sometimes turning into broad flat maxima. W Vir variables are present in globular clusters and at high galactic latitudes. They may be separated into the subtypes CWA and CWB.
        - DCEP. These are the classical Cepheids, or δ Cephei-type variables. Comparatively young objects that have left the main sequence and evolved into the instability strip of the Hertzsprung-Russell (H-R) diagram, they obey the well-known Cepheid period-luminosity relation and belong to the young disk population. DCEP stars are present in open clusters. They display a certain relation between the shapes of their light curves and their periods.'''
    ),
    JoinedVarType(
        'δ Sct',
        'DSCT', 'DSCTC', 'HADS', 'HADS(B)', 'SXPHE', 'SXPHE(B)', 'DSCTr',
        description='''Variables of the δ Scuti type.
        - DSCT. Variables of the δ Scuti type. These are pulsating variables of spectral types A0-F5 III-V displaying light amplitudes from 0.003 to 0.9 mag. in V (those with amplitudes larger than 0.15 mag. and assymetric light curves are designated HADS) and periods from 0.01 to 0.2 days. The shapes of the light curves, periods, and amplitudes usually vary greatly. Radial as well as non-radial pulsations are observed. The variability of some members of this type appears sporadically and sometimes completely ceases, this being a consequence of strong amplitude modulation with the lower value of the amplitude not exceeding 0.001 mag. in some cases. The maximum of the surface layer expansion does not lag behind the maximum light for more than 0.1 periods. DSCT stars are representatives of the galactic disk (flat component), SXPHE stars are halo objects.
        - DSCTC. Low-amplitude group of δ Scuti variables (light amplitude <0.1 mag. in V). The majority of this type's representatives are stars of luminosity class V; objects of this subtype generally are representative of the δ Sct variables in open clusters. This type has become obsolete in VSX since most DSCT have small amplitudes and the only clear distinction is the one between DSCT and HADS (amplitudes <0.15 mag.).
        - HADS. High Amplitude δ Scuti stars. They are radial pulsators showing asymmetric light curves (steep ascending branches) and amplitudes >0.15 mag.
        - HADS(B). First/second overtone double-mode δ Scuti variables. Period ratios P1/P0 = 0.77 and P2/P1 = 0.80.
        - SXPHE. Phenomenologically, these resemble HADS variables but they are pulsating sub-dwarfs of the spherical component, or old disk galactic population, with spectral types in the range A2-F5. They may show several simultaneous periods of oscillation, generally in the range 0.04-0.08 days, with variable-amplitude light changes that may reach 0.7 mag. in V. These stars are present in globular clusters.
        - SXPHE(B). Old population analogs to the double-mode HADS(B) stars.'''
    ),
    JoinedVarType(
        'γ Dor',
        'GDOR',
        description='''γ Doradus stars. They are high order g-mode non-radial pulsators, dwarfs (luminosity classes IV and V) from spectral types A7 to F7 showing one or multiple frequencies of variability. Amplitudes do not exceed 0.1 mag. and periods usually range from 0.3 to 3 days.'''
    ),
    JoinedVarType(
        'Slow irregular',
        'L', 'LB', 'LC',
        description='''Slow irregular variables.
        - L. Slow irregular variables. The light variations of these stars show no evidence of periodicity, or any periodicity present is very poorly defined and appears only occasionally. Stars are often attributed to this type because of being insufficiently studied. Many type L variables are really semi-regulars or belong to other types.
        - LB. Slow irregular variables of late spectral types (K, M, C, S); as a rule, they are giants. This type is also ascribed, in the GCVS, to slow red irregular variables in the case of unknown spectral types and luminosities.
Example: CO Cyg.
        - LC. Irregular variable supergiants of late spectral types having amplitudes of about 1 mag. in V.
Example: TZ Cas.''',
    ),
    JoinedVarType(
        'Mira',
        'M',
        description='''ο (omicron) Ceti-type (Mira) variables. These are long-period variable giants with characteristic late-type emission spectra (Me, Ce, Se) and light amplitudes from 2.5 to 11 mag. in V. Their periodicity is well pronounced, and the periods lie in the range between 80 and 1000 days. Infrared amplitudes are usually less than in the visible and may be <2.5 mag. For example, in the K band they usually do not exceed 0.9 mag.''',
    ),
    JoinedVarType(
        'RR Lyr',
        'RR', 'RRAB', 'RRC', 'RRc', 'RRD',
        description='''RR Lyrae.
        - RR. Variables of the RR Lyrae type, which are radially-pulsating giant A-F stars having amplitudes from 0.2 to 2 mag. in V. Cases of variable light-curve shapes as well as variable periods are known. If these changes are periodic, they are called the "Blazhko effect" (denoted by the subtype BL). The majority of these stars belong to the spherical component of the Galaxy; they are present, sometimes in large numbers, in some globular clusters, where they are known as pulsating horizontal-branch stars. Like Cepheids, maximum expansion velocities of surface layers for these stars practically coincide with maximum light.
        - RRAB. RR Lyrae variables with asymmetric light curves (steep ascending branches), periods from 0.3 to 1.0 days, and amplitudes from 0.5 to 2 mag. in V. They are fundamental mode pulsators.
        - RRC. RR Lyrae variables with nearly symmetric, sometimes sinusoidal, light curves, periods from 0.2 to 0.5 days, and amplitudes not greater than 0.8 mag. in V. They are overtone pulsators.
Example: SX UMa.
        - RRD. Double-mode RR Lyrae stars which pulsate in the fundamental mode as well as in the first overtone with a period ratio of 0.74 and a fundamental period near 0.5 days (or in the first and second overtones with a period ratio of 0.80).
GCVS class RR(B).'''
    ),
    JoinedVarType(
        'Semi-regular',
        'SR', 'Sr', 'SRA', 'SRB', 'SRC', 'SRD', 'SRS',
        description='''Semi-regular variables.
        - SR. Semi-regular variables, which are giants or supergiants of intermediate and late spectral types showing noticeable periodicity in their light changes, accompanied or sometimes interrupted by various irregularities. Periods lie in the range from 20 to >2000 days, while the shapes of the light curves are rather different and variable, and the amplitudes may be from several hundredths to several magnitudes (usually 1-2 mag. in V).'''
    ),
    JoinedVarType(
        'ZZ Ceti',
        'ZZ', 'ZZA', 'ZZB', 'ZZO', 'ZZLep',
        description='''ZZ Ceti variables. These are non-radially pulsating white dwarfs that change their brightnesses with periods from 30 s. to 25 min. and amplitudes from 0.001 to 0.2 mag. in V. They usually show several close period values.'''
    ),
    JoinedVarType(
        'Other eruptive',
        'BE', 'cPNB[e]', 'DPV', 'DYPer', 'EXOR', 'FF', 'FSCMa', 'FUOR', 'RCB', 'SDOR', 'WR',
        description='''Other eruptive variables'''
    ),
    JoinedVarType(
        'YSO',
        'CTTS', 'TTS', 'WTTS', 'YSO', 'DIP',
        description='''Young stellar objects'''
    ),
    JoinedVarType(
        'Irregular',
        'GCAS', 'UV', 'UVN', 'UXOR',
        'I', 'IA', 'IB', 'IN', 'INA', 'INAT', 'INT', 'INT(YY)', 'INB', 'INS', 'INSA', 'INSB', 'INSB(YY)',
        'INST', 'INST(YY)', 'IS', 'ISA', 'ISB',
        description='''Various irregular eruptive stars'''
    ),
    JoinedVarType(
        'Other cataclysmic',
        'AM', 'CBSS', 'DQ', 'IBWD', 'N', 'NA', 'NB', 'NC', 'NR',
        'SN', 'SN I', 'SN Ia', 'SN Iax', 'SN Ib', 'SN Ic', 'SN Ic-BL', 'SN II', 'SN IIa', 'SN IIb', 'SN IId', 'SN II-L',
        'SN IIn', 'SN II-P', 'SN-pec',
        'V838MON', 'ZAND', 'CV', 'HMXB', 'IMXB', 'LMXB', 'X', 'BHXB', 'XB',
        'Transient',
        description='''Interacting binary systems with white dwarfs or stars showing large amplitude outbursts'''
    ),
    JoinedVarType(
        'Nova-like',
        'NL',
        description='''Nova-like stars. Cataclysmic variables where the mass transfer rate is above a certain limit and their accretion disks are stable because they are nearly fully ionized to their outer (tidal cut off) boundary and this condition suppresses dwarf nova outbursts. Also known as UX (UX Ursae Majoris stars).'''
    ),
    JoinedVarType(
        'Dwarf nova',
        'UG', 'UGER', 'UGSS', 'UGSU', 'UGWZ', 'UGZ',
        description='''
U Geminorum-type variables, quite often called dwarf novae. They are close binary systems consisting of a dwarf or subgiant K-M star that fills the volume of its inner Roche lobe and a white dwarf surrounded by an accretion disk. Orbital periods are in the range 0.003-0.5 days. Usually only small, in some cases rapid, light fluctuations are observed, but from time to time the brightness of a system increases rapidly by several magnitudes and, after an interval of from several days to a month or more, returns to the original state. Intervals between two consecutive outbursts for a given star may vary greatly, but every star is characterized by a certain mean value of these intervals, i.e., a mean cycle that corresponds to the mean light amplitude. The longer the cycle, the greater the amplitude. The period given in VSX is usually the orbital period. Outburst cycles are given between parentheses. These systems are frequently sources of X-ray emission. The spectrum of a system at minimum is continuous, with broad H and He emission lines. At maximum these lines almost disappear or become shallow absorption lines. Some of these systems are eclipsing, possibly indicating that the primary minimum is caused by the eclipse of a hot spot that originates in the accretion disk from the infall of a gaseous stream from the K-M star. According to the characteristics of the light changes, U Gem variables may be subdivided into three types: SS Cyg-type (UGSS), SU UMa-type (UGSU), and Z Cam-type (UGZ).'''
    ),
    JoinedVarType(
        'Other',
        'AGN', 'BLLAC', 'GRB', 'QSO', 'Microlens', '*', 'S', 'VBD', 'APER', 'MISC', 'non-cv', 'NSIN', 'PER', 'SIN',
        'VAR', 'Galaxy',
        description='''Other variable types, including extragalactic, unique and rare types'''
    )
]


VSX_TYPE_MAP = {t: vsx_type.name for vsx_type in VSX_TYPES for t in vsx_type.types}
