from setuptools import setup

setup(
    name='ch-vars',
    version='0.0.1',
    packages=['ch_vars'],
    entry_points={'console_scripts': [
        'plot-var-stats = ch_vars.plot:main',
        'prepare = ch_vars.prepare_models:main',
        'cut-vsx = ch_vars.prepare_training_data:main',
        'put-vsx = ch_vars.vsx:main',
        'put-asassn-var = ch_vars.asassn:main',
        'put-sdss-82-candidates = ch_vars.sdss_candidates:main',
        'put-sdss-82-vars = ch_vars.sdss_vars:main',
        'plot-milky-way = ch_vars.prepare_models:plot_milky_way_entrypoint',
        'plot-extragal = ch_vars.prepare_models:plot_extragalactic_entrypoint',
        'plot-map = ch_vars.prepare_models:plot_map_entrypoint',
    ]},
    include_package_data=True,
    url='',
    license='',
    author='Konstantin Malanchev',
    author_email='',
    description=''
)
