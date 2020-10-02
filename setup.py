from setuptools import setup

setup(
    name='ch-vars',
    version='0.0.1',
    packages=['ch_vars'],
    entry_points={'console_scripts': [
        'put-vsx = ch_vars.vsx:main',
        'put-asassn-var = ch_vars.asassn:main',
    ]},
    include_package_data=True,
    url='',
    license='',
    author='Konstantin Malanchev',
    author_email='',
    description=''
)
