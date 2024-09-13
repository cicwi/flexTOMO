from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='flextomo',
    version='1.0.0',
    description='ASTRA-based cone beam tomography reconstructions',
    url='https://github.com/cicwi/flextomo',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Alex Kostenko',
    license='GNU General Public License v3',
    packages=find_packages(include=['flextomo']),
    install_requires= [
        'numpy',
        'astra-toolbox>1.9.0',
        'tqdm',
        'scipy',
        'flexdata'
    ],
    extras_require={
        'dev': [
            'sphinx',
            'sphinx_rtd_theme',
            'myst-parser',
        ]
    }
)
