from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    author="Alex Kostenko",
    description="ASTRA-based cone beam tomography reconstructions.",
    install_requires= [
        "numpy",
        "astra-toolbox>1.9.0",
        "tqdm",
        "scipy",
        "flexdata"
    ],
    license="GNU General Public License v3",
    long_description=readme,
    long_description_content_type="text/markdown",
    name='flextomo',
    packages=find_packages(include=['flextomo']),
    extras_require={
        'dev': [
            'sphinx',
            'sphinx_rtd_theme',
            'myst-parser',
        ]
    },
    url='https://github.com/cicwi/flextomo',
    version='1.0.0',
)
