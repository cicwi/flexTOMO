# flexTOMO

This project is a part of the larger X-ray tomographic reconstruction toolbox comprised of [flexDATA], [flexTOMO] and [flexCALC].
flexTOMO provides a wrapper around a GPU-based tomographic reconstruction toolbox [ASTRA](https://www.astra-toolbox.com/).
The main purpose of this project is to provide an easy way to use cone-beam forward- and back-projectors. Another purpose is to collect various algebraic reconstruction algorithms, providing support for large disk-mapped arrrays (memmaps) and subsets that allow to both accelerate convergence and to save RAM.

## Getting Started

Before installing flexTOMO, please download and install [flexDATA](https://github.com/cicwi/flexdata). Once installation of flexDATA is complete, one can install flexTOMO from the source code or using [Anaconda](https://www.anaconda.com/download/).

### Installing with conda

Simply install with:
```
conda create -n <your-environment> python=3.7
conda install -c cicwi -c astra-toolbox/label/dev -c conda-forge -c owlas flextomo
```

### Installing from source

To install flexTOMO you will need the latest version of the ASTRA toobox (preferably development version).
We also recommend to install [Xraylib](https://anaconda.org/conda-forge/xraylib) module using Anaconda from the Conda-Forge channel:

```
conda install -c astra-toolbox/label/dev astra-toolbox
conda install -c conda-forge xraylib
```

To install flexTOMO, simply clone this GitHub project. Go to the cloned directory and run PIP installer:
```
git clone https://github.com/cicwi/flextomo.git
cd flexdata
pip install -e .
```
## Running the examples

To learn about the functionality of the package check out our examples folder. Examples are separated into blocks that are best to run in Spyder environment step-by-step.

## Modules

flexTOMO is comprised of two modules:

* phantom:     a very simple modelue with a few phantom object generators
* project:    main module that contains forward- and back-projectors, and algebraic reconstruction algorithms

Typical code:
```
# Import:
import numpy

from flextomo import project
from flextomo import phantom

# Initialize projection images:
proj = numpy.zeros([512, 361, 512], dtype = 'float32')

# Define a simple projection geometry:
geom = geometry.circular(src2obj = 100, det2obj = 100, det_pixel = 0.01, ang_range = [0, 360])

# Create phantom and project into proj:
vol = phantom.abstract_nudes([512, 512, 512], geom, complexity = 10)

# Forward project:
project.forwardproject(proj, vol, geometry)
```

## Authors and contributors

* **Alexander Kostenko** - *Initial work*

See also the list of [contributors](https://github.com/cicwi/flexdata/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. Please submit pull requests against the `develop` branch.

If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the GNU GENERAL PUBLIC License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* To Willem Jan Palenstijn for endles advices regarding the use of ASTRA toolbox.
