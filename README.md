# flexTOMO

This project is a part of the larger X-ray tomographic reconstruction toolbox comprised of [flexDATA](https://github.com/cicwi/flexDATA), [flexTOMO](https://github.com/cicwi/flexTOMO) and [flexCALC](https://github.com/cicwi/flexCALC).
flexTOMO provides a wrapper around the GPU-accelerated tomographic reconstruction toolbox [ASTRA](https://github.com/astra-toolbox/astra-toolbox).
The main purpose of this project is to provide an easy way to use cone-beam forward- and back-projectors. Another purpose is to collect various algebraic reconstruction algorithms, providing support for large disk-mapped arrays (memmaps) and subsets that allow to both accelerate convergence and to save RAM.

## Getting Started

We recommend that the user installs [conda package manager](https://docs.anaconda.com/miniconda/) for Python 3.

### Installing with conda

`conda install flextomo -c cicwi -c astra-toolbox -c nvidia`

### Installing with pip

`pip install flextomo`

### Installing from source

```bash
git clone https://github.com/cicwi/flextomo.git
cd flextomo
pip install -e .
```

## Running the examples

To learn about the functionality of the package check out our `examples/` folder. Examples are separated into blocks that are best to run in VS Code / Spyder environment step-by-step.

## Modules

flexTOMO is comprised of two modules:

* `flextomo.phantom`: A very simple modelue with a few phantom object generators
* `flextomo.project`: Main module that contains forward- and back-projectors, and algebraic reconstruction algorithms

Typical usage:

```python
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
* **Willem Jan Palenstijn** - *Packaging, installation and maintenance*
* **Alexander Skorikov** - *Packaging, installation and maintenance*

See also the list of [contributors](https://github.com/cicwi/flexdata/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the GNU GENERAL PUBLIC License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* To Willem Jan Palenstijn for endless advices regarding the use of ASTRA toolbox.
