# paces: Parallelized Application of Co-Evolving Subspaces
## A method for computing quantum dynamics on GPUs
The `main` branch will be updated with new features to support a wider variety of models.
If you are interested in the single-exciton Holstein version that was used to create the initial data
shown in the publication `[link to be inserted here]`, please see the [holstein_only](https://github.com/rkevk/paces/tree/holstein_only) branch instead.

## Dependencies:
This code requires [CuPy](https://cupy.dev) and its dependencies,
chiefly [Python 3](https://python.org), [NumPy](https://numpy.org)
and [CUDA](https://developer.nvidia.com/cuda-gpus).
It has been tested with the following versions:
* CuPy from 12.2 up to 14.0.1
* CUDA 11.8 and 12.0
* NumPy from 1.24 up to 2.4
* Python from 3.10 up to 3.13.3

## Running a calculation with an existing model
After cloning the repository, choose a model to run from the `models` submodule and then do the following.

We will use the example calculation given in `main_example.py` (a single-exciton 1D Holstein chain),
located in the same directory as this `README.md`. This shall also be the working directory of our shell.
Say we want to store our files in `../../paces_results/example/`.

Then the steps are:
1. Ensure that the calculation directory `../../paces_results/example/` exists.
2. Set the parameters of the calculation as desired in the main wrapper file `main_example.py`.
3. Run the calculation using `python3 main_example.py`.

By default, the calculation directory specified within `main_example.py` is given as a
relative path to `main_example.py` itself, not to the location from which it is called.
To change this behavior, remove the `os.chdir` call from the start of the file (or use absolute paths).

### Installing `paces` for use from other locations
`paces` can also be installed via pip if you would like to import it as a library
from an arbitrary working directory (preferably in a [venv](https://docs.python.org/3/library/venv.html)).
Due to the way CuPy is packaged and interfaces with CUDA, this is slightly non-trivial.
Note that you will need to download the file `main_example.py` separately from GitHub
if installing `paces` via pip.

#### If you already have a working CuPy installation...
... then you can simply run either
```
pip install /path/to/downloaded/repo
```
if you have already cloned the repository, or
```
pip install "paces @ git+https://github.com/rkevk/paces.git"
```
to install directly from GitHub.

#### If you don't have a working CuPy installation yet...
... then you at least need a working CUDA installation including nvcc (part of the dev toolkit).
Once you have a working CUDA installation, you can install `paces` and CuPy at the same time
by specifying the version of CUDA you have installed, e.g.:
```
pip install /path/to/downloaded/repo[cuda12]
```
or
```
pip install "paces[cuda12] @ git+https://github.com/rkevk/paces.git"
```
for CUDA versions 12.x. The available options are `cuda11`, `cuda12` and `cuda13`.
Any other versions of CUDA will require you to get CuPy running in advance.

## Internal structure & defining a new model
Defining a new model (i.e., a new "type" of Hamiltonian) is more involved.

There are three fundamental classes which any calculation is based on:
- `Hamiltonian` introduces the actual Hamiltonian generation procedure based on the current basis state.
    This is the heart of any calculation. At the beginning of a calculation, a single, unchanging instance
    of `Hamiltonian` is created upon which everything else builds.
- `Observables` contains observable calculation routines. It must also depend on the Hamiltonian model in question,
    but is only ever used during the observables-computation part of each time step.
- `TimeEvolution` takes an instance of `Hamiltonian` *and* an uninstantiated concretized `Observables` as arguments.
    It uses the former to continuously regenerate new Hamiltonians, expand bases, etc.,
    while the latter is instantiated appropriately and then used to compute expectation values during the time evolution.

Each of these three must be defined as a concretized subclass of the model-independent respective framework classes
`HamiltonianFramework`, `ObservablesFramework`, `TimeEvolutionFramework`. 
The abstract base classes are contained in the `core` submodule,
whereas the concretized subclasses are contained in the `models` submodule.
If you are interested in developing a new model,
consider checking the existing models in the `models` submodule to understand how they are built.
