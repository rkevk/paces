# paces: Parallelized Application of Co-Evolving Subspaces
## A method for computing quantum dynamics on GPUs
The `main` branch will be updated with new features to support a wider variety of models.
If you are interested in the single-exciton Holstein version that was used to create the initial data
shown in the publication `[link to be inserted here]`, please see the [holstein_only](https://github.com/rkevk/paces/tree/holstein_only) branch instead.

## Dependencies:
This code requires [CuPy](https://cupy.dev) and its dependencies,
chiefly [Python 3](https://python.org), [NumPy](https://numpy.org)
and [CUDA](https://developer.nvidia.com/cuda-gpus).
It is known to work with the following combinations of versions:
1. * CuPy 12.2.0
   * CUDA 11.8 with cuDNN 8.4.0.27, cuTENSOR 1.5.0.3, NCCL 2.14.3
   * NumPy 1.24.1
   * Python 3.10.8
2. * CuPy 13.6.0
   * CUDA 11.8 with cuTENSOR 1.6.2.3
   * NumPy 2.4.2
   * Python 3.12.3
3. * CuPy 14.0.1
   * CUDA 12.0
   * NumPy 2.3.1
   * Python 3.13.3

## Running a calculation with an existing model
Choose a model to run from the `models` submodule and then do the following.
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
