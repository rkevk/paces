# paces: Parallelized Application of Co-Evolving Subspaces
## A method for computing quantum dynamics on GPUs
The `main` branch will be updated with new features to support a wider variety of models.
If you are interested in the single-exciton Holstein version that was used to create the initial data
shown in the publication `[link to be inserted here]`, please see the [holstein_only](https://github.com/rkevk/paces/tree/holstein_only) branch instead.

## Dependencies:
This code requires [CuPy](https://cupy.dev) and its dependencies,
chiefly [Python 3](https://python.org), [NumPy](https://numpy.org)
and [CUDA](https://developer.nvidia.com/cuda-gpus).
It is known to work with the following combination of versions,
but will most likely work with newer versions as well:
* CuPy 11.2
* CUDA 11.8 with cuDNN 8.4.0.27, cuTENSOR 1.5.0.3, NCCL 2.14.3
* Python 3.10.8
* NumPy 1.24.1

## Running a calculation with an existing model
Choose a model to run from the `models` submodule and then do the following:

### Initial (one-time) setup:
After having ensured that the dependencies are met:
Clone this repository, then copy `paces/config/device_config_template.py` to `paces/config/device_config.py`
and make necessary changes to the `device_config` file in line with your system setup (number of GPUs, memory).
If you choose to push commits later on, the `device_config` file and any changes made to it will be excluded (via `.gitignore`).

### Per-calculation setup:
As an example, let the working directory of our shell be the directory containing this `README.md`
and say we want to store our files in `../parent_dir/main_calc_dir`.
We will run the example calculation given in `main_example.py` (a single-exciton 1D Holstein chain).

Then the steps are:
1. Ensure that `../parent_dir` already exists.
2. Create the calculation directory and its subdirectories `wf_coeffs` and `observables`.
    This can be automated using `bash prepare_dir.sh ../parent_dir/main_calc_dir`.
3. Set the parameters of the calculation as desired in the short wrapper file `main_example.py`.
4. Run the calculation using `python3 main_example.py`.

By default, the calculation directory specified within `main_example.py` is given as a relative path to `main_example.py` itself,
not to the location from which it is called. To change this behavior, remove the `os.chdir` call from the start of the file
(or use absolute paths).

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
