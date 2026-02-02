# paces: 
## Parallelized Application of Co-Evolving Subspaces, a method for computing quantum dynamics on GPUs

This branch contains the version that was used to create the initial data shown in the publication `[link to be inserted here]` for archival purposes.
It only supports the single-exciton Holstein model, while the other branches will continue development to support other classes of Hamiltonians.

## Dependencies:
This code requires [SciPy](https://scipy.org) and [CuPy](https://cupy.dev) and its dependencies, chiefly [Python 3](https://python.org), [NumPy](https://numpy.org), [CUDA](https://developer.nvidia.com/cuda-gpus).
It is known to work with the following combination of versions, but will most likely work with other versions as well:
* CuPy 11.2
* CUDA 11.8 with cuDNN 8.4.0.27, cuTENSOR 1.5.0.3, NCCL 2.14.3
* Python 3.10.8
* NumPy 1.24.1
* SciPy 1.10.0

## Initial (one-time) setup:
After having ensured that the dependencies are met:
Copy `device_config_template.py` to `device_config.py` and make necessary changes according to your system setup (number of GPUs, memory).
`device_config.py` and any changes made to it will not be synced via git.

## Per-calculation setup:
Before starting a calculation, first create a suitable subdirectory structure to store the data in.

As an example, we assume you are located in the directory containing this `README.md` and say we want to store our files in `../parent_dir/main_calc_dir`.
Then the steps are:
1. Ensure that `../parent_dir` already exists.
2. Create the actual calculation directory itself including necessary subdirectories using `bash prepare_dir.sh ../parent_dir/main_calc_dir`.
3. Set the parameters of the calculation as desired in the short wrapper file `main.py`.
4. Run the calculation using `python3 main.py`.

By default, the calculation directory specified within `main.py` is given as a relative path to `main.py` itself, not to the location from which it is called. To change this behavior, remove the `os.chdir` call from the start of the file.
