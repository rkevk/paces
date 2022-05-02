# lazy_holstein, or:
## Parallelized Multi-Step Dynamically Adapted (ParMuDA) basis set method for quantum dynamics

## Initial (one-time) setup:
Copy `device_config_template.py` to `device_config.py` and make necessary changes according to your system setup (number of GPUs, memory).
`device_config.py` and any changes made to it will not be synced via git.

## Per-calculation setup:
Before starting a calculation, first create a suitable subdirectory structure to store the data in.
The structure for existing calculations has been similar to: `adaptive_results/vib_gendatseg/${SIGN}t_n${NUMSITES}_d${DIM_HO}_g${G_COUPL}_delta_eps${DELTA_EPS}_coherence`, where 
- `SIGN` is the sign of the dipole-dipole interaction (`pos` or `neg`),
- `NUMSITES` is the chain length,
- `DIM_HO` is the dimension of the local quantum harmonic oscillators (max. number of phonons - 1),
- `G_COUPL` is the electron-phonon coupling strength times ten, as an integer,
- `DELTA_EPS` is the tilt parameter times a power of ten, as an integer,
- and `coherence` stands for the `coherence` method of weighting basis states.

This structure is by no means mandatory and can be changed as desired.

As an example, we assume you are located in the directory containing this `README.md` and say we want to store our files in `../adaptive_results/vib_gendatseg/post_n25_d64_g20_delta_eps100_coherence`.
Then the steps are:
1. Create all but the lowest level of directories: `mkdir ../adaptive_results; mkdir ../adaptive_results/vib_gendatseg`.
2. Create the actual calculation directory itself including necessary subdirectories using `bash prepare_dir.sh ../adaptive_results/vib_gendatseg/post_n25_d64_g20_delta_eps100_coherence`.
3. Set the parameters of the calculation as desired in the short wrapper file `main.py` (recommended to make these at least match the directory description...). Note that the calculation directory must also be specified in `main.py`.
4. Run the calculation using `python3 main.py`.

By default, the calculation directory specified within `main.py` is given as a relative path to `main.py` itself, not to the location from which it is called. To change this behavior, remove the `os.chdir` call from the start of the file.
