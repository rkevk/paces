"""observables: Hosts the class ObservablesFramework whose subclasses provide observable funcs."""

import os
import abc

import numpy

from ..config import INIT_VERBOSITY_LEVEL, OBSERVABLES_LEVEL
from ..aux.helpers import obs_attrs

####################################################################################################

class ObservablesFramework:
    """
    Abstract base class for the calculation of any observables over the course of a timeline.

    The user must define a subclass of this class for use with a specific model.
    Note that the TimeEvolution and Hamiltonian are accessible as the attributes
    self.teobj and self.hamobj and, for convenience, use_module is inherited as self.use_module.

    When designing a subclass, the following things must be taken into account:
    1.  If necessary, preprocessing can take place by defining a primer_hook method.
        This method may not have any arguments beyond self.
    2.  The heart of this class will be the methods that compute the observables.
        These must have the following signature/structure:
            @obs_attr(fname="fname for the observable", header="header for the file")
            def calculate_coupling(self, vector):
                return observable_list
        The decorator should contain the relative filename and the header line *without* time/norm.
        Note! The observables output should not be renormalized to account for lost norm.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, teobj, obs_list):
        """
        Initialize the Observables object and write the headers.

        Args:
            teobj: The instantiated subclass of TimeEvolution that this is attached to.
            obs_list (list): List of observables to compute.
        """
        if teobj.debug_verb > INIT_VERBOSITY_LEVEL:
            print("Initializing Observables object...")
        self.teobj      = teobj
        self.hamobj     = teobj.hamobj
        self.use_module = teobj.hamobj.use_module
        self.norm       = None
        self.obs_dict   = {}
        self.fname_dict = {}

        obs_dir    = os.path.join(teobj.dirname, "observables")
        if not os.path.exists(obs_dir):
            os.mkdir(obs_dir)
        for key in obs_list:
            try:
                meth    = getattr(self, "calculate_" + key)
            except AttributeError as exc:
                raise KeyError(f"Calculation method for {key} not found.") from exc

            self.obs_dict[key] = meth

            # Write headers and store fnames in centralized dict.
            try:
                fname                   = os.path.join(obs_dir, meth.fname)
            except AttributeError as exc:
                raise AttributeError(f"The observables function {meth} is missing a decorator"
                                        " specifying the fname for the observables file.") from exc
            self.fname_dict[key]    = fname
            for part in ["real", "imag"]:
                with open(".".join([fname, part]), "w", encoding="utf-8") as f:
                    f.write("#time norm " + meth.header + "\n")

        if self.teobj.te_params.diagnostics:
            extra_dbg = " ".join(self.hamobj.extra_dbg_list)
            if len(extra_dbg) > 0:
                extra_dbg = " " + extra_dbg
            self.fname_dict["diagnostics"] = os.path.join(obs_dir, "diagnostics.log")
            with open(self.fname_dict["diagnostics"], "w", encoding="utf-8") as diag_file:
                diag_file.write("#time norm seconds_since_start post_adapt_norm post_adapt_H"
                                " expm_converged final_m final_expm_term rel_error_expm numstates"
                                + extra_dbg + "\n")


    def primer_hook(self):
        """
        An initialization function to be run at the start of each set of observables calculation.

        This function is just a placeholder, children should override it if necessary.
        However, if the children do not require any preprocessing, this method can be left as-is.
        This function should not have any arguments beyond self.
        """
        pass


    @obs_attrs(fname="total_energy", header="energy")
    def calculate_total_energy(self, vector):
        """Compute the expected total energy of a vector."""
        h_ket = self.teobj.total_ham(vector, use_ham_mat=self.teobj.expm_params.use_scaling)
        return self.use_module.vdot(vector, h_ket).item()

    def compute_all_and_write(self, vector, t, norm, log_index):
        """
        Call all necessary observables functions and write the results to file.

        Args:
            vector (ndarray): The vector whose expectation values shall be computed.
            t (float): The current simulation time.
            norm (float): The norm of the vector
            log_index (int): The number to use for debug printing.
                This should be one larger than the last number used in the code that calls this.
        """
        for i, obsname in enumerate(self.obs_dict):
            result  = self.obs_dict[obsname](vector)
            if self.teobj.debug_verb > OBSERVABLES_LEVEL:
                print(f"      {log_index}.{i+1}a: Calculated {obsname}.")
            fname   = self.fname_dict[obsname]
            for part in ["real", "imag"]:
                with open(".".join([fname, part]), "a", encoding="utf-8") as f:
                    s_data  = numpy.append([t, norm], getattr(result, part))
                    numpy.savetxt(f, s_data, newline=" ")
                    f.write("\n")
            if self.teobj.debug_verb > OBSERVABLES_LEVEL:
                print(f"      {log_index}.{i+1}b: Saved {obsname} to file.")
