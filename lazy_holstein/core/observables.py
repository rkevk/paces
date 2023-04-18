#!/usr/bin/env python3
# coding: utf-8
 
import time
import inspect
import sys
import os
import os.path
import pprint

import numpy
import cupy
import cupyx

from ..aux.helpers import *
from ..device_config import *

#import logging

#logging.basicConfig(filename="last_instance.log", filemode='w')

#if os.path.dirname(sys.argv[0]) != '':
#    os.chdir(os.path.dirname(sys.argv[0]))

#for mod in (cupy, numpy):
#    mod.set_printoptions(linewidth=200, edgeitems=10)

if numpy.uintc != numpy.uint32:
    raise TypeError("Well well well, who's working on a non-64 bit system? This code will explode if run on a system whose integer size is not 32 bits.")


###################################################################################################################################################################################

class ObservablesFramework:
    """
    Object responsible for the calculation of any observables over the course of a timeline.
    The functions within this object are time-agnostic, but take the current state-vector as an argument.
    """
    def __init__(self, obs_list, dirname):
        if verbose:
            print("Initializing Observables object...")
        self.dirname    = dirname

        fmode_dict = {"n_b": "ab", "reduced_dm": "ab", "globals": "ab", "diagnostics": "ab", "phonon": "ab"}
        for item in obs_list:
            if not item in  ["H", "hopping", "coupling", "n_b", "reduced_dm", "diagnostics"] + ["phonon%i" % i for i in range(self.HamObj.nchain)] + ["phononproj%i" % i for i in range(self.HamObj.nchain)]:
                raise ValueError('Unrecognized observable "%s".' % item)

        hopstring       = b"hopping " + b''.join([b"hop%i " %i for i in range(self.HamObj.nchain - 1 + int(self.HamObj.periodic))])
        vibstring       = b''.join([b"vib%i " %i for i in range(self.HamObj.nchain)])
        H_file_header   = b"#tag norm " + b"H_tevol " * ("H" in obs_list) + hopstring * ("hopping" in obs_list) + vibstring * ("coupling" in obs_list) + b'\n'
        if "H" in obs_list or "hopping" in obs_list or "coupling" in obs_list:
            with open(os.path.join(self.dirname, "global_operators_real"), 'wb') as H_real_file, open(os.path.join(self.dirname, "global_operators_imag"), 'wb') as H_imag_file:
                H_real_file.write(H_file_header)
                H_imag_file.write(H_file_header)
        if "reduced_dm" in obs_list:
            with open(os.path.join(self.dirname, "dm_files/reduced_dm_real"), 'wb') as dm_real_file, open(os.path.join(self.dirname, "dm_files/reduced_dm_imag"), 'wb') as dm_imag_file:
                numstring = [b"rho_{%i,%i}" % (i, j) for i in range(9) for j in range(9)]
                dm_real_file.write(b"#tag norm " + b' '.join(numstring) + b'\n')
                dm_imag_file.write(b"#tag norm " + b' '.join(numstring) + b'\n')
        if "diagnostics" in obs_list:
            with open(os.path.join(self.dirname, "diagnostics.log"), 'wb') as diag_file:
                diag_file.write(b"#tag norm seconds_since_start post_adapt_norm post_adapt_H expm_converged final_m final_expm_term rel_error_expm numstates ceiling_hits\n")
        if "n_b" in obs_list:
            with open(os.path.join(self.dirname, "local_operators/n_b_real"), 'wb') as n_b_file:
                n_b_file.write(b"#tag norm n^exc_0 n^pho_0 max_HO-n^pho_0 n^exc_1 n^pho_1 max_HO-n^pho_1 etc...\n")

        phononrdm_list  = sorted([i for i in obs_list if i[:6] == "phonon"])
        phononrdm_file_list = []
        if len(phononrdm_list) > 0:
            for obsv in phononrdm_list:
                if obsv[6:10] == "proj":
                    rdmsite     = int(obsv[10:])
                    sname       = "dm_files/phonon_reduced_dm_site%i_projected" % rdmsite
                else:
                    rdmsite     = int(obsv[6:])
                    sname       = "dm_files/phonon_reduced_dm_site%i" % rdmsite
                with open(os.path.join(self.dirname, sname), 'wb') as rdm_phonon_file:
                    n_b_file.write(b"#tag norm rho_00 rho_01 rho_02 etc...\n")
                phononrdm_file_list += [sname,]


#################################### CONTINUE HERE

    def calculate_and_write(self, vector):
            ### Start computing observables:
        if "reduced_dm" in obs_list:
            raise NotImplementedError("The reduced density matrix calculation has not yet been converted to the generalized data segmentation.")
            dm_complex  = self.calculate_reduced_dm(vector, mindiff=dm_mindiff)
            if type(dm_complex) != numpy.ndarray:
                dm_complex = cupy.asnumpy(dm_complex)
            dm_real     = dm_complex.real
            dm_imag     = dm_complex.imag
            if self.HamObj.periodic:
                hopping_complex     = numpy.empty(self.HamObj.hopnum)
                hopping_complex[:-1]= 2 * self.HamObj.t_sys * numpy.diag(dm_real, 1)
                hopping_complex[-1] = 2 * self.HamObj.t_sys * dm_real[0,-1]
            else:
                hopping_complex     = 2 * self.HamObj.t_sys * numpy.diag(dm_real, 1)
            n_b_sys     = self.use_module.diag(dm_real)
            if self.debug_verb > 0:
                print("Check 5.1: Calculated reduced density matrix.")
            with open(os.path.join(self.dirname, "dm_files/reduced_dm_real"), fmode_dict["reduced_dm"]) as dm_real_file:
                numpy.savetxt(dm_real_file, numpy.append([t, norm], dm_real.flatten()), newline=" ")
                dm_real_file.write(b'\n')
            with open(os.path.join(self.dirname, "dm_files/reduced_dm_imag"), fmode_dict["reduced_dm"]) as dm_imag_file:
                numpy.savetxt(dm_imag_file, numpy.append([t, norm], dm_imag.flatten()), newline=" ")
                dm_imag_file.write(b'\n')
            if self.debug_verb > 0:
                print("Check 5.2: Saved reduced density matrix.")
            del dm_complex


        for obsv in phononrdm_file_list:
            if obsv[-10:] == "_projected":
                rdmsite     = int(obsv[31:-10])
                mask        = self.HamObj.get_pos(self.whoami) == rdmsite
                phononrdm   = phonon_rdm_funcs(self.HamObj).calculate_rdm(rdmsite, self.whoami[mask], vector[mask])
                del mask
            else:
                rdmsite     = int(obsv[31:])
                phononrdm   = phonon_rdm_funcs(self.HamObj).calculate_rdm(rdmsite, self.whoami, vector)
            with open(os.path.join(self.dirname, obsv), fmode_dict["phonon"]) as rdm_phonon_file:
                numpy.savetxt(rdm_phonon_file, numpy.append([t, norm], cupy.asnumpy(phononrdm).flatten()), newline=" ")
                rdm_phonon_file.write(b'\n')

        if "n_b" in obs_list:
            n_b_result          = self.use_module.empty(self.HamObj.nchain*3)
            if n_b_sys is None:
                n_b_sys         = self.calculate_sys_n_b(vector)
            n_b_result[::3]     = n_b_sys
            n_b_result[1::3]    = self.calculate_bath_n_b(vector)
            n_b_result[2::3]    = self.HamObj.max_HO_dims_v - n_b_result[1::3]
            if type(n_b_result) != numpy.ndarray:
                n_b_result = cupy.asnumpy(n_b_result)
            n_b                 = numpy.append([t, norm], n_b_result)
            if self.debug_verb > 0:
                print("Check 6.1: Calculated occupations.")
            with open(os.path.join(self.dirname, "local_operators/n_b_real"), fmode_dict["n_b"]) as n_b_file:
                numpy.savetxt(n_b_file, n_b, newline=" ")
                n_b_file.write(b'\n')
            if self.debug_verb > 0:
                print("Check 6.2: Saved occupations.")
        if "H" in obs_list:
            H_complex   = self.calculate_total_energy(vector)
            H_real      = H_complex.real
            H_imag      = H_complex.imag
            if self.debug_verb > 0:
                print("Check 7: Calculated energies.")
            H_real_file_savedata += [H_real]
            H_imag_file_savedata += [H_imag]
        if "hopping" in obs_list:
            H_real_file_savedata += [hopping_complex.real.sum()] + list(hopping_complex.real)
            H_imag_file_savedata += [hopping_complex.imag.sum()] + list(hopping_complex.imag)
        if "coupling" in obs_list:
            coupling_complex = self.calculate_coupling(vector)
            if self.debug_verb > 0:
                print("Check 8: Calculated coupling terms.")
            H_real_file_savedata += list(coupling_complex.real)
            H_imag_file_savedata += list(coupling_complex.imag)
        if "coupling" in obs_list or "H" in obs_list or "hopping" in obs_list:
            with open(os.path.join(self.dirname, "global_operators_real"), fmode_dict["globals"]) as H_real_file, open(os.path.join(self.dirname, "global_operators_imag"), fmode_dict["globals"]) as H_imag_file:
                numpy.savetxt(H_real_file, H_real_file_savedata, newline=' ')
                numpy.savetxt(H_imag_file, H_imag_file_savedata, newline=' ')
                H_real_file.write(b'\n')
                H_imag_file.write(b'\n')
            if self.debug_verb > 0:
                print("Check 9: Saved non-local observables data.")
        if (save_first and i == 0) or (save_last and i == len(t_array) - 1) or (save_every is not None and i % save_every == 0):
            self.use_module.save(wf_name_list[i], vector)
            self.use_module.save(wf_name_list[i].replace("wf_file_", "whoami_"), self.whoami)   # this works even if self.whoami is not on current_device
        if "diagnostics" in obs_list:
#   new format:
#                diag_file.write(b"#tag norm seconds_since_start post_adapt_norm post_adapt_H expm_converged final_m final_expm_term rel_error_expm numstates ceiling_hits\n")
            with open(os.path.join(self.dirname, "diagnostics.log"), fmode_dict["diagnostics"]) as diag_file:
                if t == 0:
                    numpy.savetxt(diag_file, [t, norm, time.time() - timeline_start_time, post_adapt_norm, post_adapt_H, 0, 0, 0, 0, self.numstates, 0], newline=" ")
                else:
                    numpy.savetxt(diag_file, [t, norm, time.time() - timeline_start_time, post_adapt_norm, post_adapt_H, expm_converged, final_m, final_expm_term.item(), rel_error_expm.item(), self.numstates, ceiling_hits.item()], newline=" ")
                diag_file.write(b'\n')

        if self.debug_verb > mem_info_level:
            print("Used MiB on vector_device: %1.1f or %1.1f" % (cupy.get_default_memory_pool().used_bytes()/1024**2, numpy.diff(cupy.cuda.Device(vector_device).mem_info)[0]/1024**2))
            with cupy.cuda.Device(whoami_device):
                print("Used MiB on whoami_device: %1.1f or %1.1f" % (cupy.get_default_memory_pool().used_bytes()/1024**2, numpy.diff(cupy.cuda.Device(whoami_device).mem_info)[0]/1024**2))



###################################################################################################################################################################################


