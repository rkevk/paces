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

class HamiltonianFramework:
    """
    Internal (non-user-exposed) object from which the master object for the generation of all Hamiltonian terms/matrix elements are derived.
    This object and its children (HamiltonianObjects and HilbertSkeleton) contain all of the Hilbert space and Hamiltonian generation information;
    all of these objects are time- and state-vector-agnostic, and HilbertSkeleton is also agnostic about the current set of basis states.
    """
    def __init__(self, use_terms, use_complex_type=numpy.complex128, use_module=cupy, debug_verb=0, search_mindiff=32, wordsize=32):
        if not ("vector_device" in globals() and "whoami_device" in globals()):
            raise NameError("The CUDA devices to be used must be defined as global variables (check file device_config.py)")
        if cupy.cuda.Device() != vector_device:
            raise ValueError("This class expects to be instantiated while vector_device is current.")
        self.complex_type   = use_complex_type
        self.use_module     = use_module
        self.debug_verb     = debug_verb
        self.search_mindiff = search_mindiff
        self.wordsize       = wordsize          # this determines the int type used to store the compressed basis states (best should be 32, resulting in uint32)
        if self.wordsize not in (8, 16, 32, 64):
            raise ValueError("The specified wordsize (%i) is illegal." % self.wordsize)
        self.dtype          = getattr(self.use_module, "uint%i" % self.wordsize)

        # use_terms is the main dict specifying the Hamiltonian.
        # This is a dict of dicts: the set of major keys specifies which terms to include, 
        #   each sub-dict specifies that term's parameter(s).
        self.use_terms      = use_terms
        # for convenience, define a dict with the diag terms removed:
        self.offdiag_terms  = {k: self.use_terms[k] for k in self.use_terms if k != "diag"}

        # old stuff that may be required in the future:
#        self.term_tuple     = tuple(self.use_terms.keys())  # keep a tuple of the names of the terms to have a guaranteed order for the keys
#        if self.periodic:
#            self.hopnum         = self.nchain
#        else:
#            self.hopnum         = self.nchain - 1


    def searchsorted(self, phonebook, findme, allow_escapes=False):
        if self.dtype == cupy.uint8:
            return cupy_search.searchsorted_multidim_list_vanilla(phonebook, findme, mindiff=self.search_mindiff, allow_escapes=allow_escapes, linear_only=False, shutup=True)
        elif self.dtype == cupy.uint32:
            return cupy_search.searchsorted_full_binary(phonebook, findme, allow_escapes)
#            return cupy_search.searchsorted_binary_only(phonebook, findme, allow_escapes, left_check=True)
#            return cupy_search.searchsorted_multidim_list_8bits(phonebook, findme, mindiff=self.search_mindiff, allow_escapes=allow_escapes, linear_only=False, shutup=True)
        else:
            raise TypeError("No searchsorted algorithm implemented for dtype %s." % self.dtype)

    ###################################
    # generate generate a mask to remove values that occur in both the input inds and plus_inds
    # (general helper function, not to be modified)
    ###################################
    def _generate_o_star_mask(self, basis_states, plus_inds, plus_mask):
        t0 = time.time()
        mpluind     = plus_inds[plus_mask]
        sortplus    = mpluind[cupy.lexsort(mpluind.T[::-1])]
        del mpluind
        o_star_mask = cupy.any(sortplus[self.searchsorted(sortplus, basis_states, allow_escapes=True)] != basis_states, axis=1)
        if self.debug_verb > searchsorted_timing_level:
            t1 = time.time()
            print("This application of (sorting +) searchsorted (+ masking) took %f ms (arg sizes %i, %i)." % ((t1-t0)*1000, sortplus.shape[0], basis_states.shape[0]))
        return o_star_mask


    ###################################
    # Wrapper function for Hamiltonian generation;
    # This is the heart of this object.
    ###################################
    def generate_mel(self, basis_states, enlarge_steps=0):
#        if basis_states.shape[0] > self.maxstates:
#            raise ValueError("Number of basis states exceeds maxstates.")
        if basis_states.shape[1] != self.totwordwidth:
            raise ValueError("Basis states do not match the number of bits specified at initialization.")
        for i in range(enlarge_steps):
            basis_states = self.enlarge_basis_set(basis_states)

        # Generate the various non-diagonal matrix elements:
        melpack_dict    = {}
        debug_dict      = {}
        len_dict        = {}
        totallen        = basis_states.shape[0]
        for i, term in enumerate(self.offdiag_terms):
            melpack, debug_dict[term]   = getattr(super(), "generate_mel_" + term)(basis_states)
            len1, len2                  = int(melpack[0][2].sum()), int(melpack[2][2].sum())
            len_dict[term]              = [len1, len2]
            melpack_dict[term]          = melpack
            totallen                    += len1 + len2
            if self.debug_verb > 2:
                print("    b.%i: Determined %s indices." % (i+1, term))
#                if item[2] in (Ellipsis, slice(None), True):
#                    lenlist += [item[0].shape[0],]

        # Now create a new array to hold all of the new indices:
        all_inds                            = cupy.empty((totallen, basis_states.shape[1]), dtype=self.dtype)

        # Start filling in all the indices, starting with the diagonal terms:
        all_inds[:basis_states.shape[0]]    = basis_states
        start                               = basis_states.shape[0]

        for term, (plus_triple, o_star, minus_triple) in melpack_dict.items():
            thislen                             = len_dict[term]
            midp                                = start+thislen[0]
            all_inds[start:midp]                = plus_triple[0][plus_triple[2]]
            all_inds[midp:midp+thislen[1]]      = minus_triple[0][minus_triple[2]]
#                if item[2] in (Ellipsis, slice(None), True):
#                    all_inds[start:start+lenlist[i]]    = item[0]
            start   += sum(thislen)

        if start != totallen:
            raise ValueError("Fatal error that I shall not further specify because I want to annoy you.")

        # Remove duplicates within the new indices:
        new_inds        = cupy_unique(all_inds)

        if self.debug_verb > 2:
            print("    b.%i: Determined new whoami array." % (len(self.offdiag_terms) + 1))

        diag_vals   = self.generate_diag_vals(new_inds)
        if self.debug_verb > 2:
            print("    Number of diag vals: %i" % diag_vals.shape[0])

        ### Now convert these indices into the dense vector indices:
        # (the diagonal elements are already given)
        dtype       = self.use_module.uint32
        if len(new_inds) >= 2**32:
            raise ValueError("The indexing array is too long to be stored in a 32-bit number format.")

        t0 = time.time()
        basis_lookup    = self.searchsorted(new_inds, basis_states)
        print_searchsorted_timing(self.debug_verb, time.time() - t0, new_inds.shape[0], basis_states.shape[0])

        # Based on the newly determined unique set of indices, convert the existing melpacks into the format that can be fed into the sparse matrix routines:
        COO_dict    = {}
        for term, melpack in melpack_dict.items():
            COO_dict[term]  = self.fill_in_param_arrays(new_inds, basis_lookup, *len_dict[term], *melpack, dtype)
            if self.debug_verb > mem_info_level:
                print("Near-maximal memory usage on whoami_device, est. 1: %1.1f MiB" % (cupy.get_default_memory_pool().used_bytes()/1024**2))
            del melpack

        if self.debug_verb > 2:
            print("    b.%i: Converted Hamiltonian matrix elements into dense coo format." % (len(self.offdiag_terms) + 2))
        if self.debug_verb > 3:
            t0              = time.time()
            unique, counts  = cupy.unique(self.get_pos(new_inds), return_counts=True)
            t1              = time.time()
            print("      Number of basis states with exciton at each site: " + str(dict(zip(unique.tolist(), counts.tolist()))) + " (this calculation took %f ms)" % ((t1 - t0)*1000))
#            print("      Number of positions where coupling is not diagonal: %i." % (new_inds[coupl_from][:,0] != new_inds[coupl_to][:,0]).sum())

        if (self.get_pos(new_inds[COO_dict["coupling"][1][0]]) != self.get_pos(new_inds[COO_dict["coupling"][1][1]])).sum() != 0:
            raise RuntimeError("Coupling matrix is not diagonal in the exciton Hilbert space!")
        return (diag_vals, new_inds), COO_dict, debug_dict



    ###################################
    # simple but annoying array-filling function
    ###################################
    def fill_in_param_arrays(self, new_inds, basis_lookup, seg1, seg2, plus_triple, o_star, minus_triple, dtype):
        # order: plus, plus c.c., minus, minus c.c.

        ### map_from indices:
        inds_from                   = self.use_module.empty(2 * (seg1 + seg2), dtype=dtype)

        # plus part:
        inds_from[:seg1]            = basis_lookup[plus_triple[2]]
        # plus, c.c.:
        t0 = time.time()
        findme                      = plus_triple[0][plus_triple[2]]

        inds_from[seg1:2*seg1]      = self.searchsorted(new_inds, findme)
        print_searchsorted_timing(self.debug_verb, time.time() - t0, new_inds.shape[0], findme.shape[0])

        prev    = 2*seg1
        # minus:
        inds_from[prev:prev+seg2]   = basis_lookup[o_star][minus_triple[2]]
        # minus, c.c.:
        t0 = time.time()
        findme                      = minus_triple[0][minus_triple[2]]

        inds_from[prev+seg2:]       = self.searchsorted(new_inds, findme)
        print_searchsorted_timing(self.debug_verb, time.time() - t0, new_inds.shape[0], findme.shape[0])

        ### map_to indices:
        inds_to                     = self.use_module.empty(2 * (seg1 + seg2), dtype=dtype)
        # plus part:
        inds_to[:seg1]              = inds_from[seg1:2*seg1]
        # plus, c.c.:
        inds_to[seg1:2*seg1]        = inds_from[:seg1]
        # minus part:
        inds_to[prev:prev+seg2]     = inds_from[prev+seg2:]
        # minus, c.c.:
        inds_to[prev+seg2:]         = inds_from[prev:prev+seg2]


        try:                    # assume the values *are* stored as arrays
            valdtype    = plus_triple[1].dtype
            if valdtype != minus_triple[1].dtype:
                raise TypeError("dtypes for plus and minus components of a Hamiltonian term were found to differ (%s and %s)." % (valdtype, minus_triple[1].dtype))

            vals                        = self.use_module.empty(len(inds_to), dtype=valdtype)
            vals[:seg1]                 = plus_triple[1][plus_triple[2]]
            vals[seg1:2*seg1]           = self.use_module.conj(plus_triple[1])[plus_triple[2]]
            vals[prev:prev+seg2]        = minus_triple[1][minus_triple[2]]
            vals[prev+seg2:]            = self.use_module.conj(minus_triple[1])[minus_triple[2]]

        except AttributeError:  # if the values are not stored as arrays
            valdtype    = type(plus_triple[1])
            if valdtype != type(minus_triple[1]):
                raise TypeError("Value types for plus and minus components of a Hamiltonian term were found to differ (%s and %s)." % (valdtype, type(minus_triple[1])))

            vals                        = self.use_module.empty(len(inds_to), dtype=valdtype)
            vals[:seg1]                 = plus_triple[1]
            vals[seg1:2*seg1]           = plus_triple[1].conjugate()
            vals[prev:prev+seg2]        = minus_triple[1]
            vals[prev+seg2:]            = minus_triple[1].conjugate()

        return vals, (inds_to, inds_from)



    ###################################
    # allow for multi-step adaptation by adding neighboring basis states to the existing basis set
    ###################################
    def enlarge_basis_set(self, basis_states):
        # Generate the various non-diagonal matrix elements:
        ind_dict    = {}
        len_dict    = {}
        totallen    = basis_states.shape[0]
        for i, term in enumerate(self.offdiag_terms):
            plus, minus     = getattr(super(), "generate_mel_" + term)(basis_states, raw_map_to=True)
            ind_dict[term]  = plus, minus
            len_dict[term]  = [len(plus), len(minus)]
            totallen        += sum(len_dict[term])

        # Now create a new array to hold all of the new indices:
        all_inds                            = cupy.empty((totallen, basis_states.shape[1]), dtype=self.dtype)

        # Start filling in all the indices, starting with the diagonal terms:
        all_inds[:basis_states.shape[0]]    = basis_states
        start                               = basis_states.shape[0]
        for term, (plus, minus) in ind_dict.items():
            thislen                             = len_dict[term]
            midp                                = start+thislen[0]
            all_inds[start:midp]                = plus
            all_inds[midp:midp+thislen[1]]      = minus
            start   += sum(thislen)
        if start != totallen:
            raise ValueError("Fatal error that I shall not further specify because I want to annoy you.")

        # Remove duplicates within the new indices:
        return cupy_unique(all_inds)

###################################################################################################################################################################################


