#!/usr/bin/env python3
# coding: utf-8
 
import time
import inspect
import sys
import os
import os.path

import numpy
import cupy
import cupyx
import _lh_aux.cupy_expm_multiply as cupy_expm_multiply
import _lh_aux.cupy_search as cupy_search
import _lh_aux.cupy_gendatseg_kernels as cupy_gendatseg_kernels

from device_config import *

#import logging

#logging.basicConfig(filename="last_instance.log", filemode='w')

#if os.path.dirname(sys.argv[0]) != '':
#    os.chdir(os.path.dirname(sys.argv[0]))

#for mod in (cupy, numpy):
#    mod.set_printoptions(linewidth=200, edgeitems=10)

if numpy.uintc != numpy.uint32:
    raise TypeError("Well well well, who's working on a non-64 bit system? This code will explode if run on a system whose integer size is not 32 bits.")

###################################################################################################################################################################################

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = numpy.result_type(*arrays)
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def format_function_args(frame, start_time=None):
    if start_time is None:
        start_time  = time.time()
    args, _, _, values  = inspect.getargvalues(frame)
    arg_list            = [(str(i) + "=" + str(values[i])) for i in args if str(i) != "self"]
    fname               = frame.f_code.co_name
    return ("\nFunction call at %f:\n   %s(" % (start_time, fname)) + ', '.join(arg_list) + ")\n"



###################################################################################################################################################################################


###################################
# global variables:
###################################
# the following are debug verbosity levels, nothing critical:
searchsorted_timing_level       = 999
simple_taylor_level             = 6
move_data_timing_level          = 4
mem_info_level                  = 4


###################################################################################################################################################################################


class phonon_rdm_funcs:
    def __init__(self, HamObj):
        self.searchsorted   = HamObj.searchsorted
        self.get_phonon_occ = HamObj.get_phonon_occ
        self.add_n_to_site  = HamObj.add_n_to_site
        self.QHO_dims       = HamObj.max_HO_dims_v

    ###################################
    # phonon reduced dm from a given vector
    ###################################
    def calculate_rdm(site, whoami, vector):
        occs        = self.get_phonon_occ(whoami, site)
        uniques     = cupy.unique(occs)
        adagger     = self.csr_adagger(whoami, site)
        dim         = self.QHO_dims[site]
        upper_tri   = cupy.zeros((dim, dim), dtype=complex)
        rightvec    = vector.copy()
        for diarow in range(dim):
            vecprod     = numpy.conj(vector) * rightvec
            this_res    = cupy.zeros(dim - diarow, dtype=complex)
            for val in uniques:         # this creates entry rho_{val-diarow,val}
                if val < diarow:
                    continue
                this_res[val-diarow]    = cupy.sum(vecprod[occs == val])
            upper_tri   += cupy.diag(this_res, diarow)
            rightvec    = adagger.dot(rightvec)

        reduced_dm  = upper_tri + cupy.conj(cupy.triu(upper_tri, 1).T)
        return reduced_dm


    ###################################
    # generate phonon adagger csr matrix
    ###################################
    def csr_adagger(basis_states, site):
        plus_inds       = self.generate_mel_adagger_noprefac(basis_states, site)
        inds_to         = self.searchsorted(basis_states, plus_inds, allow_escapes=True)
        mask            = cupy.all(basis_states[inds_to] == plus_inds, axis=1)
        numbs           = len(basis_states)
        return cupy.sparse.coo_matrix((cupy.ones(mask.sum().item()), (inds_to[mask], cupy.arange(numbs, dtype=inds_to.dtype)[mask])), shape=(numbs, numbs)).tocsr()


    ###################################
    # generate creation operator matrix elements, but without sqrt prefactor
    ###################################
    def generate_mel_adagger_noprefac(basis_states, site):
        plus_inds               = basis_states.copy()
        self.add_n_to_site(plus_inds, 1, site)
        return plus_inds


###################################################################################################################################################################################


class lazy_holstein_model:
    def __init__(self, nchain, eps_sys, t_sys, eps_bath, coupling_g, periodic, maxstates, max_HO_dims, delta_eps=0, verbose=True, use_complex_type=numpy.complex128, use_module=cupy, debug_verb=0, search_mindiff=32, wordsize=32):
        if not ("vector_device" in globals() and "whoami_device" in globals()):
            raise NameError("The CUDA devices to be used must be defined as global variables.\n\tIf you are importing lazy_holstein as a module, then set lazy_holstein.xxxxxx_device = cupy.cuda.Device(i).")
        if cupy.cuda.Device() != vector_device:
            raise ValueError("This class expects to be instantiated while vector_device is current.")
        self.use_module     = use_module
        self.nchain         = nchain
        self.eps_sys        = eps_sys
        self.t_sys          = t_sys
        self.eps_bath       = eps_bath
        self.coupling_g     = coupling_g
        self.delta_eps      = delta_eps         # this is the tilt in the excitonic potential energy
        self.verbose        = verbose
        self.complex_type   = use_complex_type
        self.periodic       = bool(periodic)
        self.maxstates      = maxstates
        self.debug_verb     = debug_verb
        self.search_mindiff = search_mindiff
        self.wordsize       = wordsize          # this determines the int type used to store the compressed basis states (best should be 32, resulting in uint32)
        with vector_device:
            self.max_HO_dims_v  = self.use_module.asarray(max_HO_dims)
        with whoami_device:
            self.max_HO_dims_w  = self.use_module.asarray(max_HO_dims)

        if self.periodic:
            self.hopnum         = self.nchain
        else:
            self.hopnum         = self.nchain - 1
        if self.wordsize not in (8, 16, 32, 64):
            raise ValueError("The specified wordsize (%i) is illegal." % self.wordsize)
        if self.max_HO_dims_v.max() > 2**self.wordsize:
            raise ValueError("One of the QHO dimensions given as input exceeds %i bits (= %i)." % (self.wordsize, 2**self.wordsize))
        if self.max_HO_dims_v.shape != (self.nchain,):
            raise ValueError("Length of max_HO_dims does not match chain length.")

        self.dtype          = getattr(self.use_module, "uint%i" % self.wordsize)

#        self.max_dim_Hilbert= self.nchain * numpy.prod(self.max_HO_dims)

        self.posbitwidth    = int(self.use_module.ceil(self.use_module.log2(self.nchain)))                          # this is a single int
        with vector_device:
            self.QHObitwidth_v  = cupy.ceil(cupy.log2(self.max_HO_dims_v)).astype(self.dtype)                       # this is a cupy array on vector_device
        with whoami_device:
            self.QHObitwidth_w  = cupy.ceil(cupy.log2(self.max_HO_dims_w)).astype(self.dtype)                       # this is a cupy array on whoami_device
        self.totwordwidth   = int(self.use_module.ceil((self.posbitwidth + self.QHObitwidth_v.sum())/self.wordsize))  # this is a single int

        if self.posbitwidth > 16:
            raise NotImplementedError("Chain lengths requiring more than 16 bits to store (N > 65536) are not yet implemented.")
        if self.periodic:
            raise NotImplementedError("Periodic boundary conditions should not be considered correctly implemented yet.")


    def print_searchsorted_timing(self, delta_t, size1, size2):
        if self.debug_verb > searchsorted_timing_level:
            print("This application of searchsorted took %f ms (arg sizes %i, %i)." % (delta_t*1000, size1, size2))
            return
        else:
            return


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
    # data segmentation functions, 
    # basic compression/decompression
    ###################################
    def decompress(self, arr):
        maxbytes    = numpy.ceil(numpy.log2(max(self.posbitwidth, self.QHObitwidth_v.max()))/8)
        result      = self.use_module.zeros((len(arr), self.nchain+1), dtype="uint%i" % (8*maxbytes))
        result[:,0] = self.get_pos(arr)
        for i in range(self.nchain):
            result[:,i+1] = self.get_phonon_occ(arr, i)
        return result


    def compress_states(self, raw_basis_states):
        if raw_basis_states.shape[1] != self.nchain+1:
            raise ValueError("Shape of input states does not match chain length.")

        QHObitwidth = self.QHObitwidth_v if raw_basis_states.device == vector_device else self.QHObitwidth_w

        result      = self.use_module.zeros((len(raw_basis_states), self.totwordwidth+1), dtype=self.dtype) # add one column to allow for shape matching when assigning values to the last real column
        lastbit     = self.posbitwidth + QHObitwidth.cumsum()          # index of first bit after the block has ended
        word_i      = (lastbit - QHObitwidth) // self.wordsize         # index of first word containing the block

        if self.totwordwidth > 1:
            ud2             = "uint%i" % (2*self.wordsize)
#            conv            = self.use_module.zeros((len(raw_basis_states), 1), dtype=ud2)
            conv            = raw_basis_states[:,0].astype(ud2) << (2*self.wordsize - self.posbitwidth)
            conv            = conv[None].T
            result[:,:2]    += conv.view(self.dtype)[:,::-1]
            for i in range(self.nchain):
                conv                            = raw_basis_states[:,i+1].astype(ud2) << ((word_i[i]+2)*self.wordsize - lastbit[i])
                conv                            = conv[None].T
                result[:,word_i[i]:word_i[i]+2] += conv.view(self.dtype)[:,::-1]

        else:
            result[:,0]     += raw_basis_states[:,0].astype(self.dtype) << (self.wordsize - self.posbitwidth)
            for i in range(self.nchain):
                result[:,0]     += raw_basis_states[:,i+1].astype(self.dtype) << (self.wordsize - lastbit[i])

        # remove the last column:
        if not self.use_module.all(result[:,-1] == 0):
            raise ValueError("Fatal internal error: In the process of compressing the states, an accessory column was not left unchanged.\nPlease contact your system administrator or pray for relief.")
        result = self.use_module.ascontiguousarray(result[:,:-1])
        return result


    ###################################
    # data segmentation functions, 
    # retrieve values
    ###################################
    def get_phonon_occ(self, arr, site):
        QHObitwidth = self.QHObitwidth_v if arr.device == vector_device else self.QHObitwidth_w
        bit_offset  = self.posbitwidth + QHObitwidth[:site].sum()      # number of bits preceding the block in question (equivalently, index of first relevant bit)
        lastbit     = bit_offset + QHObitwidth[site]                   # index of first bit after the block has ended
        word_i      = bit_offset // self.wordsize                           # index of first word/byte containing the block
        num2bits    = lastbit - self.wordsize*word_i                        # wordsize + number of relevant bits contained in word/byte_i+1 (num2bits == wordsize is "zero")
        numlead     = bit_offset % self.wordsize                            # number of leading bits in word/byte_i that must be trimmed

        result      = arr[:,word_i].copy()
        if numlead != 0:
            result &= (1 << (self.wordsize - numlead)) - 1

        if num2bits < self.wordsize:
            result  >>= self.wordsize - num2bits
        elif num2bits > self.wordsize:
            result  <<= num2bits - self.wordsize
            result  += arr[:,word_i+1] >> (2*self.wordsize-num2bits)
        return result


    def get_pos(self, arr):
        lastbit     = self.posbitwidth
        if lastbit < self.wordsize:
            return arr[:,0] >> (self.wordsize-lastbit)
        elif lastbit == self.wordsize:
            return arr[:,0]
        elif lastbit > self.wordsize:
            if self.wordsize != 8:
                raise NotImplementedError("The chain length seems to exceed 2**16 = 65536, which has not been implemented yet.")
            result  = arr[:,0].astype("uint16")
            result  <<= (lastbit-8)
            result  += (arr[:,1] >> (16-lastbit))
            return result

    def get_phonon_at_exc(self, arr):
        if arr.device   == vector_device:
            return cupy_gendatseg_kernels.get_phonon_at_exc(arr, self.posbitwidth, self.QHObitwidth_v, self.wordsize)
        elif arr.device == whoami_device:
            return cupy_gendatseg_kernels.get_phonon_at_exc(arr, self.posbitwidth, self.QHObitwidth_w, self.wordsize)

    def sum_all_phonons(self, arr, omega=None):
        if type(omega) in (int, float):
            omega   = self.use_module.full(self.nchain, omega, dtype="float64")
        elif isinstance(omega, (list, tuple, numpy.ndarray)) or isinstance(omega, cupy.ndarray):
            omega   = self.use_module.asarray(omega, dtype="float64")
        elif omega is None:
            omega   = cupy.ones(self.nchain, dtype="float64")
        else:
            raise TypeError("Unrecognized type of omega.")
        if arr.device   == vector_device:
            return cupy_gendatseg_kernels.sum_all_phonons(omega, arr, self.posbitwidth, self.QHObitwidth_v, self.wordsize)
        elif arr.device == whoami_device:
            return cupy_gendatseg_kernels.sum_all_phonons(omega, arr, self.posbitwidth, self.QHObitwidth_w, self.wordsize)


    ###################################
    # data segmentation functions, 
    # manipulate values
    ###################################
    def add_n_to_site(self, arr, n, site):
        """
        Takes an array arr and adds (or subtracts) n phonons to site n (to arr itself, not a copy of it).
        This method takes care of rollovers between bytes, but is unsafe with regard to over-/underflows due to floor/ceiling hits.
        """
        if n == 0:
            return

        QHObitwidth = self.QHObitwidth_v if arr.device == vector_device else self.QHObitwidth_w
        bit_offset  = self.posbitwidth + QHObitwidth[:site].sum()      # number of bits preceding the block in question (equivalently, index of first relevant bit)
        lastbit     = bit_offset + QHObitwidth[site]                   # index of first bit after the block has ended
        word_i      = bit_offset // self.wordsize                           # index of first byte containing the block
        num2bits    = lastbit - self.wordsize*word_i                        # wordsize + number of relevant bits contained in word/byte_i+1 (num2bits == wordsize is "zero")

        if num2bits == self.wordsize:
            arr[:,word_i]   += self.dtype(n)
        elif num2bits < self.wordsize:
            arr[:,word_i]   += self.dtype(n << (self.wordsize-num2bits))

        elif num2bits > self.wordsize:
            if wordsize != 8:
                raise NotImplementedError("This method has not been implemented for wordsizes != 8 (consider using add_phonon_to_exc instead).")
            # If the data spans two bytes, add the number to a 16-bit view spanning both bytes.
            # To do that, we must create a contiguous array from arr and also reverse the order due to the CPU/GPU-level little-endian byte ordering.
            # Unfortunately, this requires creating a copy of the two columns in question.
#            arr[:,byte_i:byte_i+2]  = (self.use_module.ascontiguousarray(arr[:,byte_i:byte_i+2][:,::-1]).view("uint16") + n*2**(8-num2bits)).view("uint8")[:,::-1]
            arr[:,word_i:word_i+2]  = (self.use_module.ascontiguousarray(arr[:,word_i:word_i+2][:,::-1]).view("uint16") + self.dtype(n << (16-num2bits))).view("uint8")[:,::-1]
        return


    def add_n_to_pos(self, arr, n):
        """
        Takes an array arr and moves the particle by n sites (acts on arr itself, does not return a copy of it).
        This method takes care of rollovers between bytes, but is unsafe with regard to over-/underflows due to system boundary excursions.
        """
        if n == 0:
            return

        lastbit     = self.posbitwidth
        if lastbit < self.wordsize:
            arr[:,0]    += self.dtype(n << (self.wordsize-lastbit))
        elif lastbit == self.wordsize:
            arr[:,0]    += self.dtype(n)
        elif lastbit > self.wordsize:
            if self.wordsize != 8:
                raise NotImplementedError("The chain length seems to exceed 2**16 = 65536, which has not been implemented yet.")
            # see add_n_to_site for relevant comment on code
            arr[:,:2]   = (self.use_module.ascontiguousarray(arr[:,:2][:,::-1]).view("uint16") + self.dtype(n << (16-lastbit))).view("uint8")[:,::-1]

        return

    def add_phonon_at_exc(self, arr):
        if arr.device   == vector_device:
            cupy_gendatseg_kernels.add_phonon_at_exc(arr, self.posbitwidth, self.QHObitwidth_v, self.wordsize)
        elif arr.device == whoami_device:
            cupy_gendatseg_kernels.add_phonon_at_exc(arr, self.posbitwidth, self.QHObitwidth_w, self.wordsize)

    def rem_phonon_at_exc(self, arr):
        if arr.device   == vector_device:
            cupy_gendatseg_kernels.rem_phonon_at_exc(arr, self.posbitwidth, self.QHObitwidth_v, self.wordsize)
        elif arr.device == whoami_device:
            cupy_gendatseg_kernels.rem_phonon_at_exc(arr, self.posbitwidth, self.QHObitwidth_w, self.wordsize)

    ###################################
    # wrapper function for Hamiltonian generation
    ###################################
    def generate_mel(self, basis_states, enlarge_steps=0):
        if basis_states.shape[0] > self.maxstates:
            raise ValueError("Number of basis states exceeds maxstates.")
        if basis_states.shape[1] != self.totwordwidth:
            raise ValueError("Basis states do not match the number of bits specified at initialization.")
        for i in range(enlarge_steps):
            basis_states = self.enlarge_basis_set(basis_states)
        plus_hop_triple, hop_o_star, minus_hop_triple                       = self.generate_mel_hopping(basis_states)
        if self.debug_verb > 2:
            print("    b.1: Determined hopping indices.")
        plus_coupl_triple, coupl_o_star, minus_coupl_triple, ceiling_hits   = self.generate_mel_coupling(basis_states)
        if self.debug_verb > 2:
            print("    b.2: Determined coupling indices.")

        ### Diagonal terms shall include the new terms! :
        lenlist     = [basis_states.shape[0],]
        for item in [plus_hop_triple, minus_hop_triple, plus_coupl_triple, minus_coupl_triple]:
            if item[2] is True:
                lenlist += [item[0].shape[0],]
            else:
                lenlist += [int(item[2].sum()),]
        all_inds    = cupy.empty((sum(lenlist), basis_states.shape[1]), dtype=self.dtype)
        start       = 0
        for i, item in enumerate([(basis_states, 0, True), plus_hop_triple, minus_hop_triple, plus_coupl_triple, minus_coupl_triple]):
            if item[2] is True:
                all_inds[start:start+lenlist[i]]    = item[0]
            else:
                all_inds[start:start+lenlist[i]]    = item[0][item[2]]                
            start                               += lenlist[i]
        if start != sum(lenlist):
            raise ValueError("Fatal error that I shall not further specify because I want to annoy you.")

        new_inds        = self.cupy_unique(all_inds)

        if self.debug_verb > 2:
            print("    b.3: Determined new whoami array.")

        diag_vals   = self.eps_sys + self.eps_bath * self.sum_all_phonons(new_inds)
        if self.delta_eps != 0:
            diag_vals   += self.delta_eps * self.get_pos(new_inds)

        if self.debug_verb > 2:
            print("    Number of diag vals: %i" % diag_vals.shape[0])

        ### Now convert these indices into the dense vector indices:
        # (the diagonal elements are already given)
#        if len(new_inds) < 2**16:
#            dtype       = self.use_module.uint16
        dtype       = self.use_module.uint32
        if len(new_inds) >= 2**32:
            raise ValueError("The indexing array is too long to be stored in a 32-bit number format.")
#        elif len(new_inds) < 2**64:
#            dtype       = self.use_module.uint64

        t0 = time.time()
        basis_lookup    = self.searchsorted(new_inds, basis_states)
        self.print_searchsorted_timing(time.time() - t0, new_inds.shape[0], basis_states.shape[0])

        hopping_vals, (hop_to, hop_from)    = self.fill_in_param_arrays(new_inds, basis_lookup, lenlist[1], lenlist[2], plus_hop_triple, minus_hop_triple, hop_o_star, dtype)
        if self.debug_verb > mem_info_level:
            print("Near-maximal memory usage on whoami_device, est. 1: %1.1f MiB" % (cupy.get_default_memory_pool().used_bytes()/1024**2))
        del plus_hop_triple, minus_hop_triple

        coupl_vals, (coupl_to, coupl_from)  = self.fill_in_param_arrays(new_inds, basis_lookup, lenlist[3], lenlist[4], plus_coupl_triple, minus_coupl_triple, coupl_o_star, dtype)
        if self.debug_verb > mem_info_level:
            print("Near-maximal memory usage on whoami_device, est. 2: %1.1f MiB" % (cupy.get_default_memory_pool().used_bytes()/1024**2))
        del plus_coupl_triple, minus_coupl_triple


        if self.debug_verb > 2:
            print("    b.4: Converted Hamiltonian matrix elements into dense coo format.")
        if self.debug_verb > 3:
            t0              = time.time()
            unique, counts  = cupy.unique(self.get_pos(new_inds), return_counts=True)
            t1              = time.time()
            print("      Number of basis states with exciton at each site: " + str(dict(zip(unique.tolist(), counts.tolist()))) + " (this calculation took %f ms)" % ((t1 - t0)*1000))
#            print("      Number of positions where coupling is not diagonal: %i." % (new_inds[coupl_from][:,0] != new_inds[coupl_to][:,0]).sum())

        if (self.get_pos(new_inds[coupl_from]) != self.get_pos(new_inds[coupl_to])).sum() != 0:
            raise RuntimeError("Coupling matrix is not diagonal in the exciton Hilbert space!")
        return (diag_vals, new_inds), (hopping_vals, (hop_to, hop_from)), (coupl_vals, (coupl_to, coupl_from)), ceiling_hits



    ###################################
    # simple but annoying array-filling function
    ###################################
    def fill_in_param_arrays(self, new_inds, basis_lookup, seg1, seg2, plus_triple, minus_triple, o_star, dtype):
        # order: plus, plus c.c., minus, minus c.c.

        ### map_from indices:
        inds_from                   = self.use_module.empty(2 * (seg1 + seg2), dtype=dtype)

        if plus_triple[2] is True:
            # plus part:
            inds_from[:seg1]                = basis_lookup
            # plus, c.c.:
            t0 = time.time()
            findme                          = plus_triple[0]
        else:
            # plus part:
            inds_from[:seg1]                = basis_lookup[plus_triple[2]]
            # plus, c.c.:
            t0 = time.time()
            findme                          = plus_triple[0][plus_triple[2]]

        inds_from[seg1:2*seg1]          = self.searchsorted(new_inds, findme)
        self.print_searchsorted_timing(time.time() - t0, new_inds.shape[0], findme.shape[0])

        prev    = 2*seg1
        if minus_triple[2] is True:
            # minus:
            inds_from[prev:prev+seg2]       = basis_lookup[o_star]
            # minus, c.c.:
            t0 = time.time()
            findme                          = minus_triple[0]
        else:
            # minus:
            inds_from[prev:prev+seg2]       = basis_lookup[o_star][minus_triple[2]]
            # minus, c.c.:
            t0 = time.time()
            findme                          = minus_triple[0][minus_triple[2]]

        inds_from[prev+seg2:]           = self.searchsorted(new_inds, findme)
        self.print_searchsorted_timing(time.time() - t0, new_inds.shape[0], findme.shape[0])

        ### map_to indices:
        inds_to                         = self.use_module.empty(2 * (seg1 + seg2), dtype=dtype)
        # plus part:
        inds_to[:seg1]                  = inds_from[seg1:2*seg1]
        # plus, c.c.:
        inds_to[seg1:2*seg1]            = inds_from[:seg1]
        # minus part:
        inds_to[prev:prev+seg2]         = inds_from[prev+seg2:]
        # minus, c.c.:
        inds_to[prev+seg2:]             = inds_from[prev:prev+seg2]

        vals                            = self.use_module.empty(len(inds_to), dtype=type(self.t_sys))
        if plus_triple[2] is True or type(plus_triple[1]) not in (cupy.ndarray, numpy.ndarray):
            vals[:seg1]                 = plus_triple[1]
            vals[seg1:2*seg1]           = numpy.conj(plus_triple[1])
        else:
            vals[:seg1]                 = plus_triple[1][plus_triple[2]]
            vals[seg1:2*seg1]           = self.use_module.conj(plus_triple[1])[plus_triple[2]]
        if minus_triple[2] is True or type(minus_triple[1]) not in (cupy.ndarray, numpy.ndarray):
            vals[prev:prev+seg2]        = minus_triple[1]
            vals[prev+seg2:]            = numpy.conj(minus_triple[1])
        else:
            vals[prev:prev+seg2]        = minus_triple[1][minus_triple[2]]
            vals[prev+seg2:]            = self.use_module.conj(minus_triple[1])[minus_triple[2]]

        return vals, (inds_to, inds_from)



    ###################################
    # allow for two-step adaptation by adding neighboring basis states to the existing basis set
    ###################################
    def enlarge_basis_set(self, basis_states):
        max_HO_dims     = self.max_HO_dims_v if basis_states.device == vector_device else self.max_HO_dims_w

       # start with hopping
        hop_plus        = basis_states.copy()
        self.add_n_to_pos(hop_plus, 1)
        hop_minus       = basis_states.copy()
        self.add_n_to_pos(hop_minus, -1)
        excpos          = self.get_pos(basis_states) 
        if self.periodic:
            raise NotImplementedError("Periodic hopping not yet implemented.")
        else:
            hop_plus    = hop_plus[excpos < self.nchain-1]
            hop_minus   = hop_minus[excpos > 0]

        # now coupling:
        phonon_occs     = self.get_phonon_at_exc(basis_states)

        coupl_plus      = (basis_states[phonon_occs < max_HO_dims[excpos] - 1]).copy()
        self.add_phonon_at_exc(coupl_plus)

        coupl_minus     = (basis_states[phonon_occs > 0]).copy()
        self.rem_phonon_at_exc(coupl_minus)

        lenlist         = [len(i) for i in [basis_states, hop_plus, hop_minus, coupl_plus, coupl_minus]]
        total           = self.use_module.empty((sum(lenlist), basis_states.shape[1]), dtype=self.dtype)
        for i, thing in enumerate([basis_states, hop_plus, hop_minus, coupl_plus, coupl_minus]):
            total[sum(lenlist[:i]):sum(lenlist[:i+1])]  = thing
            del thing

        return self.cupy_unique(total)



    ###################################
    # generate hopping matrix elements
    ###################################
    def generate_mel_hopping(self, basis_states):   # basis_states does not have to be sorted for this to work
        excpos  = self.get_pos(basis_states)

        # generate the plus side of the hopping (= to the right):
        plus_inds               = basis_states.copy()
        self.add_n_to_pos(plus_inds, 1)
        plus_vals               = self.t_sys
        if self.periodic:
            raise NotImplementedError("Periodic hopping not yet implemented.")
#            plus_inds[:,0]          = plus_inds[:,0] % self.nchain
#            plus_mask               = self.use_module.ones_like(plus_vals, dtype=bool)
#            plus_mask               = True
        else:         ### for OBC, remove boundary excursions:
            plus_mask               = excpos < self.nchain-1

        # generate a mask to remove values that occur in both the input and plus_inds:
        t0 = time.time()
        if plus_mask is True:
            sortplus    = plus_inds[cupy.lexsort(plus_inds.T[::-1])]
        else:
            mpluind     = plus_inds[plus_mask]
            sortplus    = mpluind[cupy.lexsort(mpluind.T[::-1])]
            del mpluind
        o_star_mask = cupy.any(sortplus[self.searchsorted(sortplus, basis_states, allow_escapes=True)] != basis_states, axis=1)
        if self.debug_verb > searchsorted_timing_level:
            t1 = time.time()
            print("This application of (sorting +) searchsorted (+ masking) took %f ms (arg sizes %i, %i)." % ((t1-t0)*1000, sortplus.shape[0], basis_states.shape[0]))


        # generate the minus side of the coupling (= to the left):
        minus_inds              = basis_states[o_star_mask].copy()
        self.add_n_to_pos(minus_inds, -1)
        minus_vals              = numpy.conj(self.t_sys)
        if self.periodic:
            raise NotImplementedError("Periodic hopping not yet implemented.")
#            minus_mask              = True
        else:         ### for OBC, remove boundary excursions:
            minus_mask              = excpos[o_star_mask] > 0 # remove the ones that mapped to the last site

#        return self.fill_into_results(basis_states, plus_inds, plus_vals, plus_mask, basis_states[o_star_mask], minus_inds, minus_vals, minus_mask)
        return (plus_inds, plus_vals, plus_mask), o_star_mask, (minus_inds, minus_vals, minus_mask)


    ###################################
    # generate vibronic coupling matrix elements
    ###################################
    def generate_mel_coupling(self, basis_states):
        max_HO_dims     = self.max_HO_dims_v if basis_states.device == vector_device else self.max_HO_dims_w
        phonon_occs     = self.get_phonon_at_exc(basis_states)
        excpos          = self.get_pos(basis_states)

        # generate the plus side of the coupling:
        plus_inds               = basis_states.copy()
        self.add_phonon_at_exc(plus_inds)
        plus_vals               = self.coupling_g * self.use_module.sqrt(phonon_occs+1, dtype=cupy.float64)
        # add mask for ceiling hits:
        plus_mask               = phonon_occs < max_HO_dims[excpos] - 1
        ceiling_hits            = (len(basis_states) - plus_mask.sum())

        # generate a mask to remove values that occur in both the input and plus_inds:
        t0 = time.time()
        mpluind     = plus_inds[plus_mask]
        sortplus    = mpluind[cupy.lexsort(mpluind.T[::-1])]
        del mpluind
        o_star_mask = cupy.any(sortplus[self.searchsorted(sortplus, basis_states, allow_escapes=True)] != basis_states, axis=1)
        if self.debug_verb > searchsorted_timing_level:
            t1 = time.time()
            print("This application of (sorting +) searchsorted (+ masking) took %f ms (arg sizes %i, %i)." % ((t1-t0)*1000, sortplus.shape[0], basis_states.shape[0]))

        # generate the minus side of the coupling:
        minus_inds                              = basis_states[o_star_mask].copy()
        self.rem_phonon_at_exc(minus_inds)
        minus_vals                              = numpy.conj(self.coupling_g) * self.use_module.sqrt(phonon_occs[o_star_mask], dtype=cupy.float64)
        minus_mask                              = phonon_occs[o_star_mask] > 0

        return (plus_inds, plus_vals, plus_mask), o_star_mask, (minus_inds, minus_vals, minus_mask), ceiling_hits


    ###################################
    # replacement for numpy.unique with option axis=0
    ###################################
    def cupy_unique(self, array):
        if len(array.shape) != 2:
            raise ValueError("Input array must be 2D.")
        sortarr     = array[cupy.lexsort(array.T[::-1])]
        mask        = cupy.empty(array.shape[0], dtype=cupy.bool_)
        mask[0]     = True
        mask[1:]    = cupy.any(sortarr[1:] != sortarr[:-1], axis=1)
        return sortarr[mask]

    ###################################
    # a function to test the performance of a cupy-based replacement for numpy.isin with view
    # (in case you forgot the result: for arrays of size (1e7, 10), the cupy method is 100 times faster)
    ###################################
    def test_isin(self, num=int(1e7)):
        a       = numpy.ones((num, self.nchain + 1), dtype=numpy.uint8)
        a[:,0]  = (numpy.random.random(num) * self.nchain).astype(numpy.uint8)
        a[:,1:] = (numpy.random.random((num, self.nchain)) * 256).astype(numpy.uint8)
        a       = numpy.unique(a, axis=0)

        b       = numpy.ones((num//2, self.nchain + 1), dtype=numpy.uint8)
        b[:,0]  = (numpy.random.random(num//2) * self.nchain).astype(numpy.uint8)
        b[:,1:] = (numpy.random.random((num//2, self.nchain)) * 256).astype(numpy.uint8)
        b       = numpy.unique(b, axis=0)

        form                = [('f%i' % i, numpy.uint8) for i in range(a.shape[1])]

        t0 = time.time()
        nres = numpy.isin(b.view(form), a.view(form), assume_unique=True).flatten()       # assign common values to positions
        t1 = time.time()
        print("Runtime of numpy.isin with unconventional view: %f seconds (arg sizes %i, %i)." % (t1-t0, b.shape[0], a.shape[0]))

        ca      = cupy.array(a)
        cb      = cupy.array(b)
        t0 = time.time()
        cres = cupy.all(ca[self.searchsorted(ca, cb, allow_escapes=True)] == cb, axis=1)       # this only works if ca is sorted
        t1 = time.time()
        print("Runtime of cupy alternative: %f seconds." % (t1-t0))
        print(numpy.all(nres == cupy.asnumpy(cres)))


###################################################################################################################################################################################

def write_params(fname, obj, itemstr):
    try:
        value   = getattr(getattr(obj, itemstr), "__name__")
    except AttributeError:
        value   = getattr(obj, itemstr)
    fname.write(itemstr + " = " + str(value) + '\n')

class time_evolution:
    def __init__(self, HamObj, dirname=None, verbose=True, m_star=25, debug_verb=0, shuffle_seed=0, U_weighting_method="coherence"):
        if verbose:
            print("Initializing time_evolution object...")
        self.HamObj     = HamObj
        self.dirname    = dirname
        self.verbose    = verbose
        self.m_star     = m_star                    # this is the max iteration of the Taylor approximation
        self.use_module     = self.HamObj.use_module
        self.debug_verb         = debug_verb
        self.HamObj.debug_verb  = self.debug_verb   # time_evolution debug_verb overrides HamObj debug_verb
        self.shuffle_seed       = shuffle_seed      # when determining the new Hilbert space, shuffle equal values to avoid bias using this value as the initial seed
        if shuffle_seed is not None:
            self.use_module.random.seed(shuffle_seed)
        self.first_order_U_importance   = getattr(self, "first_order_U_importance_" + U_weighting_method)

        self.hop_board          = None              # this will be lazily constructed if calculate_hopping is called

        self.params_file        = os.path.join(self.dirname, "HamObj_params_run" + str(time.time()) + ".log")
        with open(self.params_file, "w") as HamObj_params_file:
            HamObj_params_file.write("### HamObj parameters:\n")
            for itemstr in ("nchain", "eps_sys", "t_sys", "eps_bath", "coupling_g", "delta_eps", "max_HO_dims_v", "complex_type", "periodic", "use_module", "maxstates", "wordsize"):
                write_params(HamObj_params_file, self.HamObj, itemstr)
            HamObj_params_file.write("### time_evolution parameters:\n")
            for itemstr in ("m_star", "shuffle_seed", "first_order_U_importance"):
                write_params(HamObj_params_file, self, itemstr)
        if verbose:
            print("Finished initialization of time_evolution object!\n")


    ###################################
    # generate initial basis set
    ###################################
    def create_uniform_truncated_basis(self, truncate_d):
        function_start_time = time.time()
        if self.verbose:
            print("Creating initial basis set...", end=' ')
            sys.stdout.flush()        
        if truncate_d > self.HamObj.max_HO_dims_v.min():
            raise ValueError("Specified basis truncation value exceeds at least one of the max_HO_dims.")
        if self.HamObj.nchain * truncate_d**self.HamObj.nchain > self.HamObj.maxstates:
            raise ValueError("Number of states that would result from this value of truncate_d exceeds maxstates!")
        # print current basis creation to file
        header = format_function_args(inspect.currentframe(), function_start_time)
        with open(self.params_file, 'a') as params_file:
            params_file.write(header)
        raw_whoami      = self.use_module.asarray(cartesian_product(numpy.arange(self.HamObj.nchain, dtype=numpy.uint8), *(numpy.arange(truncate_d, dtype=numpy.uint8) * numpy.ones(self.HamObj.nchain, dtype=numpy.uint8)[None].T)))
        self.whoami     = self.HamObj.compress_states(raw_whoami)
        self.numstates  = self.whoami.shape[0]
        if self.verbose:
            print("Done!")

    def create_nonuniform_basis(self, truncate_d_list, lowest_d_list=None, minpos=0, maxpos=None):
        """
        truncate_d_list:    Number of basis states at given phonon site
        lowest_d_list:      Lowest basis state to construct (defaults to 0 everywhere)
            The highest n at each site is then lowest_d + truncate_d
        """
        function_start_time = time.time()
        if self.verbose:
            print("Creating initial basis set...", end=' ')
            sys.stdout.flush()
        if maxpos is None:
            maxpos = self.HamObj.nchain
        if lowest_d_list is None:
            lowest_d_list = numpy.zeros_like(truncate_d_list)
        if len(truncate_d_list) != self.HamObj.nchain or len(lowest_d_list) != self.HamObj.nchain:
            raise ValueError("Incorrect chain length.")
        if any([lowest_d_list[i] + truncate_d_list[i] > self.HamObj.max_HO_dims_v[i] for i in range(self.HamObj.nchain)]):
            raise ValueError("Specified basis truncation value exceeds at least one of the max_HO_dims.")
        if numpy.product(truncate_d_list) * (maxpos-minpos) > self.HamObj.maxstates:
            raise ValueError("Number of states that would result from this value of truncate_d exceeds maxstates!")
        if minpos < 0 or maxpos > self.HamObj.nchain or minpos >= maxpos:
            raise ValueError("Invalid minpos or maxpos.")

        # print current basis creation to file
        header = format_function_args(inspect.currentframe(), function_start_time)
        with open(self.params_file, 'a') as params_file:
            params_file.write(header)

        nontrivialdim       = numpy.array(truncate_d_list) > 1
        dimlist             = [numpy.arange(lowest_d_list[i], lowest_d_list[i] + truncate_d_list[i], dtype=numpy.uint8) for i in range(self.HamObj.nchain) if nontrivialdim[i]]
        nopad_whoami        = self.use_module.asarray(cartesian_product(numpy.arange(minpos, maxpos, dtype=numpy.uint8), *dimlist))

        raw_whoami          = self.use_module.zeros((nopad_whoami.shape[0], self.HamObj.nchain+1), dtype=numpy.uint8)
        raw_whoami[:,numpy.nonzero(nontrivialdim)[0]+1] = nopad_whoami[:,1:]
        raw_whoami[:,0]     = nopad_whoami[:,0]
        self.whoami         = self.HamObj.compress_states(raw_whoami)
        self.numstates      = self.whoami.shape[0]
        if self.verbose:
            print("Done!")


    def create_moving_gaussian_OBC_basis(self, super_mu, super_sigma, sigma, maxval):
        """
        Create an initial basis consisting of gaussian distributions of phonon occupations around the exciton position.
        super_mu selects the initial exciton position, and super_sigma then sets how much bias is given to the initial position (where numpy.inf corresponds to zero bias and 0 to infinite bias).
        sigma sets how sharp the individual distributions are and maxval sets the highest phonon occupation in total (note that the true max occupation will be slightly higher than maxval, however).
        """
        function_start_time = time.time()
        if self.verbose:
            print("Creating initial basis set...", end=' ')
            sys.stdout.flush()
#        if len(truncate_d_list) != self.HamObj.nchain:
#            raise ValueError("Incorrect chain length.")
#        if numpy.product(truncate_d_list) * self.HamObj.nchain > self.HamObj.maxstates:
#            raise ValueError("Number of states that would result from this value of truncate_d exceeds maxstates!")
        if maxval >= self.HamObj.max_HO_dims_v.max():
            raise ValueError("Specified basis truncation value exceeds the maximal max_HO_dims.")
        # print current basis creation to file
        header = format_function_args(inspect.currentframe(), function_start_time)
        with open(self.params_file, 'a') as params_file:
            params_file.write(header)
        def gauss(x, mu, sigma):
            return numpy.exp(-0.5 * ((x-mu)/sigma)**2)
        narr                = numpy.arange(self.HamObj.nchain)
        super_gauss         = 1 + maxval * gauss(narr, super_mu, super_sigma)
        basis_list          = []
        numstates           = 0
        for i in range(self.HamObj.nchain):
            this_d_gauss    = gauss(narr, i, sigma)
            basis_list      += [cartesian_product(*[numpy.arange(super_gauss[i] * this_d_gauss[j], dtype=numpy.uint8) for j in range(self.HamObj.nchain)])]
#            print()
            numstates       += len(basis_list[-1])
            if numstates > self.HamObj.maxstates:
                raise ValueError("The parameters specified for the initial basis set generate a basis set whose size exceeds maxstates.")

        self.numstates      = numstates
        whoami              = self.use_module.empty((self.numstates, self.HamObj.nchain + 1), dtype=cupy.uint8)
        c = 0
        for i, basis_part in enumerate(basis_list):
            whoami[c:c+len(basis_part),1:]      = self.use_module.asarray(basis_part)
            whoami[c:c+len(basis_part),0]       = i
            c += len(basis_part)

        if self.use_module.any(whoami.max(axis=0) >= self.HamObj.max_HO_dims_v):
            raise ValueError("max_HO_dims exceeded!")
        self.whoami = self.HamObj.compress_states(whoami)

        if self.verbose:
            print("Done!")

    def load_basis_from_file(self, loadfile):
        function_start_time = time.time()
        if self.verbose:
            print("Loading basis set from file...", end=' ')
            sys.stdout.flush()
        self.whoami     = self.use_module.load(loadfile)
        self.numstates  = self.whoami.shape[0]
        if self.whoami.shape[1] != self.HamObj.totwordsize:
            raise ValueError("Loaded basis set does not match total number of bits as given by HamObj construction.")
        if self.whoami.dtype != self.HamObj.dtype:
            raise ValueError("Loaded basis set does not match the wordsize as given by HamObj construction.")

        # print current basis creation to file
        header = format_function_args(inspect.currentframe(), function_start_time)
        with open(self.params_file, 'a') as params_file:
            params_file.write(header)

        if self.verbose:
            print("Done!")

    ###################################
    # take a small initial basis set and
    # optimize via Hamiltonian enlargement:
    ###################################
    def grow_optimal_basis(self, n_max=numpy.inf, fillfac=1.0):
        """
        n_max:      The maximum number of enlargement iterations to perform
        fillfac:    The proportion of maxstates to use up
        """
        if self.verbose:
            print("Growing initial basis set...", end=' ')
            sys.stdout.flush()

        # print current basis creation to file
        header = format_function_args(inspect.currentframe())
        with open(self.params_file, 'a') as params_file:
            params_file.write(header)

        initsize            = len(self.whoami)
        whoami              = self.whoami
        n                   = 0
        while len(whoami) <= self.HamObj.maxstates * fillfac:
            previous_whoami = whoami
            n               += 1
            if n > n_max:
                break
            whoami          = self.HamObj.enlarge_basis_set(previous_whoami)

        # use previous_whoami, since the last was the one that triggered the break condition
        self.whoami         = previous_whoami
        self.numstates      = self.whoami.shape[0]
        if self.verbose:
            print("Done! Performed %i enlargements (grew from %i to %i states)." % (n, initsize, len(self.whoami)))

    ###################################
    # generate initial vector (requires an existing basis set)
    ###################################
    def create_initial_vector(self, vector_coeffs, vector_coo, auto_normalize=True):
        """
        Takes vector parameters in the format coeffs, (sys, bath) and, by looking up the positions in whoami, creates the state vector.
        """
        if self.verbose:
            print("Creating the initial state vector...", end=' ')
            sys.stdout.flush()

        vector_coo  = cupy.asnumpy(vector_coo)
        if vector_coo.ndim != 2:
            raise ValueError("The coo list must be 2D, even if it contains only the entry for one basis state.")
        if numpy.unique(vector_coo, axis=0).shape != vector_coo.shape:
            raise ValueError("There are duplicates in the specified vector coordinates.")
        if vector_coo.shape[0] != len(vector_coeffs):
            raise ValueError("The number of coefficients does not match the number of coordinates.")
        if auto_normalize:
            vector_coeffs /= numpy.linalg.norm(vector_coeffs)

        vector_coo      = self.HamObj.compress_states(cupy.asarray(vector_coo))

        self.vector     = self.use_module.zeros(len(self.whoami), dtype=self.HamObj.complex_type)
        for i, coo in enumerate(vector_coo):
            index               = self.use_module.all(self.whoami == self.use_module.asarray(coo), axis=1).nonzero()[0][0]
            self.vector[index]  = vector_coeffs[i]

        if self.verbose:
            print("Done!")

    ###################################
    # generate tensor product phonon state (requires an existing basis set)
    ###################################
    def create_tensor_init_state(self, bstates, coeffs, excsite, sites="all"):
        bstates = numpy.asarray(bstates)
        coeffs  = numpy.asarray(coeffs, dtype=complex)
        if sites != "all" and len(sites) > self.HamObj.nchain:
            raise ValueError("Invalid number of sites.")
        if len(bstates) != len(coeffs):
            raise ValueError("Number of coefficients does not match number of states.")

        if sites == "all":
            sitemask    = numpy.ones(self.HamObj.nchain, dtype=bool)
        else:
            sitemask    = numpy.array([i in sites for i in range(self.HamObj.nchain)])
        coeffs  = [coeffs] * sitemask.sum()
        mat     = cartesian_product(*coeffs)
        vector  = mat.prod(axis=1)
        if abs(1 - numpy.linalg.norm(vector)) > 1e-12:
            print(numpy.linalg.norm(vector))
            raise ValueError("State is not normalized, aborting.")

        bmask       = numpy.append([False], sitemask)
        basis       = numpy.zeros((len(vector), self.HamObj.nchain+1))
        basis[:,bmask]  = cartesian_product(*numpy.tile(bstates, (sum(sitemask), 1)))
        basis[:,0]      = excsite

        self.create_initial_vector(vector, basis, auto_normalize=True)

    ###################################
    # generate sparse matrices
    ###################################
    def create_matrices(self, diag_params, hopping_params, coupl_params):
        self.sparse_hopping     = self.use_module.sparse.coo_matrix((hopping_params[0], hopping_params[1]), shape=(self.numstates, self.numstates)).tocsr()
        self.sparse_coupling    = self.use_module.sparse.coo_matrix((coupl_params[0], coupl_params[1]), shape=(self.numstates, self.numstates)).tocsr()
        self.diag_vals          = diag_params


    ###################################
    # perform time evolution step
    ###################################
    def simple_taylor(self, delta_t, diagnostics=False, explosion_cutoff=1e5):
        n = self.numstates
        u_d = 2**-53
        tol = u_d
        B = self.vector
        if self.debug_verb > simple_taylor_level:
            print("Check 0.")
#        print(self.whoami[self.diag_vals.argmax()])
        if self.debug_verb > 3:
            print("  Norm prior to evolution: %f." % self.use_module.linalg.norm(B))

        converged   = False             # added
        final_m     = 0                 # added
        F = B.copy()
        if self.debug_verb > simple_taylor_level:
            print("Check 1.")
        c1 = cupy_expm_multiply.cupy_exact_inf_norm(B)
        if self.debug_verb > simple_taylor_level:
            print("Check 2.")
        for j in range(self.m_star):
            if self.debug_verb > 4:
                print("Norm of F: %f." % self.use_module.linalg.norm(F))
            final_m     += 1            # added
            coeff       = -1j * delta_t / float(j+1)
            if self.debug_verb > simple_taylor_level:
                print("Check 3.")
            B_old       = B.copy()
            B           = self.sparse_hopping.dot(B_old)
            B           += self.sparse_coupling.dot(B_old)
            B           += self.diag_vals * B_old
            B           *= coeff
            del B_old
            if self.debug_verb > simple_taylor_level:
                print("Check 4.")
            c2          = cupy_expm_multiply.cupy_exact_inf_norm(B)
            if self.debug_verb > simple_taylor_level:
                print("Check 5.")
            F           += B
            c1_plus_c2  = c1 + c2
            if c1_plus_c2 <= tol * cupy_expm_multiply.cupy_exact_inf_norm(F):
                converged = True    # added
                break
            if self.debug_verb > simple_taylor_level:
                print("Check 6.")
            c1 = c2
        if self.debug_verb > simple_taylor_level:
            print("Check 7.")
        if self.debug_verb > 3:
            print("  Norm after evolution: %f." % self.use_module.linalg.norm(F))
#        B = F
        del B
        if self.use_module.linalg.norm(F) > explosion_cutoff:
            raise RuntimeError("Norm has exceeded preset explosion cutoff value (%f)!" % explosion_cutoff)
        if diagnostics:                 # added
            return F, converged, final_m, c1_plus_c2, c1_plus_c2/cupy_expm_multiply.cupy_exact_inf_norm(F)
        else:
            return F


    ###################################
    # system reduced dm from a given vector
    ###################################
    def calculate_reduced_dm(self, vector, mindiff=32):
        raise NotImplementedError("The reduced density matrix calculation has not yet been converted to the generalized data segmentation.")
        upper_tri   = self.use_module.zeros((self.HamObj.nchain, self.HamObj.nchain), dtype=self.HamObj.complex_type)
#        cupy.save("last_whoami", self.whoami)
        # use the fact that whoami is lexicographically sorted to determine the break points between the different indices:
        indli       = cupy_search.find_changes_local_single(self.whoami[:,0]).tolist() + [len(self.whoami),]
        lci         = 0
        for left_i in range(self.HamObj.nchain):
            if lci >= len(indli) - 1:
                break
            if self.whoami[indli[lci]][0] != left_i:    # if there is no basis state corresponding to the requested index, skip the iteration
                continue
            left_vec    = vector[indli[lci]:indli[lci+1]]
            left_side   = self.whoami[:,1:][indli[lci]:indli[lci+1]]
            lci         += 1
            rci         = lci   # start testing the position left_index + 1 as the first right_index
            for right_i in range(left_i, self.HamObj.nchain):
                if left_i == right_i:
                    upper_tri[left_i, right_i] = cupy.vdot(left_vec, left_vec)
                elif rci >= len(indli) - 1:
                    break
#                print(left_i, right_i)
                else:
                    if self.whoami[indli[rci]][0] != right_i:    # if there is no basis state corresponding to the requested index, skip the iteration
                        print("(%i, %i) not found. rci, location: %i, %i" % (left_i, right_i, rci, self.whoami[indli[rci]][0]))
                        continue
                    right_vec   = vector[indli[rci]:indli[rci+1]]
                    right_side  = self.whoami[:,1:][indli[rci]:indli[rci+1]]
                    rci         += 1
                    # left_vec are the coefficients matching the left_i
                    # searchsorted(right_side, left_side) returns the indexes of the left_i whoami as they occur in the right_i whoami if they occur, else some value between 0 and 2**32 - 1 depending on certain criteria 
                    # right_side[searchsorted(right_side, left_side)] yields a matching (relative to left_side) value from right_side or some arbitrary value from right_side if there is no matching value
                    # all(right_side[searchsorted(...)] == left_side, axis=1) is a mask that is true only where the values of left_side occur exactly in right_side
                    t0 = time.time()
                    left_search     = cupy_search.searchsorted_multidim_list(left_side, right_side, allow_escapes=True, linear_only=False, mindiff=mindiff)
#                    cupy.cuda.Stream.null.synchronize()
                    if self.debug_verb > searchsorted_multidim_list_timing_level:
                        t1 = time.time()
                        print("This application of searchsorted_multidim_list took %f ms (arg sizes %i, %i)." % ((t1-t0)*1000, left_side.shape[0], right_side.shape[0]))

#                    mask    = left_search < len(left_vec)
#                    upper_tri[left_i, right_i] = ((left_vec[left_search[mask]] * self.use_module.conj(right_vec[mask]))[cupy.all(left_side[left_search[mask]] == right_side[mask], axis=1)]).sum()
                    upper_tri[left_i, right_i] = ((left_vec[left_search] * self.use_module.conj(right_vec))[cupy.all(left_side[left_search] == right_side, axis=1)]).sum()
                    del left_search, right_side
                    mempool.free_all_blocks()

#                    upper_tri[left_i, right_i] = (vector[left_inds][self.use_module.isin(self.whoami[:,1][left_inds], self.whoami[:,1][right_inds], assume_unique=True)] *
#                            self.use_module.conj(vector[right_inds][self.use_module.isin(self.whoami[:,1][right_inds], self.whoami[:,1][left_inds], assume_unique=True)])).sum()
#                upper_tri[left_i, right_i] = (vector[self.whoami[:,0] == left_i] * self.use_module.conj(vector[self.whoami[:,0] == right_i])).sum()
        reduced_dm  = upper_tri + self.use_module.conj(self.use_module.triu(upper_tri, 1).T)
#        cupy.save("sec_to_last_whoami", self.whoami)
        return reduced_dm

    ###################################
    # calculate a partitioned sum, taking into consideration the length of the partitions:
    ###################################
    def calculate_partitioned_sum(self, sumvals, part_constant=5):
        if self.partition_lens.max() < part_constant * self.numstates/self.HamObj.nchain:      # to conserve RAM, only do this if the max partition length is less than part_constant times an equal partitioning:
            return cupy_search.jagged_to_regular(sumvals, self.partition_lens).sum(axis=1)
        else:
            c = 0
            result  = self.use_module.empty(self.HamObj.nchain, dtype=sumvals.dtype)
            for i in range(self.HamObj.nchain):
                result[i]   = sumvals[c:c+self.partition_lens[i]].sum()
                c += self.partition_lens[i]
            return result

    ###################################
    # system population density from a given vector
    ###################################
    def calculate_sys_n_b(self, vector):
        return self.calculate_partitioned_sum(abs(vector)**2)

    ###################################
    # compute avg. HO occs of a given vector
    ###################################
    def calculate_bath_n_b(self, vector):
        # old version, slow:
#        result  = self.use_module.zeros(self.HamObj.nchain)
#        weights = self.use_module.abs(vector)**2
#        for i in range(self.HamObj.nchain):
#            result[i]   = self.HamObj.get_phonon_occ(self.whoami, i).dot(weights)
#        return result
        weights = cupy.abs(vector)**2
        return cupy_gendatseg_kernels.calculate_bath_n_b(weights, cupy.asarray(self.whoami), self.HamObj.posbitwidth, self.HamObj.QHObitwidth_v, self.HamObj.wordsize)

    ###################################
    # compute avg. hopping interaction
    ###################################
    def calculate_hopping(self, vector):
        """
        As long as we don't have an even number of lattice sites with periodic boundary conditions, there is a nice trick
        to calculate the hopping energies using the position projector and the total hopping operator.
        The trick is: hop_i = (... + V_{i-2} - V_{i-1}^\dag) + V_i + (V_{i+1} - V_{i+1}^\dag + V_{i+3} - V_{i+4}^\dag + ...),
        where V_i = <P_i T>, with T being the total hopping operator.
        """
        if self.HamObj.nchain % 2 == 0 and self.HamObj.periodic:
            raise NotImplementedError("This doesn't work for even chain lengths with periodic boundary conditions.")
        allvals = vector.conj() * self.sparse_hopping.dot(vector)

        V_i = self.calculate_partitioned_sum(allvals)

        if self.hop_board is None:
            L               = self.HamObj.nchain
            checkerboard    = self.use_module.array([[1,0] * ((L+1)//2), [0,1] * ((L+1)//2)] * ((L+1)//2))[:L,:L]
            self.hop_board  = self.use_module.triu(checkerboard, 1) + self.use_module.triu(1 - checkerboard).T

        result  = ((1 - self.hop_board) * V_i - self.hop_board * V_i.conj()).sum(axis=1)

        if self.HamObj.periodic:
            return result
        else:
            assert abs(result[-1]) < 1e-12
            return result[:-1]

    ###################################
    # compute avg. coupling interaction
    ###################################
    def calculate_coupling(self, vector):
        allvals     = vector.conj() * self.sparse_coupling.dot(vector)
        return self.calculate_partitioned_sum(allvals)

    ###################################
    # compute total energy of a state
    ###################################
    def calculate_total_energy(self, vector):
        return self.use_module.vdot(vector, self.sparse_hopping.dot(vector) + self.sparse_coupling.dot(vector) + self.diag_vals * vector)

    ###################################
    # perform entire time evolution
    ###################################
    def generate_timeline(self, t_array=numpy.arange(0, 50.25, 0.25), observables=["n_b", "H", "hopping"], save_every=None, save_first=False, save_last=False, garbage_tol=-1, dm_mindiff=32, use_U_weight_delta_t=0.0, enlarge_steps=0, use_two_streams=False):
        """
        This is the master time evolution function that evolves the system along the time-points given in t_array and calculates and saves to file the requested variables along the way.
        In order for this to work, the HamObj object must be initialized (which is a required argument for the initialization of the time_evolution object),
        and there must be an initial basis (see the initial basis set functions in the time_evolution object) and an initial vector (via te.create_initial_vector(args) or by loading from file).

        Arguments of this function:
        t_array:                float array, the initial state is assumed to correspond to the first time value in t_array. Then evolve step-by-step until the last value in t_array is reached.
        observables:            list of str, observables to compute and save. Must be a subset of the following: ["H", "hopping", "coupling", "n_b", "reduced_dm", "diagnostics"]
            Note: "diagnostics" saves non-observable but useful information about the health of the simulation, etc.
        save_every:             int, will save the wavefunction and basis set at every n-th timestep. Default is None, which disables saving (but is overridden by save_first and save_last).
        save_first:             bool, will save the initial wavefunction and basis set if True.
        save_last:              bool, will save the final wavefunction and basis set at the end of the time evolution if True.
        garbage_tol:            float, determines what value to use as a garbage cutoff when determining the Hilbert subspace evolution. Negative values disable the garbage function and keep all basis states.
            Note that using this function is generally not advisable if the initial basis set is well chosen (i.e. you should probably use a negative value)!
        dm_mindiff:             int, tells the cupy_search algorithm when to switch from a binary to a linear search (set to something between 10 and 100 for typical use).
        use_U_weight_delta_t:   float, the delta_t to use for the forward-looking part of the Hilbert subspace determination. 0 disables forward-looking.
        enlarge_steps:          int, the number of additional matrix elements to incorporate when determining the next Hilbert subspace. 0 takes only directly interacting basis states, 1 adds indirect interactions via 1 intermediate, 2 via 2 etc.
        """
        if self.verbose:
            print("Setting up generate_timeline function...")
        header = format_function_args(inspect.currentframe())
        with open(self.params_file, 'a') as params_file:
            params_file.write(header)

        if save_every is not None or save_first or save_last:     # boring string formatting, no physics here
            smallest_val    = numpy.min(numpy.abs(t_array))
            smallest_diff   = numpy.diff(numpy.abs(t_array)).min()
            if smallest_val == 0:
                decnum          = numpy.abs(numpy.floor(numpy.log10(smallest_diff))) + 1
            else:
                decnum          = numpy.abs(numpy.floor(numpy.log10(min(smallest_val, smallest_diff)))) + 1
            largest_val     = numpy.max(numpy.abs(t_array))
            intnum          = numpy.abs(numpy.ceil(numpy.log10(largest_val)))

            format_string   = "wf_file_{0:0%i.%if}" % (intnum + decnum + 1, decnum)
            wf_name_list    = [self.dirname + "/wf_coeffs/" + format_string.format(i) for i in t_array]

        fmode_dict = {"n_b": "ab", "reduced_dm": "ab", "globals": "ab", "diagnostics": "ab", "phonon": "ab"}
        for item in observables:
            if not item in  ["H", "hopping", "coupling", "n_b", "reduced_dm", "diagnostics"] + ["phonon%i" % i for i in range(self.HamObj.nchain)] + ["phononproj%i" % i for i in range(self.HamObj.nchain)]:
                raise ValueError('Unrecognized observable "%s".' % item)

        hopstring       = b"hopping " + b''.join([b"hop%i " %i for i in range(self.HamObj.hopnum)])
        vibstring       = b''.join([b"vib%i " %i for i in range(self.HamObj.nchain)])
        H_file_header   = b"#tag norm " + b"H_tevol " * ("H" in observables) + hopstring * ("hopping" in observables) + vibstring * ("coupling" in observables) + b'\n'
        if "H" in observables or "hopping" in observables or "coupling" in observables:
            with open(os.path.join(self.dirname, "global_operators_real"), 'wb') as H_real_file, open(os.path.join(self.dirname, "global_operators_imag"), 'wb') as H_imag_file:
                H_real_file.write(H_file_header)
                H_imag_file.write(H_file_header)
        if "reduced_dm" in observables:
            with open(os.path.join(self.dirname, "dm_files/reduced_dm_real"), 'wb') as dm_real_file, open(os.path.join(self.dirname, "dm_files/reduced_dm_imag"), 'wb') as dm_imag_file:
                numstring = [b"rho_{%i,%i}" % (i, j) for i in range(9) for j in range(9)]
                dm_real_file.write(b"#tag norm " + b' '.join(numstring) + b'\n')
                dm_imag_file.write(b"#tag norm " + b' '.join(numstring) + b'\n')
        if "diagnostics" in observables:
            with open(os.path.join(self.dirname, "diagnostics.log"), 'wb') as diag_file:
                diag_file.write(b"#tag norm seconds_since_start post_adapt_norm post_adapt_H expm_converged final_m final_expm_term rel_error_expm numstates ceiling_hits\n")
        if "n_b" in observables:
            with open(os.path.join(self.dirname, "local_operators/n_b_real"), 'wb') as n_b_file:
                n_b_file.write(b"#tag norm n^exc_0 n^pho_0 max_HO-n^pho_0 n^exc_1 n^pho_1 max_HO-n^pho_1 etc...\n")

        phononrdm_list  = sorted([i for i in observables if i[:6] == "phonon"])
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

        if vector_device != whoami_device and use_two_streams:
            with whoami_device:
                whoami_stream = cupy.cuda.Stream()

        if self.verbose:
            print("Finished generate_timeline setup! Beginning time evolution...")

        vector = self.vector
        prepare_results = None
        impstates       = None
        max_bath_pre    = None
        for i in range(len(t_array)):
            t       = t_array[i]
            delta_t = t - t_array[i-1]
            if i == 0:
                delta_t = 0        # turn off delta_t at the first timestep
            if self.debug_verb > 0:
                if t == 0:
                    print("\nCheck 1: Initiated initial timestep (no evolution in this step).")
                else:
                    print("\nCheck 1: Initiated timestep from %f to %f." % (t-delta_t, t))

            if self.debug_verb > 0:
                print("Check 2.1: Determining next Hilbert subspace.")
            if vector_device != whoami_device and use_two_streams:
                whoami_stream.synchronize()
#                prepare_results = cupy.asarray(prepare_results)
            ceiling_hits, post_adapt_norm, post_adapt_H     = self.generate_new_Hilbert_space_cupy( vector,
                                                                                                    enlarge_steps=enlarge_steps,
                                                                                                    garbage_tol=garbage_tol,
                                                                                                    do_fancy_stuff="diagnostics" in observables, 
                                                                                                    use_U_weight_function=(use_U_weight_delta_t != 0 and i != 0),
                                                                                                    delta_t=use_U_weight_delta_t,
                                                                                                    prepare_results=prepare_results,
                                                                                                    impstates=impstates)    # max_bath_pre=max_bath_pre)
#                ceiling_hits, post_adapt_norm, post_adapt_H     = self.generate_new_Hilbert_space_cupy(vector, garbage_tol=garbage_tol, do_fancy_stuff="diagnostics" in observables, use_U_weight_function=(use_U_weight_function and delta_t != 0), delta_t=0.2)
            if self.debug_verb > 0:
                print("Check 2.2: Finished determining next Hilbert subspace.")
            if t == 0:
                vector = self.vector    # this is the vector with the new whoami
            else:
                vector, expm_converged, final_m, final_expm_term, rel_error_expm = self.simple_taylor(delta_t, diagnostics=True)
            if self.debug_verb > 0:
                print("Check 3: Calculated v(t).")
            if vector_device != whoami_device and use_two_streams and i < len(t_array) - 1:
                select_whoami, impstates    = self.determine_select_whoami(vector, use_U_weight_function=(use_U_weight_delta_t != 0 and i != 0), delta_t=use_U_weight_delta_t, garbage_tol=garbage_tol)
                if self.debug_verb > 0:
                    print("Beginning concurrent calculation of next Hilbert space.")
                    with whoami_device:
                        with whoami_stream:
                            new_select      = cupy.asarray(select_whoami)
                            prepare_results = self.HamObj.generate_mel(new_select, enlarge_steps=enlarge_steps)
                    del select_whoami

            norm        = self.use_module.linalg.norm(vector).item()
            if self.debug_verb > 0:
                print("Check 4: Calculated norm.")
            H_real_file_savedata = [t, norm]
            H_imag_file_savedata = [t, norm]

            hopping_complex = None
            n_b_sys         = None

            ### Start computing observables:
            if "reduced_dm" in observables:
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

            if "n_b" in observables:
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
            if "H" in observables:
                coupling_complex = self.calculate_coupling(vector)
                if type(coupling_complex) != numpy.ndarray:
                    coupling_complex    = cupy.asnumpy(coupling_complex)
                coupling_real    = coupling_complex.real
                coupling_imag    = coupling_complex.imag
                diag_energy = (abs(vector)**2 * self.diag_vals).sum().item()
                if hopping_complex is None:
                    hopping_complex = self.calculate_hopping(vector)
                if type(hopping_complex) != numpy.ndarray:
                    hopping_complex = cupy.asnumpy(hopping_complex)
                H_complex   = (diag_energy + coupling_complex.sum() + hopping_complex.sum())
                H_real      = H_complex.real
                H_imag      = H_complex.imag
                if self.debug_verb > 0:
                    print("Check 7: Calculated energies.")
                H_real_file_savedata += [H_real]
                H_imag_file_savedata += [H_imag]
            if "hopping" in observables:
                H_real_file_savedata += [hopping_complex.real.sum()] + list(hopping_complex.real)
                H_imag_file_savedata += [hopping_complex.imag.sum()] + list(hopping_complex.imag)
            if "coupling" in observables:
                if coupling_real is None:
                    coupling_complex = self.calculate_coupling(vector)
                    coupling_real    = coupling_complex.real
                    coupling_imag    = coupling_complex.imag
                if self.debug_verb > 0:
                    print("Check 8: Calculated coupling terms.")
                H_real_file_savedata += list(coupling_real)
                H_imag_file_savedata += list(coupling_imag)
            if "coupling" in observables or "H" in observables or "hopping" in observables:
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
            if "diagnostics" in observables:
#   new format:
#                diag_file.write(b"#tag norm seconds_since_start post_adapt_norm post_adapt_H expm_converged final_m final_expm_term rel_error_expm numstates ceiling_hits\n")
                with open(os.path.join(self.dirname, "diagnostics.log"), fmode_dict["diagnostics"]) as diag_file:
                    if t == 0:
                        numpy.savetxt(diag_file, [t, norm, time.time() - function_start_time, post_adapt_norm, post_adapt_H, 0, 0, 0, 0, self.numstates, 0], newline=" ")
                    else:
                        numpy.savetxt(diag_file, [t, norm, time.time() - function_start_time, post_adapt_norm, post_adapt_H, expm_converged, final_m, final_expm_term.item(), rel_error_expm.item(), self.numstates, ceiling_hits.item()], newline=" ")
                    diag_file.write(b'\n')

            if self.debug_verb > mem_info_level:
                print("Used MiB on vector_device: %1.1f or %1.1f" % (cupy.get_default_memory_pool().used_bytes()/1024**2, numpy.diff(cupy.cuda.Device(vector_device).mem_info)[0]/1024**2))
                with cupy.cuda.Device(whoami_device):
                    print("Used MiB on whoami_device: %1.1f or %1.1f" % (cupy.get_default_memory_pool().used_bytes()/1024**2, numpy.diff(cupy.cuda.Device(whoami_device).mem_info)[0]/1024**2))


    ###################################
    # generate new Hilbert subspace from evolved vector using a cupy alternative to numpy.isin
    # (the built-in cupy.isin method is very memory-inefficient)
    ###################################
    def generate_new_Hilbert_space_cupy(self, vector, use_U_weight_function, enlarge_steps=0, delta_t=0, garbage_tol=0, do_fancy_stuff=False, prepare_results=None, impstates=None):
        if prepare_results is None:
            select_whoami, impstates    = self.determine_select_whoami(vector, use_U_weight_function=use_U_weight_function, delta_t=delta_t, garbage_tol=garbage_tol)
            if use_U_weight_function:   # this is somewhat shady, but we need to deal with the t=0 case, where we haven't even created these matrices yet, which will correspond to use_U_weight_function == False:
                del self.sparse_coupling, self.sparse_hopping, self.diag_vals
            # create the new matrices:
            # switch to whoami_device to use that device's RAM instead:
            if whoami_device != vector_device:
                with whoami_device:
                    prepare_results = self.HamObj.generate_mel(cupy.asarray(select_whoami), enlarge_steps=enlarge_steps)
            else:
                prepare_results = self.HamObj.generate_mel(select_whoami, enlarge_steps=enlarge_steps)

#            labels  = [("diag_vals", "new_inds"), ("hopping_vals", ("hop_to", "hop_from")), ("coupl_vals", ("coupl_to", "coupl_from"))]
#            for i in range(3):
#                numpy.save(labels[i][0], prepare_results[i][0])
#                if i == 0:
#                    numpy.save(labels[i][1], prepare_results[i][1])
#                else:
#                    for j in range(2):
#                        numpy.save(labels[i][1][j], prepare_results[i][1][j])

            del select_whoami

        # Now transfer to current_device, if necessary:
        if whoami_device != vector_device:
            t0 = time.time()
            (diag_vals, new_inds), (hopping_vals, hopping_inds), (coupl_vals, coupl_inds), ceiling_hits  = (cupy.asarray(prepare_results[0][0]), cupy.asarray(prepare_results[0][1])), (cupy.asarray(prepare_results[1][0]), (cupy.asarray(prepare_results[1][1][0]), cupy.asarray(prepare_results[1][1][1]))), (cupy.asarray(prepare_results[2][0]), (cupy.asarray(prepare_results[2][1][0]), cupy.asarray(prepare_results[2][1][1]))), prepare_results[3]
            if self.debug_verb > move_data_timing_level:
                t1 = time.time()
                print("Moving data from device %i to device %i took %f ms." % (whoami_device, vector_device, (t1-t0)*1000))
        else:
            (diag_vals, new_inds), (hopping_vals, hopping_inds), (coupl_vals, coupl_inds), ceiling_hits = prepare_results

        if self.debug_verb > 1:
            print("  a: Determined important states: (%i in total)" % impstates)
#        if self.debug_verb > 2:
#            print("    Max bath occ prior to recalc: %i." % max_bath_pre)


        if self.debug_verb > 1:
            print("  b: Created matrix elements.")

#        max_bath_post = new_inds[:,1:].max()
#        if self.debug_verb > 2:
#            print("    Max bath occ after recalc: %i." % max_bath_post)
#        if max_bath_post > max_bath_pre + 1 + enlarge_steps:
#            raise RuntimeError("Max bath occ has increased by more than %i over the course of a single step!" % (1 + enlarge_steps))
#        if max_bath_post > self.HamObj.max_HO_dim:
#            raise RuntimeError("Max phonon occupation (%i) exceeded!" % self.HamObj.max_HO_dim)

        # self.numstates now represents the new numstates:
        self.numstates      = new_inds.shape[0]
        self.create_matrices(diag_vals, (hopping_vals, hopping_inds), (coupl_vals, coupl_inds)) # this needs the new numstates to work
        del hopping_vals, hopping_inds, coupl_vals, coupl_inds

        # insert old vector coefficients; this assumes that all whoami's are sorted
        if self.debug_verb > 2:
            print("    Norm prior to reassignment: %f." % cupy.linalg.norm(vector))
        new_vector          = self.use_module.zeros(self.numstates, dtype=self.HamObj.complex_type)
#        nv                  = new_vector.copy()
        t0 = time.time()
        ind_array           = self.HamObj.searchsorted(self.whoami, new_inds, allow_escapes=True)
        mask_array          = cupy.all(self.whoami[ind_array] == new_inds, axis=1)        # True only where an old value can be copied, i.e. new is in old
        if self.debug_verb > searchsorted_timing_level:
            t1 = time.time()
            print("This application of searchsorted (and masking) took %f ms (arg sizes %i, %i)." % ((t1-t0)*1000, self.whoami.shape[0], new_inds.shape[0]))
            t0 = time.time()
#        new_vector[mask_array]  = vector[cupy.all(new_inds[cupy_search.searchsorted_multidim_list(new_inds, self.whoami, allow_escapes=True, mindiff=mindiff)] == self.whoami, axis=1)]       # assign common values to positions

        new_vector[mask_array]  = vector[ind_array][mask_array]     # assign common values to positions
        del mask_array, ind_array

        self.whoami         = new_inds
        self.partition_lens = cupy_search.calc_partition_lens(self.HamObj.get_pos(self.whoami), self.HamObj.nchain)
        self.vector         = new_vector

        pre_evolve_norm     = -1
        pre_evolve_H        = numpy.nan
        if do_fancy_stuff:
            pre_evolve_norm =   self.use_module.linalg.norm(self.vector).item()
            pre_evolve_H    =   self.calculate_total_energy(self.vector)
            if pre_evolve_H.imag > 1e-12:
                raise RuntimeError("Energy has non-vanishing imaginary part: %1.15f." % pre_evolve_H.imag)
            else:
                pre_evolve_H    = pre_evolve_H.real.item()

        if self.debug_verb > 2:
            if pre_evolve_norm == -1:
                print("    Norm after reassignment: %f." % cupy.linalg.norm(self.vector))
            else:
                print("    Norm after reassignment: %f." % pre_evolve_norm)

        if self.debug_verb > 1:
            print("  c: Inserted previous vector coefficients.")

        return ceiling_hits, pre_evolve_norm, pre_evolve_H



    ###################################
    # if we have two GPUs at our disposal, we can perform
    # the first half of the adaptation step concurrently to the observable calculations:
    ###################################
    def determine_select_whoami(self, vector, use_U_weight_function, delta_t=0, garbage_tol=0):
        # determine the most relevant states:
        if use_U_weight_function:
            if delta_t == 0:
                raise ValueError("Please specify a non-zero delta_t for this function to use U-weighting.")
            tmpvec      = self.first_order_U_importance(vector, delta_t)
        else:
            tmpvec      = self.use_module.abs(vector)
        sorted_indices  = tmpvec.argsort()
        sorted_vector   = tmpvec[sorted_indices]
        del tmpvec

        if (self.shuffle_seed is None
                or len(vector) < self.HamObj.maxstates
                or sorted_vector[-self.HamObj.maxstates-1] < sorted_vector[-self.HamObj.maxstates]):
            if self.debug_verb > 4 and self.shuffle_seed is not None:
                print("    No state shuffling to be performed.")
            select_whoami       = self.whoami[sorted_indices][-self.HamObj.maxstates:]
            if garbage_tol >= 0:
                select_whoami       = select_whoami[sorted_vector[-self.HamObj.maxstates:] > garbage_tol]
        else:
            decision_val        = sorted_vector[-self.HamObj.maxstates]
            firstind            = cupy.searchsorted(sorted_vector, decision_val, "left")
            lastind             = cupy.searchsorted(sorted_vector, decision_val, "right")
            if self.debug_verb > 4:
                print("    Number of states to be shuffled: %i" % (lastind - firstind))
            indfromback         = len(vector) - lastind
            if indfromback > self.HamObj.maxstates:
                raise ValueError("A catastrophic error occurred while shuffling the equal-valued basis states.")

            sorted_whoami                   = self.whoami[sorted_indices]

            select_whoami                   = self.use_module.zeros((self.HamObj.maxstates, self.whoami.shape[1]), dtype=self.whoami.dtype)
            select_whoami[-indfromback:]    = sorted_whoami[-indfromback:]
#            assert cupy.all(sorted_vector[firstind:lastind] == sorted_vector[firstind])
            select_whoami[:-indfromback]    = cupy.random.permutation(sorted_whoami[firstind:lastind])[:self.HamObj.maxstates-indfromback]
            if garbage_tol >= 0:
                raise NotImplementedError("garbage_tol combined with equal-value shuffling has not yet been implemented.")
            del sorted_whoami
        del sorted_indices, sorted_vector
        select_whoami       = self.use_module.array(select_whoami[self.use_module.lexsort(select_whoami.T[::-1])])
        mempool.free_all_blocks()

#        max_bath_pre = select_whoami[:,1:].max()
        return select_whoami, select_whoami.shape[0] #, max_bath_pre


    ###################################
    # determine weight of basis states using a forward-looking method
    ###################################
    def first_order_U_importance_coherence(self, vector, delta_t):
        """
        Determine, more or less, the contribution of each basis state of |psi> to coherence it provides in the future.
        """
        if self.HamObj.coupling_g.imag != 0 or self.HamObj.t_sys.imag != 0:
            raise NotImplementedError("This function hasn't been adapted to complex-valued off-diagonal Hamiltonian matrix elements yet.")
        psi_squared = self.use_module.abs(vector)**2
        ones        = self.use_module.ones(self.numstates)
    # the following is the original version, which, however, doesn't work, so use the other one as long as M is hermitian and real:
#        return psi_squared + (delta_t**2) * psi_squared * (self.use_module.ones(self.numstates).dot(self.sparse_coupling) + self.use_modules.ones(self.numstates).dot(self.sparse_hopping) + self.diag_vals)**2
        return psi_squared + (delta_t**2) * psi_squared * self.use_module.power(self.sparse_coupling.dot(ones) + self.sparse_hopping.dot(ones) + self.diag_vals, 2)

    ###################################
    # determine weight of basis states using a forward-looking method
    ###################################
    def first_order_U_importance_norm(self, vector, delta_t):
        """
        Determine the contribution of each basis state of |psi> to the norm_squared of (1 - i δt H)|psi>.
        """
        return self.use_module.abs(vector - 1j*delta_t * (self.sparse_coupling.dot(vector) + self.sparse_hopping.dot(vector) + self.diag_vals * vector))

###################################################################################################################################################################################


