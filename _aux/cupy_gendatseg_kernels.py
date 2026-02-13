#!/usr/bin/env python3
# coding: utf-8
 
import timeit
import time
import numpy

import cupy
import cupyx

###################################################################################################################################################################################

def add_phonon_at_exc(basis_states, posbitwidth, QHObitwidth, wordsize=32):
    """
    Takes a (compressed) array basis_states and, for each of the states therein, adds a single phonon to the site occupied by the exciton.
    This is done in-place, without creating a copy of basis_states.
    Handles inter-byte rollovers, but does not handle under- or overflows.
    """
    _gen_phonon_at_exc(basis_states, posbitwidth, QHObitwidth, wordsize).add()


def rem_phonon_at_exc(basis_states, posbitwidth, QHObitwidth, wordsize=32):
    """
    Takes a (compressed) array basis_states and, for each of the states therein, removes a single phonon to the site occupied by the exciton.
    This is done in-place, without creating a copy of basis_states.
    Handles inter-byte rollovers, but does not handle under- or overflows.
    """
    _gen_phonon_at_exc(basis_states, posbitwidth, QHObitwidth, wordsize).subtract()


def get_phonon_at_exc(basis_states, posbitwidth, QHObitwidth, wordsize=32):
    """
    Takes a (compressed) array basis_states of shape (M, N) and returns a new array of shape (M,) containing the occupation number at the site occupied by the excitation.
    """
    return _gen_phonon_at_exc(basis_states, posbitwidth, QHObitwidth, wordsize).get_occ()


def sum_all_phonons(omega_arr, basis_states, posbitwidth, QHObitwidth, wordsize=32):
    """
    Takes a (compressed) array basis_states of shape (M, N) together with an array omega_arr of shape (N,) and returns a new array of shape (M,) containing the sum of all QHO energies for the given basis state.
    If all(omega_arr == 1), this is simply equal to the sum of all occupation numbers for each basis state.
    """
    return _gen_phonon_at_exc(basis_states, posbitwidth, QHObitwidth, wordsize).sum_all(omega_arr)


def calculate_bath_n_b(weights, basis_states, posbitwidth, QHObitwidth, wordsize=32):
    """
    Takes a (compressed) array basis_states of shape (M, N) and returns a new array of shape (N,) containing the expectation value of the bath occupation number at each site.
    """
    return _gen_all_phonon(basis_states, posbitwidth, QHObitwidth, wordsize).bath_n_b(weights)

###################################################################################################################################################################################
# CUDA code snippets:

_gpu_add32_prefix = r'''
    extern "C" __global__
    void add_pte_kernel(unsigned int* bs, 
        const unsigned int* bit_offset, const unsigned int* word_i, const unsigned int* num2bits,
        const unsigned int rownum, const unsigned int rowlen) 
    {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < rownum) {
            unsigned int pos    = bs[tid*rowlen] >> (32u-bit_offset[0]);
            unsigned int arrind = tid*rowlen + word_i[pos];
    '''

_gpu_add32_core_vanilla = r'''
            if (num2bits[pos] < 33) {
                bs[arrind] += (1u << 32u-num2bits[pos]);
            }
            else {
                bs[arrind+1]    += (1u << (64u-num2bits[pos]));
                if ( bs[arrind+1]   >> (64u-num2bits[pos]) == ((~0u) >> (64u-num2bits)) ) {
                    bs[arrind]      += 1;
                }
            }
        }
    }
    '''


_gpu_add32_core = r'''
            // add an appropriately bit-shifted 1 to the first or second word:
            bs[arrind + (num2bits[pos] > 32u)] += 1u << ((num2bits[pos] > 32u)*32u + 32u - num2bits[pos]);

            // if the addition was to the second word and caused an overflow there, add 1 to the first word:
            if ( (num2bits[pos] > 32u) && ((bs[arrind+1] >> (64u-num2bits[pos])) == 0) ) {
                bs[arrind]      += 1;
            }
        }
    }
    '''

##########################################################################################

_gpu_rem32 = r'''
    extern "C" __global__
    void rem_pte_kernel(unsigned int* bs, 
        const unsigned int* bit_offset, const unsigned int* word_i, const unsigned int* num2bits,
        const unsigned int rownum, const unsigned int rowlen) 
    {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < rownum) {
            unsigned int pos    = bs[tid*rowlen] >> (32u-bit_offset[0]);
            unsigned int arrind = tid*rowlen + word_i[pos];

            // if the subtraction is from the second word and would cause and underflow there, subtract 1 from the first word and bit-flip zeros in second word:
            if ( (num2bits[pos] > 32u) && ((bs[arrind+1] >> (64u-num2bits[pos])) == 0) ) {
                bs[arrind]      -= 1;
                bs[arrind+1]    += (~0u << (64u-num2bits[pos]));
            }
            else {
            // subtract an appropriately bit-shifted 1 from the first or second word:
                bs[arrind + (num2bits[pos] > 32u)] -= 1u << ((num2bits[pos] > 32u)*32u + 32u - num2bits[pos]);
            }

        }
    }
    '''

##########################################################################################

_gpu_get32 = r'''
    extern "C" __global__
    void get_occ_pte_kernel(const unsigned int* bs, 
        const unsigned int* bit_offset, const unsigned int* word_i, const unsigned int* num2bits,
        const unsigned int rownum, const unsigned int rowlen,
        unsigned int* result)
    {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < rownum) {
            unsigned int pos    = bs[tid*rowlen] >> (32u-bit_offset[0]);
            unsigned int arrind = tid*rowlen + word_i[pos];

            result[tid]     = bs[arrind];
            result[tid]     &= ~0u >> (bit_offset[pos] - 32u*word_i[pos]);

            if (num2bits[pos] > 32u) {
                result[tid]     <<= num2bits[pos] - 32u;
                result[tid]     += bs[arrind+1] >> (64u - num2bits[pos]);
            }
            else {
                result[tid]     >>= 32u - num2bits[pos];
            }
        }
    }
    '''

##########################################################################################

_gpu_sum32 = r'''
    extern "C" __global__
    void sum_occ_pte_kernel(const unsigned int* bs, 
        const unsigned int* bit_offset, const unsigned int* word_i, const unsigned int* num2bits,
        const unsigned int rownum, const unsigned int rowlen,
        double* result, const double* omega, const unsigned int numsites)
    {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < rownum) {
            unsigned int arrind;
            unsigned int tmp;

            for (unsigned int site = 0u; site < numsites; ++site) {
                arrind  = tid*rowlen + word_i[site];
                tmp     = bs[arrind];
                tmp     &= ~0u >> (bit_offset[site] - 32u*word_i[site]);
                if (num2bits[site] > 32u) {
                    tmp     <<= num2bits[site] - 32u;
                    tmp     += bs[arrind+1] >> (64u - num2bits[site]);
                }
                else {
                    tmp     >>= 32u - num2bits[site];
                }
                result[tid] += tmp * omega[site];
            }
        }
    }
    '''


##########################################################################################

_gpu_calc_n_b32 = r'''
    #include <cub/cub.cuh> 
    extern "C" __global__
    void calc_bath_n_b_kernel(const unsigned int* bs, 
        const unsigned int* bit_offset, const unsigned int* word_i, const unsigned int* num2bits,
        const unsigned int rownum, const unsigned int rowlen,
        double* result, const double* weights, const unsigned int numsites)
    {
        unsigned int tid    = blockDim.x * blockIdx.x + threadIdx.x;
        int num_valid       = min(rownum - blockDim.x * blockIdx.x, blockDim.x);
        double thread_data;
        unsigned int arrind;
        unsigned int tmp;

        // Specialize BlockReduce
        typedef cub::BlockReduce<double, 256> BlockReduce;
        // Allocate shared memory for BlockReduce
        __shared__ typename BlockReduce::TempStorage temp_storage;

        for (unsigned int site = 0u; site < numsites; ++site) {
            if (tid < rownum) {
                arrind  = tid*rowlen + word_i[site];
                tmp     = bs[arrind];
                tmp     &= ~0u >> (bit_offset[site] - 32u*word_i[site]);
                if (num2bits[site] > 32u) {
                    tmp     <<= num2bits[site] - 32u;
                    tmp     += bs[arrind+1] >> (64u - num2bits[site]);
                }
                else {
                    tmp     >>= 32u - num2bits[site];
                }
                thread_data = tmp * weights[tid];
            }
            double aggregate = BlockReduce(temp_storage).Sum(thread_data, num_valid);
            if (threadIdx.x == 0) {
                atomicAdd(result + site, aggregate);
            }
            __syncthreads();
        }
    }
    '''

###################################################################################################################################################################################
# kernel definitions:

_add_pte_kernel         = cupy.RawKernel(_gpu_add32_prefix + _gpu_add32_core, "add_pte_kernel")

_remove_pte_kernel      = cupy.RawKernel(_gpu_rem32, "rem_pte_kernel")

_get_occ_pte_kernel     = cupy.RawKernel(_gpu_get32, "get_occ_pte_kernel")

_sum_occ_pte_kernel     = cupy.RawKernel(_gpu_sum32, "sum_occ_pte_kernel")

_calc_bath_n_b_kernel   = cupy.RawKernel(_gpu_calc_n_b32, "calc_bath_n_b_kernel", backend="nvcc")

###################################################################################################################################################################################
# the central class that ties together all of the functions provided by this script:

class _gendatseg_base:
    def __init__(self, basis_states, posbitwidth, QHObitwidth, wordsize=32):
        if basis_states.dtype != "uint%i" % wordsize:
            raise ValueError("dtype of input (%s) does not match specified wordsize (%i bits)." % (basis_states.dtype, wordsize))
        if basis_states.shape[1] != numpy.ceil((posbitwidth + QHObitwidth.sum())/wordsize):
            raise ValueError("Shape of input states does not match bin sizes.")
        if not basis_states.flags["C_CONTIGUOUS"]:
            raise ValueError("The gendatseg routines only work with C-contiguous basis_states arrays (consider calling cupy.ascontiguousarray() on the input array first).")

        bit_offset      = (posbitwidth + QHObitwidth.cumsum() - QHObitwidth).astype("uint%i" % wordsize)    # index of first relevant bit for each QHO
        word_i          = bit_offset // wordsize                                                            # index of first word containing the block
        num2bits        = bit_offset + QHObitwidth - wordsize*word_i                                        # wordsize + number of relevant bits contained in word_i+1 (num2bits == wordsize is "zero")

        threads_per_block   = 256
        blocks              = numpy.ceil(basis_states.shape[0]/threads_per_block).astype(int)

        self.funcargs       = [(blocks,), (threads_per_block,), [basis_states, bit_offset, word_i, num2bits, *[cupy.uint32(i) for i in basis_states.shape]]]
        self.wordsize       = wordsize
        self.numsites       = len(QHObitwidth)

class _gen_phonon_at_exc(_gendatseg_base):
    def add(self):
        if self.wordsize == 32:
            _add_pte_kernel(*self.funcargs)
        else:
            raise NotImplementedError("wordsizes other than 32 bits have not yet been implemented.")


    def subtract(self):
        if self.wordsize == 32:
            _remove_pte_kernel(*self.funcargs)
        else:
            raise NotImplementedError("wordsizes other than 32 bits have not yet been implemented.")


    def get_occ(self):
        result          = cupy.zeros(int(self.funcargs[2][-2]), dtype="uint%i" % self.wordsize)
        self.funcargs[2] += [result,]
        if self.wordsize == 32:
            _get_occ_pte_kernel(*self.funcargs)
        else:
            raise NotImplementedError("wordsizes other than 32 bits have not yet been implemented.")
        return self.funcargs[2].pop()  # return the last item (i.e., results) and remove it from the list


    def sum_all(self, omega_arr):
        result          = cupy.zeros(int(self.funcargs[2][-2]), dtype="float64")
        omega_arr       = omega_arr.astype("float64", copy=False)

        self.funcargs[2] += [result, omega_arr, cupy.uint32(self.numsites)]

        if self.wordsize == 32:
            _sum_occ_pte_kernel(*self.funcargs)
        else:
            raise NotImplementedError("wordsizes other than 32 bits have not yet been implemented.")
        self.funcargs[2].pop()          # remove numsites from funcargs
        self.funcargs[2].pop()          # remove omega_arr from funcargs
        return self.funcargs[2].pop()   # remove and return result array


class _gen_all_phonon(_gendatseg_base):
    def bath_n_b(self, weights):
        if self.funcargs[1][0] != 256:
            raise NotImplementedError("A block size of 256 was hard-coded into this kernel, please adapt.")

        result          = cupy.zeros(self.numsites, dtype="float64")
        self.funcargs[2] += [result, weights, cupy.uint32(self.numsites)]

        if self.wordsize == 32:
            _calc_bath_n_b_kernel(*self.funcargs)
        else:
            raise NotImplementedError("wordsizes other than 32 bits have not yet been implemented.")
        self.funcargs[2].pop()          # remove numsites from funcargs
        self.funcargs[2].pop()          # remove omega_arr from funcargs
        return self.funcargs[2].pop()   # remove and return result array

###################################################################################################################################################################################


