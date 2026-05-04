"""Kernels and related low-level functions used for the 1D single-exciton Holstein model."""

import numpy
import cupy     # pylint: disable=import-error

####################################################################################################

def add_phonon_at_exc(basis_states, posbitwidth, qhobitwidth, wordsize=32):
    """
    Add one phonon at the site of the exciton across the entire set of states.

    This is done in-place, without creating a copy of basis_states.
    Handles inter-word rollovers, but does not handle under- or overflows.

    Args:
        basis_states (array): (compressed) array of basis states of the Holstein model.
        posbitwidth (int): number of leading bits allocated to the position of the exciton.
        qhobitwidth (array): number of bits allocated to each phononic mode.
        wordsize (int): The wordsize that basis_states is given in. Default: 32.
    """
    _GenPhononAtExc(basis_states, posbitwidth, qhobitwidth, wordsize).add()


def rem_phonon_at_exc(basis_states, posbitwidth, qhobitwidth, wordsize=32):
    """
    Remove one phonon at the site of the exciton across the entire set of states.

    This is done in-place, without creating a copy of basis_states.
    Handles inter-word rollovers, but does not handle under- or overflows.

    Args:
        basis_states (array): (compressed) array of basis states of the Holstein model.
        posbitwidth (int): number of leading bits allocated to the position of the exciton.
        qhobitwidth (array): number of bits allocated to each phononic mode.
        wordsize (int): The wordsize that basis_states is given in. Default: 32.
    """
    _GenPhononAtExc(basis_states, posbitwidth, qhobitwidth, wordsize).subtract()


def get_phonon_at_exc(basis_states, posbitwidth, qhobitwidth, wordsize=32):
    """
    Return an array indicating the phonon occupation number at the site occupied by the exciton.

    Args:
        basis_states (array): (compressed) array of basis states of the Holstein model.
        posbitwidth (int): number of leading bits allocated to the position of the exciton.
        qhobitwidth (array): number of bits allocated to each phononic mode.
        wordsize (int): The wordsize that basis_states is given in. Default: 32.

    Returns:
        1D array: phonon occupation numbers at the exc. Length of array equals that of basis_states.
    """
    return _GenPhononAtExc(basis_states, posbitwidth, qhobitwidth, wordsize).get_occ()


def sum_all_phonons(omega_arr, basis_states, posbitwidth, qhobitwidth, wordsize=32):
    """
    Return an array indicating the sum of all QHO energies for each basis state.

    If all(omega_arr == 1), then this is simply the sum of all occupation numbers for each row.

    Args:
        omega_arr (1D array): energy of each oscillator.
            The shape must be compatible with the width of basis_states.
        basis_states (array): (compressed) array of basis states of the Holstein model.
        posbitwidth (int): number of leading bits allocated to the position of the exciton.
        qhobitwidth (array): number of bits allocated to each phononic mode.
        wordsize (int): The wordsize that basis_states is given in. Default: 32.

    Returns:
        1D array: total phonon occupation numbers. Length of array equals that of basis_states.
    """
    return _GenPhononAtExc(basis_states, posbitwidth, qhobitwidth, wordsize).sum_all(omega_arr)


def calculate_bath_n_b(weights, basis_states, posbitwidth, qhobitwidth, wordsize=32):
    """
    Return an array indicating the expectation value of the bath occupation number at each site.

    Args:
        weights (1D array): The oscillators will be weighted individually by this array.
            The shape must be compatible with the width of basis_states.
        basis_states (array): (compressed) array of basis states of the Holstein model.
        posbitwidth (int): number of leading bits allocated to the position of the exciton.
        qhobitwidth (array): number of bits allocated to each phononic mode.
        wordsize (int): The wordsize that basis_states is given in. Default: 32.

    Returns:
        1D array: average phonon occupation numbers. Length of array equals the number of QHOs.
    """
    return _GenAllPhonon(basis_states, posbitwidth, qhobitwidth, wordsize).bath_n_b(weights)

####################################################################################################
# CUDA code snippets:

_gpu_add32 = r'''
    extern "C" __global__
    void add_pte_kernel(unsigned int* bs, 
        const unsigned int* bit_offset, const unsigned int* word_i, const unsigned int* num2bits,
        const unsigned int rownum, const unsigned int rowlen) 
    {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < rownum) {
            INDEXTYPE arrind  = tid*rowlen;

            unsigned int pos    = bs[arrind] >> (32u-bit_offset[0]);
            arrind              += word_i[pos];

            // add an appropriately bit-shifted 1 to the first or second word:
            bs[arrind + (num2bits[pos] > 32u)] += 1u << ((num2bits[pos] > 32u)*32u + 32u - num2bits[pos]);

            // if the addition was to the second word and caused an overflow there,
            // then add 1 to the first word:
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
            INDEXTYPE arrind  = tid*rowlen;

            unsigned int pos    = bs[arrind] >> (32u-bit_offset[0]);
            arrind              += word_i[pos];

            // if the subtraction is from the second word and would cause and underflow there,
            // then subtract 1 from the first word and bit-flip zeros in second word:
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
            INDEXTYPE arrind  = tid*rowlen;

            unsigned int pos    = bs[arrind] >> (32u-bit_offset[0]);
            arrind              += word_i[pos];

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
            INDEXTYPE arrind;
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
        INDEXTYPE arrind;
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

####################################################################################################
# kernel definitions:

_add_pte_kernel     = cupy.RawKernel(_gpu_add32.replace("INDEXTYPE", "unsigned int"),
                                                "add_pte_kernel")

_remove_pte_kernel  = cupy.RawKernel(_gpu_rem32.replace("INDEXTYPE", "unsigned int"),
                                                "rem_pte_kernel")

_get_occ_pte_kernel = cupy.RawKernel(_gpu_get32.replace("INDEXTYPE", "unsigned int"),
                                                "get_occ_pte_kernel")

_sum_occ_pte_kernel = cupy.RawKernel(_gpu_sum32.replace("INDEXTYPE", "unsigned int"),
                                                "sum_occ_pte_kernel")

_calc_bath_n_b_kernel = cupy.RawKernel(_gpu_calc_n_b32.replace("INDEXTYPE", "unsigned int"),
                                                "calc_bath_n_b_kernel", backend="nvcc")


_add_pte_kernel_64      = cupy.RawKernel(_gpu_add32.replace(
                                            "INDEXTYPE", "unsigned long long").replace(
                                            "add_pte_kernel", "add_pte_kernel_64"),
                                        "add_pte_kernel_64")

_remove_pte_kernel_64   = cupy.RawKernel(_gpu_rem32.replace(
                                            "INDEXTYPE", "unsigned long long").replace(
                                            "rem_pte_kernel", "rem_pte_kernel_64"),
                                        "rem_pte_kernel_64")

_get_occ_pte_kernel_64  = cupy.RawKernel(_gpu_get32.replace(
                                            "INDEXTYPE", "unsigned long long").replace(
                                            "get_occ_pte_kernel", "get_occ_pte_kernel_64"),
                                        "get_occ_pte_kernel_64")

_sum_occ_pte_kernel_64  = cupy.RawKernel(_gpu_sum32.replace(
                                            "INDEXTYPE", "unsigned long long").replace(
                                            "sum_occ_pte_kernel", "sum_occ_pte_kernel_64"),
                                        "sum_occ_pte_kernel_64")

_calc_bath_n_b_kernel_64 = cupy.RawKernel(_gpu_calc_n_b32.replace(
                                            "INDEXTYPE", "unsigned long long").replace(
                                            "calc_bath_n_b_kernel", "calc_bath_n_b_kernel_64"),
                                            "calc_bath_n_b_kernel_64", backend="nvcc")

####################################################################################################
# the central class that ties together all of the functions provided by this script:

class _GenDatSegBase:
    def __init__(self, basis_states, posbitwidth, qhobitwidth, wordsize=32):
        if basis_states.dtype != f"uint{wordsize}":
            raise ValueError(f"dtype of input ({basis_states.dtype}) does not match"
                                f"specified wordsize ({wordsize} bits).")
        if basis_states.shape[1] != numpy.ceil((posbitwidth + qhobitwidth.sum())/wordsize):
            raise ValueError("Shape of input states does not match bin sizes.")
        if not basis_states.flags["C_CONTIGUOUS"]:
            raise ValueError("The gendatseg routines only work with"
                        " C-contiguous basis_states arrays"
                        " (consider calling cupy.ascontiguousarray() on the input array first).")

        # index of first relevant bit for each QHO:
        bit_offset  = (posbitwidth + qhobitwidth.cumsum() - qhobitwidth).astype(f"uint{wordsize}")
        # index of first word containing the block:
        word_i      = bit_offset // wordsize
        # wordsize + number of relevant bits contained in word_i+1 (num2bits == wordsize is "zero"):
        num2bits    = bit_offset + qhobitwidth - wordsize*word_i

        threads_per_block   = 256
        blocks              = numpy.ceil(basis_states.shape[0]/threads_per_block).astype(int)

        self.funcargs       = [(blocks,), (threads_per_block,),
                                [basis_states, bit_offset, word_i, num2bits,
                                *[cupy.uint32(i) for i in basis_states.shape]]]
        self.wordsize       = wordsize
        self.numsites       = len(qhobitwidth)
        if numpy.prod(basis_states.shape) > 4294967296:
            self.suffix = "_64"
        else:
            self.suffix = ""

class _GenPhononAtExc(_GenDatSegBase):
    def add(self):
        """See global function add_phonon_at_exc."""
        if self.wordsize == 32:
            globals()["_add_pte_kernel" + self.suffix](*self.funcargs)
        else:
            raise NotImplementedError("wordsizes other than 32 bits have not yet been implemented.")


    def subtract(self):
        """See global function rem_phonon_at_exc."""
        if self.wordsize == 32:
            globals()["_remove_pte_kernel" + self.suffix](*self.funcargs)
        else:
            raise NotImplementedError("wordsizes other than 32 bits have not yet been implemented.")


    def get_occ(self):
        """See global function get_phonon_at_exc."""
        result          = cupy.zeros(int(self.funcargs[2][-2]), dtype=f"uint{self.wordsize}")
        self.funcargs[2] += [result,]
        if self.wordsize == 32:
            globals()["_get_occ_pte_kernel" + self.suffix](*self.funcargs)
        else:
            raise NotImplementedError("wordsizes other than 32 bits have not yet been implemented.")
        # return the last item (i.e., results) and remove it from the list:
        return self.funcargs[2].pop()


    def sum_all(self, omega_arr):
        """See global function sum_all_phonons."""
        result          = cupy.zeros(int(self.funcargs[2][-2]), dtype="float64")
        omega_arr       = omega_arr.astype("float64", copy=False)

        self.funcargs[2] += [result, omega_arr, cupy.uint32(self.numsites)]

        if self.wordsize == 32:
            globals()["_sum_occ_pte_kernel" + self.suffix](*self.funcargs)
        else:
            raise NotImplementedError("wordsizes other than 32 bits have not yet been implemented.")
        self.funcargs[2].pop()          # remove numsites from funcargs
        self.funcargs[2].pop()          # remove omega_arr from funcargs
        return self.funcargs[2].pop()   # remove and return result array


class _GenAllPhonon(_GenDatSegBase):
    def bath_n_b(self, weights):
        """See global function calculate_bath_n_b."""
        if self.funcargs[1][0] != 256:
            raise NotImplementedError("A block size of 256 was hard-coded into this kernel.")
        result          = cupy.zeros(self.numsites, dtype="float64")
        self.funcargs[2] += [result, weights, cupy.uint32(self.numsites)]

        if self.wordsize == 32:
            globals()["_calc_bath_n_b_kernel" + self.suffix](*self.funcargs)
        else:
            raise NotImplementedError("wordsizes other than 32 bits have not yet been implemented.")
        self.funcargs[2].pop()          # remove numsites from funcargs
        self.funcargs[2].pop()          # remove omega_arr from funcargs
        return self.funcargs[2].pop()   # remove and return result array
