"""Low-level algorithms and CUDA kernels relating to searching in cupy and CUDA."""

# XXX This submodule should be further cleaned up and documented!

import time

from dataclasses import dataclass

import numpy
import cupy     # pylint: disable=import-error

####################################################################################################

def calc_partition_lens(partition_by, uniquevals):
    """
    Calculates the length of the blocks of repeating elements in an array.

    Args:
        partition_by (1D cupy.ndarray): A sorted array whose partition length are to be computed.
        uniquevals (1D cupy.ndarray): The array of unique values that should occur in partition_by.
            uniquevals must be sorted and unique.
            Not every value in uniquevals must necessarily occur in partition_by
            (i.e., some partition lengths may be zero),
            but every value in partition_by must indeed occur in uniquevals.

    Returns:
        1D cupy.ndarray: The length of each repeating block in partition_by.
    """
    if partition_by.ndim != 1 or uniquevals.ndim != 1:
        raise ValueError("This function only takes 1D arrays as input.")
    diff_inds   = find_changes_local_single(partition_by)
    # If not every value occurs in diff_inds:
    if len(diff_inds) < len(uniquevals):
        # Then find the index where each of the values in uniquevals would belong:
        insert_at   = cupy.searchsorted(partition_by[diff_inds], uniquevals)
        # And assign the nearest fault (change) index to the non-existent values:
        diff_inds   = cupy.concatenate((diff_inds, cupy.array([len(partition_by)])))[insert_at]
    elif len(diff_inds) > len(uniquevals):
        raise ValueError("More unique values were found in partition_by than in uniquevals.")
    # One could add another check confirming that
    # cupy.all(cupy.isin(partition_by[diff_inds], uniquevals)) here
    if len(partition_by) < 2**32:
        lens    = cupy.empty(len(diff_inds), dtype=cupy.uint32)
    else:
        lens    = cupy.uint64
    assert diff_inds[0] == 0
    lens[:-1]   = cupy.diff(diff_inds)
    lens[-1]    = len(partition_by) - diff_inds[-1]
    return lens


def jagged_to_regular(a, lens):
    mask        = lens[:,None] > cupy.arange(lens.max())
    out         = cupy.zeros(mask.shape, dtype=a.dtype)
    out[mask]   = a
    return out


def find_changes_local_single(arr, mindiff=0):
    """
    Takes a (sorted) 1D array and returns the segmentation indices where the value changes.
    """
    if arr.shape[0] < mindiff:
        raise ValueError("Function called on an array that is smaller than the specified mindiff.")
    if arr.ndim != 1:
        raise ValueError("This function only takes 1D arrays as input.")
    dtype       = numpy.uint32 if len(arr) < 2**32 else numpy.uint64

    mask        = cupy.empty(arr.shape[0], dtype=bool)
    mask[0]     = True
    mask[1:]    = arr[1:] != arr[:-1]
    return cupy.nonzero(mask)[0].astype(dtype)

####################################################################################################

def searchsorted_multidim_list_vanilla(phonebook, findme, mindiff=32, allow_escapes=False,
                                                                linear_only=False, shutup=True):
    return searchsorted_multidim_list_old(phonebook, findme, mindiff, allow_escapes,
                                                                linear_only, shutup).directcalc()

def searchsorted_multidim_list_8bits(phonebook, findme, mindiff=32, allow_escapes=False,
                                                                linear_only=False, shutup=True):
    return searchsorted_multidim_list_old(phonebook, findme, mindiff, allow_escapes,
                                                                linear_only, shutup).parseto8bits()


def _check_if_trivial(phonebook, findme, allow_escapes):
    if findme.shape[0]      == 0:
        return cupy.array([], dtype=cupy.uint32)
    if phonebook.shape[0]   == 1:
        result = cupy.all(phonebook < findme, axis=1).astype(cupy.uint32)
        if not allow_escapes and cupy.any(result):
            raise ValueError("Target values not found in phonebook (which was of len 1)!")
        return result

    return None

class searchsorted_multidim_list_old:
    def __init__(self, phonebook, findme, mindiff, allow_escapes, linear_only, shutup):
        if phonebook.dtype.kind == 'c' or findme.dtype.kind == 'c':
            raise ValueError("Why are you trying to search in complex-valued arrays?")
        if phonebook.shape[0] >= 2**32-1 or phonebook.shape[1] >= 2**16:
            raise NotImplementedError("This function will not work with arrays whose dimensions"
                                            " exceed (2**32 - 2, 2**16 - 1).")
        if phonebook.dtype != findme.dtype:
            raise TypeError(f"Mismatched dtypes: phonebook has dtype {phonebook.dtype} and"
                                f" findme has dtype {findme.dtype}.")
        if phonebook.dtype not in (cupy.uint8, cupy.uint32):
            raise NotImplementedError(f"dtypes {phonebook.dtype}, {findme.dtype} for phonebook"
                                        " and findme not supported.")
        if phonebook.shape[1] != findme.shape[1]:
            raise ValueError("Incompatible array shapes.")

        self.phonebook  = phonebook
        self.findme     = findme
        self.mindiff    = mindiff
        self.allow_escapes= allow_escapes
        self.linear_only= linear_only
        self.shutup     = shutup

        #############################################
        # will be set later by functions:
        #############################################
        self.skib = ''
        self.skil = ''
        self.skil_imp = ''
        self.use_binary = ''


    def parseto8bits(self):
        if self.phonebook.dtype != cupy.uint32:
            raise NotImplementedError("dtype must be uint32 for parseto8bits.")
        wordnum         = self.phonebook.shape[1]
        self.skib       = _searchkernel_individual_binary8
        self.skil       = _searchkernel_individual_linear8
        self.skil_imp   = _searchkernel_individual_linear_improved8

        trivres     = _check_if_trivial(self.phonebook, self.findme, self.allow_escapes)
        if not trivres is None:
            return trivres
        result_inds = self._preprocess()

        it_arr          = numpy.repeat(range(wordnum), 4)*4 + numpy.tile([3,2,1,0], wordnum)
        for colnum in it_arr:
            if not self.shutup:
                print(f"Column {colnum}...", end=' ')
            thisbook    = self.phonebook.view("u1")[:,colnum]
            thisfind    = self.findme.view("u1")[:,colnum]
            result_inds = self._single_loop(result_inds, thisbook, thisfind)
        return result_inds


    def directcalc(self):
        if self.phonebook.dtype == self.findme.dtype == cupy.uint8:
            self.skib       = _searchkernel_individual_binary8
            self.skil       = _searchkernel_individual_linear8
            self.skil_imp   = _searchkernel_individual_linear_improved8
        elif self.phonebook.dtype == self.findme.dtype == cupy.uint32:
            self.skib       = _searchkernel_individual_binary32
            self.skil       = _searchkernel_individual_linear32
            self.skil_imp   = _searchkernel_individual_linear_improved32

        trivres     = _check_if_trivial(self.phonebook, self.findme, self.allow_escapes)
        if not trivres is None:
            return trivres
        result_inds = self._preprocess()

        it_arr              = numpy.arange(self.phonebook.shape[1])
        for colnum in it_arr:
            if not self.shutup:
                print(f"Column {colnum}...", end=' ')
            thisbook, thisfind = self.phonebook[:,colnum], self.findme[:,colnum]
            result_inds = self._single_loop(result_inds, thisbook, thisfind)
        return result_inds

##########################################################################################

    def _preprocess(self):
        if self.phonebook.shape[0] <= self.mindiff:
        # this is the global override option for use_binary;
        # if this is True, then the diffs will not be computed and use_binary will always be False:
            self.linear_only = True
        if self.linear_only:
            self.use_binary  = False     # this is the per-loop decision variable

        result_inds     = cupy.zeros(self.findme.shape[0], dtype=cupy.uint32)
        return result_inds


    def _single_loop(self, result_inds, thisbook, thisfind):
        shutup  = self.shutup
        if not self.linear_only:
            # Check whether we should perform a binary search in the current step:
            t0 = time.time()
            thisdiff            = find_changes_local_single(thisbook, mindiff=self.mindiff)
            if len(thisdiff) == 1:
                if not shutup:
                    print("Constant column condition triggered.")
                escapes             = thisfind > thisbook[0]
                if cupy.any(escapes):
                    if self.allow_escapes:
                        result_inds[escapes] = 4294967295
                    else:
                        print(thisfind, thisbook)
                        raise RuntimeError("Column contains only one value which does not match"
                                                " the requested values.")
#                continue        # skip the rest of the loop if this condition was satisfied
                return result_inds
            diffarr = cupy.diff(thisdiff)
            if diffarr.max() >= self.mindiff:
                # use a binary search in thisdiff to find the next partner to check:
                if not shutup:
                    print("Binary search...", end=' ')
                self.use_binary = True
            else:
                # else do a linear search in thisbook and check every single partner
                # (if not use_binary and not constant column)
                if not shutup:
                    print("Linear search...", end=' ')
                self.use_binary = False
            del diffarr
            if not shutup:
                print(f"Thisdiff and method determination took {(time.time() - t0)*1000} ms.",
                                                                                            end=' ')
        t0 = time.time()
        previous_inds = result_inds.copy()
        if self.use_binary:
            self.skib(thisfind, thisbook, previous_inds, thisdiff, self.allow_escapes,
                                    cupy.array([len(thisdiff)], dtype=cupy.uint32), result_inds)
        elif not self.linear_only:
            self.skil_imp(thisfind, thisbook, previous_inds, self.allow_escapes,
                                    cupy.array([len(thisbook)], dtype=cupy.uint32),
                                    cupy.array([thisdiff[-1]], dtype=cupy.uint32), result_inds)
        else:
            self.skil(thisfind, thisbook, previous_inds, self.allow_escapes,
                                    cupy.array([len(thisbook)], dtype=cupy.uint32), result_inds)
        if not shutup:
            t1 = time.time()
            print(f"Search & sync took: {(t1 - t0) * 1000} ms.", end=' ')
            if t1 - t0 > 1:
                cupy.save("last_thisbook", thisbook)
                cupy.save("last_thisfind", thisfind)
                cupy.save("last_previous_inds", previous_inds)
                cupy.save("last_result_inds", result_inds)
                raise RuntimeError("Search timeout (>1 s), check dumped arrays.")
        if not shutup:
            print(f"Copy: {(time.time() - t1) * 1000} ms.")

        return result_inds

####################################################################################################
# CUDA code snippets:

skil_code   = r'''
    if (startfrom[i] == 4294967295) {
        y = 4294967295;
        return;        
    }
    unsigned int   res_ind = startfrom[i];

    while (phonebook[res_ind] < findme) {
        res_ind += 1;
//        printf("res_ind: %hu, len(phonebook): %u.\n", res_ind, len_phonebook[0]);
        if (res_ind >= len_phonebook[0]) {
//            printf("Condition triggered.\n");
            if (allow_escapes) {
                y = 4294967295;
                return;
            }
            else {
                printf("Error! Reached end of array.\n");
                assert(0);
            }
        }
        if (phonebook[res_ind] < phonebook[res_ind-1]) {
            if (allow_escapes) {
                y = 4294967295;
                return;
            }
            else {
                printf("Error! Went from %hu to %hu. Searching for: %hu.\n",
                            phonebook[res_ind-1], phonebook[res_ind], findme);
                assert(0);
            }
        }
    }
    y = res_ind;
    '''

##########################################################################################

skil_improved_code  = r'''
    if (startfrom[i] == 4294967295) {
        y = 4294967295;
        return;
    }
    // prevent unnecessary looping if the startfrom index is past the last change:
    if (startfrom[i] >= last_diff_ind[0] & findme > phonebook[startfrom[i]]) {
        if (allow_escapes) {
            y = 4294967295;
            return;
        }
        else {
            printf("Error! Reached end of array.\n");
            assert(0);
        }
    }

    unsigned int   res_ind = startfrom[i];

    while (phonebook[res_ind] < findme) {
        res_ind += 1;
//        printf("res_ind: %hu, len(phonebook): %u.\n", res_ind, len_phonebook[0]);
        if (res_ind >= len_phonebook[0]) {
//            printf("Condition triggered.\n");
            if (allow_escapes) {
                y = 4294967295;
                return;
            }
            else {
                printf("Error! Reached end of array.\n");
                assert(0);
            }
        }
        if (phonebook[res_ind] < phonebook[res_ind-1]) {
            if (allow_escapes) {
                y = 4294967295;
                return;
            }
            else {
                printf("Error! Went from %hu to %hu. Searching for: %hu.\n",
                            phonebook[res_ind-1], phonebook[res_ind], findme);
                assert(0);
            }
        }
    }
    y = res_ind;
    '''

##########################################################################################

skib_code   = r'''
    if (startfrom[i] == 4294967295) {
        y = 4294967295;
        return;
    }
    unsigned int   res_ind = startfrom[i];

    // if input index matches requested index, terminate immediately:
    if (phonebook[res_ind] == findme) {
        y = res_ind;
        return;
    }

    // perform binary search in difflist to find next value to compare to:
    unsigned int left  = 0;
    unsigned int right = n_diffs[0]-1;
    unsigned int m;
    while (left < right) {
        m = (right + left) / 2;
        if (difflist[m] < res_ind) {
            left = m + 1;
        } else {
            right = m;
        }
    }

    // iterate through difflist (starting with index "right") to find partners to compare to:
    res_ind = difflist[right];
    while (phonebook[res_ind] < findme) {
        right += 1;
        // prevent illegal memory accesses:
        if (right >= n_diffs[0]) {
            if (allow_escapes) {
                y = 4294967295;
                return;
            }
            else {
                printf("Error! Reached end of array.\n");
                assert(0);
            }
        }
        res_ind = difflist[right];
        if (phonebook[res_ind] <= phonebook[difflist[right-1]]) {
            if (allow_escapes) {
                y = 4294967295;
                return;
            }
            else {
                printf("Error! Went from %hu to %hu. Searching for: %hu.\n",
                            phonebook[res_ind-1], phonebook[res_ind], findme);
                assert(0);
            }
        }
    }
    y = res_ind;
    '''

##########################################################################################

_searchkernel_individual_linear8 = cupy.ElementwiseKernel(
    'uint8 findme, raw uint8 phonebook, raw uint32 startfrom,'
        ' bool allow_escapes, raw uint32 len_phonebook',
    'uint32 y',
    skil_code,
    name="searchkernel_individual_linear8")

_searchkernel_individual_linear32 = cupy.ElementwiseKernel(
    'uint32 findme, raw uint32 phonebook, raw uint32 startfrom,'
        ' bool allow_escapes, raw uint32 len_phonebook',
    'uint32 y',
    skil_code,
    name="searchkernel_individual_linear32")

_searchkernel_individual_linear_improved8 = cupy.ElementwiseKernel(
    'uint8 findme, raw uint8 phonebook, raw uint32 startfrom,'
        ' bool allow_escapes, raw uint32 len_phonebook, raw uint32 last_diff_ind',
    'uint32 y',
    skil_improved_code,
    name="searchkernel_individual_linear_improved8")

_searchkernel_individual_linear_improved32 = cupy.ElementwiseKernel(
    'uint32 findme, raw uint32 phonebook, raw uint32 startfrom,'
        ' bool allow_escapes, raw uint32 len_phonebook, raw uint32 last_diff_ind',
    'uint32 y',
    skil_improved_code,
    name="searchkernel_individual_linear_improved32")

_searchkernel_individual_binary8 = cupy.ElementwiseKernel(
    'uint8 findme, raw uint8 phonebook, raw uint32 startfrom,'
        ' raw uint32 difflist, bool allow_escapes, raw uint32 n_diffs',
    'uint32 y',
    skib_code,
    name="searchkernel_individual_binary8")

_searchkernel_individual_binary32 = cupy.ElementwiseKernel(
    'uint32 findme, raw uint32 phonebook, raw uint32 startfrom,'
        ' raw uint32 difflist, bool allow_escapes, raw uint32 n_diffs',
    'uint32 y',
    skib_code,
    name="searchkernel_individual_binary32")


####################################################################################################

def _verify_binary_input(phonebook, findme):
    if not (phonebook.flags["C_CONTIGUOUS"] and findme.flags["C_CONTIGUOUS"]):
        raise ValueError("The modified search routine only works with C-contiguous"
                            " basis_states arrays (consider calling cupy.ascontiguousarray()"
                            " on the input arrays first).")
    if phonebook.dtype != findme.dtype:
        raise TypeError(f"Mismatched dtypes: phonebook has dtype {phonebook.dtype} and"
                            " findme has dtype {findme.dtype}.")
    if phonebook.dtype != cupy.uint32:
        raise NotImplementedError(f"dtypes {phonebook.dtype}, {findme.dtype} for phonebook"
                                        "and findme not supported.")
    if phonebook.shape[0] >= 2**32-1 or phonebook.shape[1] >= 2**16:
        raise NotImplementedError("This function will not work with arrays whose dimensions exceed"
                                    " (2**32 - 2, 2**16 - 1).")
    if phonebook.shape[1] != findme.shape[1]:
        raise ValueError("Incompatible array shapes.")


def _binary_prepare_args(findme, phonebook):
    result  = cupy.zeros(findme.shape[0], dtype=cupy.uint32)
    threads_per_block   = 256
    blocks              = numpy.ceil(findme.shape[0]/threads_per_block).astype(int)
    funcargs            = [(blocks,),
                            (threads_per_block,),
                            (findme, phonebook, cupy.uint32(phonebook.shape[0]),
                            *[cupy.uint32(i) for i in findme.shape],
                            result)]
    return funcargs


def searchsorted_binary_only(phonebook, findme, allow_escapes=False, left_check=True):
    _verify_binary_input(phonebook, findme)
    trivres     = _check_if_trivial(phonebook, findme, allow_escapes)
    if not trivres is None:
        return trivres

    funcargs    = _binary_prepare_args(findme, phonebook)

    if allow_escapes:
        if left_check:
            _searchkernel_binary_only_left_check(*funcargs)
        else:
            _searchkernel_binary_only(*funcargs)
    else:
        if left_check:
            _searchkernel_binary_only_noesc_left_check(*funcargs)
        else:
            _searchkernel_binary_only_noesc(*funcargs)
    return funcargs[-1][-1]


def searchsorted_full_binary(phonebook, findme, allow_escapes=False):
    _verify_binary_input(phonebook, findme)
    trivres     = _check_if_trivial(phonebook, findme, allow_escapes)
    if not trivres is None:
        return trivres

    funcargs    = _binary_prepare_args(findme, phonebook)

    if allow_escapes:
        _searchkernel_full_binary(*funcargs)
    else:
        _searchkernel_full_binary_noesc(*funcargs)
    return funcargs[-1][-1]

##########################################################################################
# class containing all the code needed to construct the RawKernels:

@dataclass
class _BinaryConstructorCodes:
    #############################################
    # base code with left_check
    #############################################
    binary_only_raw_check_left = r'''
        extern "C" __global__
        void METHODNAME(const unsigned int* findme,
            const unsigned int* phonebook,
            const unsigned int phonebooknum,
            const unsigned int findmenum,
            const unsigned int rowlen,
            unsigned int* result)
        {
            unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < findmenum) {

                unsigned int arrind = tid*rowlen;
                unsigned int left   = 0;
                unsigned int right  = phonebooknum;
                unsigned int m;
                unsigned int col = 0;
                unsigned int prevval;

                while (left < right) {
                    m     = (left + right) >> 1u;
                    if (phonebook[m*rowlen + col] < findme[arrind]) {
                        left    = m + 1;
                    }
                    else {
                        right   = m;
                    }
                }
                if (left >= phonebooknum) {
                    END_OF_ARRAY_TRIGGER
                }
                else if (phonebook[left*rowlen + col] != findme[arrind]) {
                    NO_MATCH_TRIGGER
                }

                prevval = phonebook[left*rowlen + col];
                ++arrind;

                for (col = 1; col < rowlen; ++col) {
                    while (phonebook[left*rowlen + col] < findme[arrind]) {
                        ++left;
                        if (left >= phonebooknum) {
                            END_OF_ARRAY_TRIGGER
                        }
                        else if (phonebook[left*rowlen + col] < phonebook[(left-1)*rowlen + col]) {
                            OVERRUN_TRIGGER
                        }
                        if (phonebook[left*rowlen + col-1] != prevval) {
                            LEFT_CHECK_TRIGGER
                        }
                    }
                    prevval = phonebook[left*rowlen + col];
                    ++arrind;
                }
                result[tid] = left;
            }
        }
        '''


    #############################################
    # base code without left_check
    #############################################
    binary_only_raw = r'''
        extern "C" __global__
        void METHODNAME(const unsigned int* findme,
                        const unsigned int* phonebook,
                        const unsigned int phonebooknum,
                        const unsigned int findmenum,
                        const unsigned int rowlen,
                        unsigned int* result)
        {
            unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < findmenum) {

                unsigned int arrind = tid*rowlen;
                unsigned int left   = 0;
                unsigned int right  = phonebooknum;
                unsigned int m;
                unsigned int col = 0;

                while (left < right) {
                    m     = (left + right) >> 1u;
                    if (phonebook[m*rowlen + col] < findme[arrind]) {
                        left    = m + 1;
                    }
                    else {
                        right   = m;
                    }
                }
                if (left >= phonebooknum) {
                    END_OF_ARRAY_TRIGGER
                }
                else if (phonebook[left*rowlen + col] != findme[arrind]) {
                    NO_MATCH_TRIGGER
                }

                ++arrind;

                for (col = 1; col < rowlen; ++col) {
                    while (phonebook[left*rowlen + col] < findme[arrind]) {
                        ++left;
                        if (left >= phonebooknum) {
                            END_OF_ARRAY_TRIGGER
                        }
                        else if (phonebook[left*rowlen + col] < phonebook[(left-1)*rowlen + col]) {
                            OVERRUN_TRIGGER
                        }
                    }
                    ++arrind;
                }
                result[tid] = left;
            }
        }
        '''


    #############################################
    # base code of full-binary search
    #############################################
    full_binary_raw = r'''
        extern "C" __global__
        void METHODNAME(const unsigned int* findme,
                        const unsigned int* phonebook,
                        const unsigned int phonebooknum,
                        const unsigned int findmenum,
                        const unsigned int rowlen,
                        unsigned int* result)
        {
            unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid < findmenum) {

                unsigned int arrind = tid*rowlen;
                unsigned int left   = 0;
                unsigned int right  = phonebooknum;
                unsigned int m;
                unsigned int col;

                while (left < right) {
                    m     = (left + right) >> 1u;

                    /* Do a multi-word comparison between phonebook[m] and findme: */

                    for (col = 0; col < rowlen; ++col) {
                        if (phonebook[m*rowlen + col] > findme[arrind + col]) {
                            /* use a magic number to indicate that phonebook[m] > findme: */
                            col = rowlen + 1;
                            break;
                        }
                        else if (phonebook[m*rowlen + col] < findme[arrind + col]) {
                            break;
                        }
                    }
                    /* if col < rowlen now, then phonebook[m] < findme */


                    if (col < rowlen) {
                        left    = m + 1;
                    }
                    else {
                        right   = m;
                    }
                }
                ASSERT_VALUE_SNIPPET
                result[tid] = left;
            }
        }
        '''

    full_binary_assert = r'''
                for (col = 0; col < rowlen; ++col) {
                    if (phonebook[left*rowlen + col] != findme[arrind + col]) {
                        printf("Error! Value %u not found in array.\n", findme[arrind]);
                        assert(0);
                    }
                }
        '''

    #############################################
    # TRIGGER responses without escapes
    #############################################
    binary_only_no_escape_EOA = r'''
                        printf("Error! Reached end of array while searching for %u.\n",
                                    findme[arrind]);
                        assert(0);
        '''

    binary_only_no_escape_NOMATCH = r'''
                        if (left > 0) {
                            printf("Error! Went from %u to %u. Searching for: %u.\n",
                                        phonebook[(left-1)*rowlen + col],
                                        phonebook[left*rowlen + col],
                                        findme[arrind]);
                        } else {
                            printf("Error! Searching for %u, but smallest entry is %u.\n",
                                        findme[arrind],
                                        phonebook[left*rowlen + col]);
                        }
                        assert(0);
        '''

    binary_only_no_escape_OVERRUN = r'''
                        printf("Error! Went from %u to %u. Searching for: %u.\n",
                                        phonebook[(left-1)*rowlen + col],
                                        phonebook[left*rowlen + col],
                                        findme[arrind]);
                        assert(0);
        '''

    binary_only_no_escape_LEFTCHE = r'''
                        printf("Error! Reached end of local block while searching for %u.\n",
                                        findme[arrind]);
                        assert(0);
        '''

    #############################################
    # TRIGGER responses with escapes
    #############################################
    binary_only_escape_EOA = r'''
                        result[tid] = 4294967295;
                        return;
        '''
    binary_only_escape_NOMATCH = binary_only_escape_EOA
    binary_only_escape_OVERRUN = binary_only_escape_EOA
    binary_only_escape_LEFTCHE = binary_only_escape_EOA

##########################################################################################

def _process_code(con_obj, methname, escapes, left_check):
    TRIGGERNAMES    = ["END_OF_ARRAY_TRIGGER", "NO_MATCH_TRIGGER", "OVERRUN_TRIGGER"]
    codenames       = ["EOA", "NOMATCH", "OVERRUN"]
    if left_check:
        basecode        = con_obj.binary_only_raw_check_left
        TRIGGERNAMES    += ["LEFT_CHECK_TRIGGER",]
        codenames       += ["LEFTCHE",]
    else:
        basecode    = con_obj.binary_only_raw

    if escapes:
        codes       = [getattr(con_obj, f"binary_only_escape_{s}") for s in codenames]
    else:
        codes       = [getattr(con_obj, f"binary_only_no_escape_{s}") for s in codenames]

    for NAME, code in zip(TRIGGERNAMES, codes):
        basecode    = basecode.replace(NAME, code)

    return basecode.replace("METHODNAME", methname)


##########################################################################################

constr_obj = _BinaryConstructorCodes()

noescname   = "binary_multibyte_noesc"
_searchkernel_binary_only_noesc = cupy.RawKernel(
    _process_code(constr_obj, noescname, escapes=False, left_check=False),
    noescname)

escname     = "binary_multibyte"
_searchkernel_binary_only = cupy.RawKernel(
    _process_code(constr_obj, escname, escapes=True, left_check=False),
    escname)

noescnameLC = "binary_multibyte_noesc_left_check"
_searchkernel_binary_only_noesc_left_check = cupy.RawKernel(
    _process_code(constr_obj, noescnameLC, escapes=False, left_check=True),
    noescnameLC)

escnameLC   = "binary_multibyte_left_check"
_searchkernel_binary_only_left_check = cupy.RawKernel(
    _process_code(constr_obj, escnameLC, escapes=True, left_check=True),
    escnameLC)

##########################################################################################

noescfullbinname    = "full_binary_noesc"
_searchkernel_full_binary_noesc = cupy.RawKernel(
    constr_obj.full_binary_raw.replace("METHODNAME", noescfullbinname).replace(
                                        "ASSERT_VALUE_SNIPPET", constr_obj.full_binary_assert),
                                    noescfullbinname)

fullbinname         = "full_binary"
_searchkernel_full_binary = cupy.RawKernel(
    constr_obj.full_binary_raw.replace("METHODNAME", fullbinname).replace(
                                        "ASSERT_VALUE_SNIPPET", ''),
                                    fullbinname)
