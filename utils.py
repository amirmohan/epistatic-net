import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import r2_score
import ipdb

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from scipy.special import comb

def crop_to_window(u, n):
    n0 = u.shape[1]
    b = int((n0-n)//2)
    return u[:, b:b+n]

def blocks(repeat_length, num_bits=13):
    if repeat_locations == 0:
        return [np.zeros(num_bits)]
    
    num_blocks = num_bits - repeat_length + 1
    patterns = []
    for i in range(0, num_blocks):
        pattern = np.zeros(num_bits)
        pattern[i:i+repeat_length] = 1
            
        patterns.append(pattern)
        
    return patterns

def left_right_blocks(repeat_length, num_bits=13):
    if repeat_length == 0:
        return [np.zeros(num_bits)]
    
    left_boundary = 0
    right_boundary = int(num_bits//2)
    
    num_blocks = right_boundary - repeat_length + 1
    
    patterns = []
    
    for i in range(0, num_blocks):
        for j in range(0, num_blocks):
            pattern = np.zeros(num_bits)
            pattern[i:i+repeat_length] = 1
            pattern[right_boundary+j:right_boundary+j+repeat_length] = 1
            
            patterns.append(pattern)
            
    
    return patterns

def repeat_locations(repeat_length, num_bits=13):
    patterns = []
    for i in range(0,repeat_length+1):
        new_patterns = left_right_blocks(i, num_bits)
        patterns += new_patterns
    return patterns




def support_augmented_with_reversed(support):
    support_set = set()

    for s in support:
        support_set.add(zo_to_string(s))

    support_set_temp = set()
    for s in support_set:
        s_reversed = s[::-1]

        if s_reversed not in support_set:
            support_set_temp.add(s_reversed)
            
    support_set = support_set | support_set_temp
    
    support = []
    for s in support_set:
        support.append(string_to_zo(s))
    
    return support

def next_string_with_same_num_ones(v):
    t = (v | (v-1))+ 1
    w = t | ((( (t & -t) // (v & -v) ) >> 1) - 1 )
    return w


def all_strings_with_k_ones(bit_length,k):
    num_total = int( comb(bit_length,k) )
    c = 2**k - 1
    my_list = []
    for i in range(num_total):
        my_list.append(c)
        if i != num_total - 1:
            c = next_string_with_same_num_ones(c)
        
    return my_list

def all_strings_up_to_k_ones(bit_length,k):
    my_list = []
    
    for i in range(k+1):
        my_list = my_list + all_strings_with_k_ones(bit_length,i)
        
    return my_list

def all_strings_with_given_ones(bit_length, k_list):
    my_list = []
    
    for i in k_list:
        my_list = my_list + all_strings_with_k_ones(bit_length,i)
        
    return my_list

def synthetic_band_support(band_width, num_bits=13):
    max_number = 2**(2*band_width)
    rotate_length = num_bits//2 + band_width
    support = []
    for i in range(max_number):
        binary_loc = dec_to_bin(i, num_bits)
        binary_loc = np.roll(binary_loc, rotate_length)
        support.append(binary_loc)
    return support
    
def synthetic_band_support_capped_degree(band_width, degree_cap, num_bits=13):
    assert band_width >= 0, "width needs to be non-negative"
    assert degree_cap >= 0, "cap needs to be non-negative"
    
    rotate_length = num_bits//2 + band_width
    
    support = []
    
    if isinstance(degree_cap, list):
        all_strings = all_strings_with_given_ones(2*band_width, degree_cap)
    else:
        all_strings = all_strings_up_to_k_ones(2*band_width, degree_cap)
    
    for s in all_strings:
        binary_loc = dec_to_bin(s, num_bits)
        binary_loc = np.roll(binary_loc, rotate_length)
        support.append(binary_loc)
    
    return support


def support_to_set(support):
    support_set = set()
    for s in support:
        support_set.add(zo_to_string(s))
    return support_set

def set_to_support(the_set):
    locations = []
    for loc in the_set:
        locations.append(string_to_zo(loc))
    return locations

def pm_to_zo(pm):
    """
    Goes from plus-minus to zero-one
    """
    zo = np.zeros_like(pm)
    zo[pm < 0] = 1
    return zo.astype(int)

def zo_to_pm(zo):
    """
    Goes from plus-minus to zero-one
    """
    return (-1)**zo

def zo_to_string(u):
    return ''.join([str(i) for i in list(u)])

def string_to_zo(u):
    return np.array([int(i) for i in list(u)])

def my_string_format(s):
    N = len(s)
    return s[:N//2] + ':' + s[N//2:]

def my_print_string(s):
    print(my_string_format(s))

def random_binary_matrix(m, n, p=0.5):
    A = np.random.binomial(1,p,size=(m,n))
    return A

def dec_to_bin(x, num_bits):
    assert x < 2**num_bits, "number of bits are not enough"
    u = bin(x)[2:].zfill(num_bits)
    u = list(u)
    u = [int(i) for i in u]
    return np.array(u)

def bin_to_dec(x):
    n = len(x)
    c = 2**(np.arange(n)[::-1])
    return c.dot(x)

def get_sampling_index(x, A, p=0):
    """
    x: sampling index
    A: subsampling matrix
    p: delay
    """
    num_bits = A.shape[0]
    x = dec_to_bin(x, num_bits)
    r = x.dot(A) + p
    return r % 2

def get_random_binary_string(num_bits, p=0.5):
    a = np.random.binomial(1,p,size=num_bits)
    return a

def random_delay_pair(num_bits, target_bit):
    """
    num_bits: number of bits
    location_target: the targeted location (q in equation 26 in https://arxiv.org/pdf/1508.06336.pdf)
    """
    e_q = 2**target_bit
    e_q = dec_to_bin(e_q, num_bits)
    random_seed = get_random_binary_string(num_bits)
    return random_seed, (random_seed+e_q)%2

def make_delay_pairs(num_pairs, num_bits):
    z = []
    z.append(dec_to_bin(0,num_bits))
    for bit_index in range(0, num_bits):
        for pair_idx in range(num_pairs):
            a,b = random_delay_pair(num_bits, bit_index)
            z.append(a)
            z.append(b)
    return z

def myfwht(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    # x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N == 1:
        return x
    else:
        X_even = myfwht(x[0:(N//2)])
        X_odd = myfwht(x[(N//2):])
        return np.concatenate([(X_even + X_odd),
                               (X_even - X_odd)])


    
    
def results_to_measurements(results):
    measurement_matrix = np.zeros_like(results)

    for i in range(results.shape[0]):
        measurement_matrix[i] = myfwht(results[i])
    return measurement_matrix

def get_delay_index_base(bit_index, delay_index, D):
    return 1+ 2*D*bit_index + 2*delay_index

def estimate_location(u, num_bits=13, num_delays_per_bit=8):
    location = []
    for bit in range(num_bits):
        sign_total = 0
        for delay in range(num_delays_per_bit):
            delay_0 = get_delay_index_base(bit, delay, num_delays_per_bit)
            delay_1 = delay_0 + 1

            r0 = u[delay_0]
            r1 = u[delay_1]

            sign_total += np.sign(r0)*np.sign(r1)
        location.append(sign_total/num_delays_per_bit)
    location = np.array(location)
    # this is because we need to have bits ordered correctly
    location = location[::-1]
    return location

def get_bin_energies(measurement_matrix):
    return  np.mean(measurement_matrix**2,axis=0)

def evaluate_location(u):
    return np.min(np.abs(u))
    
def location_to_bin(A, loc):
    """
    Returns the bin where a location is hashed to
    A: sub-sampling matrix
    loc: location of the tone
    """
    hashed_bin = A.dot(loc) % 2
    return hashed_bin 


def make_system(support, sampling_locations, y, N=None):
    """
    Make WHT system (of the form Ax = y) from the samples and output
    support:             signal support as binary array
    sampling locations:  sampling locations
    y:                   output
    N:                   number of samples to use
    """
    if N is not None:
        N_max = sampling_locations.shape[0]
        if N < N_max:
            chosen_indices = np.random.choice(N_max, N, replace=False)
            sampling_locations = sampling_locations[chosen_indices]
            y = y[chosen_indices]
    M = sampling_locations.dot(support.T)
    M = M % 2
    M = (-1)**M
    return (M, y)

def train_it(support, U, y, reg, N=None):
    M, y = make_system(np.vstack(support), U, np.reshape(y,[-1]), N=N)
    reg.fit(M, y)
    return reg

def get_signature(loc, delays):
    return (-1)**(delays.dot(loc) % 2)    
    
    
class SparseWHTModel:
    def __init__(self, support, coef_):
        self.support = support
        self.coef_ = coef_
    
    def predict(self, x):
        M = x.dot(self.support.T)
        M = M % 2
        M = (-1)**M
        return M.dot(self.coef_)
    
    def get_coef(self, zo):
        '''
        returns the coefficient corresponding to the location given with the zo array
        '''
        support_str = [zo_to_string(i) for i in self.support]
        target_str = zo_to_string(zo)
        try:
            index = support_str.index(target_str)
            return self.coef_[index]
        except:
            return 0
    

    
    
    
def get_candidate_locations(measurement_matrix, sampling_matrix=None):
    """
    measurement_matrix:
    sampling_matrix:    if sampling matrix is given check if the found location
                        indeed aliases to that given location
    """
    locations = []
    evaluations = []

    # go over the bins and get the location they predict
    #for i in tqdm(range(measurement_matrix.shape[1])):
    for i in range(measurement_matrix.shape[1]):
        # this is the estimated location
        location_hat = estimate_location(measurement_matrix[:, i])

        if sampling_matrix is not None:
            aliased_bin = bin_to_dec(location_to_bin(sampling_matrix, pm_to_zo(location_hat)))
            if aliased_bin == i:
                locations.append(location_hat)
                evaluations.append(evaluate_location(location_hat))
        else:
            locations.append(location_hat)
            evaluations.append(evaluate_location(location_hat))

    return np.array(locations), np.array(evaluations)


class SampleGenerator:
    def __init__(self):
        pass
    
    def set_model(self, model):
        self.model = model
        
    def get(self, x):
        pass    
    
def generate_all_codes(num_bits):
    support = []
    for s in range(2**num_bits):
        binary_loc = dec_to_bin(s, num_bits)
        support.append(binary_loc)
    return support

def make_system_simple(support, sampling_locations):
    """
    Make measurement system from the samples
    and output
    support:             signal support as binary array
    sampling locations:  sampling locations
    N:                   number of samples to use
    """
    M = sampling_locations.dot(support.T)
    M = M % 2
    M = (-1)**M
    return M
    
class SPRIGHT:
    def __init__(self, experiment_type, run_list, model_to_remove=None):
        self.experiment_type = experiment_type
        self.run_list = run_list
        self.model_to_remove = model_to_remove
        
    def set_train_data(self, U_train, y_train):
        self.U_train = U_train
        self.y_train = y_train
    
    def get_run(self, run_number, get_sampling_locations=False):

        sampling_matrix  = pickle.load(open('N13/sampling-matrix-{}.p'.format(run_number) ,'rb'))
        delays_matrix = pickle.load(open('N13/delays-{}.p'.format(run_number),'rb'))
            
        if get_sampling_locations:
            all_sampling_locations = pickle.load(open('N13/sampling-locations-{}.p'.format(run_number), 'rb'))
        else:
            all_sampling_locations = None
            
        if self.model_to_remove is not None:
            if all_sampling_locations is None:
                all_sampling_locations = pickle.load(open('N13/sampling-locations-{}.p'.format(run_number),'rb'))
            
            r_list = []
            for sampling_locations_for_delay in all_sampling_locations:
                # TODO: incorporate model here
                
                
                a = []
                for item in sampling_locations_for_delay:
                    #print(bin_to_dec(item))
                    a.append(self.y_train[bin_to_dec(item)])
                r_list.append(np.asarray(a))
             
                #with torch.no_grad():
                     #a = self.model_to_remove(torch.from_numpy(sampling_locations_for_delay).float())
                #a = a.numpy().flatten()
                #r_list.append(a)
                
            
            
            A = np.vstack(r_list)
            #A = np.asarray(r_list)
            ins_result = A
            #ins_result = ins_result - A
            
#             print(np.shape(sampling_matrix))
#             print(np.shape(delays_matrix))
#             print(np.shape(ins_result))
#             print(np.shape(all_sampling_locations))
                                        
        return sampling_matrix, delays_matrix, ins_result, all_sampling_locations
        
    
    def get_all_locations(self, use_sampling_matrix=False):
        set_of_locs = set()

        for i in self.run_list:
            run_number = i
            sampling_matrix, delays_matrix, ins_result, all_sampling_locations = self.get_run(run_number)

            measurement_matrix = np.zeros_like(ins_result)

            for i in range(ins_result.shape[0]):
                measurement_matrix[i] = myfwht(ins_result[i])


            if use_sampling_matrix:
                locs, evals = get_candidate_locations(measurement_matrix, sampling_matrix)
            else:
                locs, evals = get_candidate_locations(measurement_matrix)

            likely_indices = evals > 0

            #print('num likely indices: {}'.format(sum(likely_indices)))

            locs  = locs[likely_indices]
            evals = evals[likely_indices]

            locs = pm_to_zo(locs)

            for loc in locs:
                set_of_locs.add(zo_to_string(loc))

        locs = []
        for loc in set_of_locs:
            locs.append(string_to_zo(loc))
        return locs
    
    def get_run_lists(self):
        A_list = []
        D_list = []
        M_list = []

        for run_number in self.run_list:
            A, D, R, _ = self.get_run(run_number)
            M = results_to_measurements(R)

            A_list.append(A)
            M_list.append(M)
            D_list.append(D)

        self.A_list = A_list 
        self.M_list = M_list
        self.D_list = D_list
        
    def initial_run(self, N=int(3e5)):
        found_support = self.get_all_locations(use_sampling_matrix=True)
        if np.size(found_support) == 0:
            return False
        else:
            support = np.array(found_support)
            reg = LinearRegression(fit_intercept=False)
            reg = train_it(support, self.U_train, self.y_train, reg, N)
            model = SparseWHTModel(np.array(support), reg.coef_)
            self.model = model 
            return True
        
    def get_all_locations2(self, M_list):
        """
        data_indices:    which data runs to use to get locations
        experiment_type: which experiment (ins/frame)
        use_sampling_matrix:    if set to True, then while finding the singletons, 
                                we check if that found singleton would have hashed to
                                that bin where it was found.  In the case of noise
                                the estimated location might be wrong hence this might be
                                false
        """
        set_of_locs = set()

        # go over each stage
        for M_dictionary, A in zip(M_list, self.A_list):
            # the locations found at the stage
            locations = []
            evaluations = []

            # go through all the bins in the stage
            for bin_index, bin_measurement in M_dictionary.items():
                location_hat = estimate_location(bin_measurement)
                aliased_bin = bin_to_dec(location_to_bin(A, pm_to_zo(location_hat)))

                if aliased_bin == bin_index:
                    locations.append(location_hat)
                    evaluations.append(evaluate_location(location_hat))

            evaluations = np.array(evaluations)
            locations = np.array(locations)

            likely_indices = evaluations > 0

            locations = locations[likely_indices]
            locations = pm_to_zo(locations)

            for loc in locations:
                set_of_locs.add(zo_to_string(loc))

        locations = []
        for loc in set_of_locs:
            locations.append(string_to_zo(loc))

        return locations
    
    
    def peel_once(self):
        """
        Peels all of the support once from the measurements
        model: the model that holds the support and the values
        A_list: sampling matrix list
        M_list: measurement matrices
        D_list: delays
        """

        # initialize the list of dictionaries to return
        residual_measurements = []
        for i in range(len(self.A_list)):
            residual_measurements.append(dict())

        # go over the support
        for s in self.model.support:
            # this is the coefficient
            v = self.model.get_coef(s)*(2**10)

            stage = 0
            # go over each stage
            for A, M, D in zip(self.A_list, self.M_list, self.D_list):
                # the bin where the support goes
                found_bin_binary = location_to_bin(A, s)
                found_bin_decimal = int(bin_to_dec(found_bin_binary))

    #             if recovered_bin_decimal == 0:
    #                 ipdb.set_trace()

                # the signature that the location generates
                signature = get_signature(s, np.array(D))
                q = v*signature

                # the residual after we peel the support
                residual = M[:, found_bin_decimal] - q

                residual_measurements[stage][found_bin_decimal] = residual

                stage += 1
        return residual_measurements 
    
    
    def peel_rest(self, num_iter_upper_bound=5, N=int(3e5)):
        self.get_run_lists()
        
        is_done = False
        
        counter = 0

        # this contains the singletons recovered in each new round
        diff_sets = []
        old_locations_set = support_to_set(self.model.support)
        diff_sets.append(old_locations_set)

        while not is_done:
            #print('-----')
            #print('running: {}'.format(counter))

            residual_measurements = self.peel_once()
            new_locations = self.get_all_locations2(residual_measurements)

            if not new_locations:
                is_done = True
            else:
                # these are the locations found from the residual measurements
                new_locations_set = support_to_set(new_locations)


                # these are the locations that had been used to create the 
                # residual ameasurements
                old_locations_set = support_to_set(self.model.support)

                # lets see if we have found anything new from the peeling process
                diff_set = new_locations_set.difference(old_locations_set)

                # append the set of newly found singletons
                diff_sets.append(diff_set)

                #print('number of new locations: {}'.format(len(diff_set)))


                combined_support_set = old_locations_set | new_locations_set
                support = set_to_support(combined_support_set)

                #print('train the system')
                reg = LinearRegression(fit_intercept=False)
                reg = train_it(support, self.U_train, self.y_train, reg, N)

                self.model = SparseWHTModel(np.array(support), reg.coef_)

                counter += 1

                if counter >= num_iter_upper_bound:
                    is_done = True
                if not diff_set:
                    is_done = True
                    