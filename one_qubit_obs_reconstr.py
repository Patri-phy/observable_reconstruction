import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import textwrap
from pprint import pformat

######################################################
# CONSTANTS
######################################################

SX = [[0, 1], [1, 0]]
SY = [[0, -1j], [1j, 0]]
SZ = [[1, 0], [0, -1]]
CN = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
CNOT = np.array(CN, dtype=complex)
SIGMA_X = np.array(SX, dtype=complex)
SIGMA_Y = np.array(SY, dtype=complex)
SIGMA_Z = np.array(SZ, dtype=complex)
PAULI_OBS = [(SIGMA_X, "sigma_x"), (SIGMA_Y, "sigma_y"), (SIGMA_Z, "sigma_z")]

######################################################
# INPUT
######################################################

# one qubit observable selection
observable_expr = None  # choose among "sigma_x", "sigma_y", "sigma_z" or use None to generate a random observable
observable_rng = 42 # valid only if observable_expr = None; use observable_rng = None to generate a different observable at each run

# unitarization method selection
unitar_method = "ha" # use "ha" for Haar measure method, "gs" for Gram-Schmidt method

# effPOVM selection (if effPOVM = True the prob. vector is evaluated fixing the number of outcomes)
eff_POVM_selection = False

# number of ancillae / number of outcomes range selection
n_range = range(2, 4)  # number of ancillae / number of outcomes

# check of calculations
prob_sum_threshold = 0.001
norm_threshold = 0.001

# dataset features
n_train_states = 100
n_test_states = 1000
unique_dataset = False # False if the dataset is set to be new for every new numb. of ancillae, True otherwise
normal_stat_state_gen = True #if True (False) it uses normal (uniform) distr. while generating ket components

# probability evaluation
rho_comp_POVM_formalism = True # activates for POVM built on computational basis; not valid for eff_POVM_selection = True

#iteration vs n_trials; use None to set infinite precision
n_trials_tr_vec = [int(1e3), int(1e4), int(1e5), int(1e6), None]
n_trials_test_vec = [int(1e3), int(1e4), int(1e5), int(1e6), None]

#statistics
super_iter = 10 #number of runs per each number of ancillae / per each number of outcomes

# name of plot
show_plot = True
save_plot = False #save a plot with the name using end_of_name

# name of the plotfile
end_of_name = "means_of_MSEs" # final part of the name of the file, it works only if save_plot = True

n_input_dim = 4

######################################################
# INPUT
######################################################

def print_inputs() -> str:
    """Print and return a grouped recap of the current experiment settings."""
    sections = {
        "Observable": {
            "observable_expr": observable_expr,
            "observable_rng": observable_rng,
        },
        "Circuit and measurement": {
            "unitar_method": unitar_method,
            "eff_POVM_selection": eff_POVM_selection,
            "n_range": n_range,
            "rho_comp_POVM_formalism": rho_comp_POVM_formalism,
        },
        "Validation thresholds": {
            "prob_sum_threshold": prob_sum_threshold,
            "norm_threshold": norm_threshold,
        },
        "Dataset": {
            "n_train_states": n_train_states,
            "n_test_states": n_test_states,
            "unique_dataset": unique_dataset,
            "normal_stat_state_gen": normal_stat_state_gen,
        },
        "Shot counts and sweep": {
            "n_trials_tr_vec": n_trials_tr_vec,
            "n_trials_test_vec": n_trials_test_vec,
        },
        "Statistics": {
            "super_iter": super_iter,
        },
        "Output": {
            "end_of_name": end_of_name,
            "show_plot": show_plot,
            "save_plot": save_plot,
        },
    }

    name_width = max(len(name) for settings in sections.values() for name in settings)
    lines = ["EXPERIMENT INPUTS"]

    for title, settings in sections.items():
        lines.append(f"\n[{title}]")
        for name, value in settings.items():
            lines.append(f"  {name:<{name_width}} = {pformat(value)}")

    recap = "\n".join(lines)
    print(recap)
    return recap

def match_observable_or_generate_random(operator):
    """Return a known operator's name and the input, or generate a random observable."""
    for matrix, name in PAULI_OBS:
        if np.array_equal(matrix, operator):
        #if name == operator:
            return name, operator
    print(f"generating a random one qubit operator with observable_rng: {observable_rng}")
    def random_one_qubit_observable_gen(seed=observable_rng):
        """Sample a uniform Bloch-sphere direction with measurement outcomes ±1."""
        rng = np.random.default_rng(seed)

        z = rng.uniform(-1.0, 1.0)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        radius = np.sqrt(1.0 - z**2)
        x = radius * np.cos(phi)
        y = radius * np.sin(phi)

        return np.array([
            [z,          x - 1j * y],
            [x + 1j * y, -z],
        ], dtype=complex)
    #random_operator = random_one_qubit_observable_gen_through_outer_pr(seed=observable_rng)
    random_operator = random_one_qubit_observable_gen(observable_rng)
    print("operator generated:")
    print(random_operator)
    return "random operator", random_operator

def validate_input(observable_expr, unitar_method, unique_dataset):
    """Return error in case of wrong input."""
    if observable_expr not in {"sigma_x", "sigma_y", "sigma_z", None}:
        raise ValueError(
                f"Invalid observable_expr: '{observable_expr}'. Expected 'sigma_x', 'sigma_y', 'sigma_z' or None")
    select_m_condition = (unitar_method == "ha" or unitar_method == "gs")
    if not select_m_condition:
        raise ValueError(
            f"Invalid unitar_method: '{unitar_method}'. Expected 'ha' or 'gs'")
    if not isinstance(unique_dataset, bool):
        raise TypeError(
            f"Invalid unique_dataset: '{unique_dataset}'. Expected True or False")

######################################################
# RANDOM MATRIX GENERATION AND UNITARIZATION
######################################################

def gen_random_sq_max_rank_matrix(_dim):
    """It is defined to randomly generate a _dim x _dim
    matrix with complex entries. It call itself until
    the rank is equal to _dim"""
    mat = (np.random.randn(_dim,_dim) + 1j*np.random.randn(_dim,_dim))/np.sqrt(2)
    _rank = np.linalg.matrix_rank(mat)
    if _rank == _dim:
        return mat
    else:
        print("calling again the function to get maximum rank")
        mat = gen_random_sq_max_rank_matrix(_dim)
        return mat


def haar_measure_unitary_m_generation(_dim):
    '''It returns a random unitary matrix through Haar measure method'''
    mat = gen_random_sq_max_rank_matrix(_dim)
    q, r = np.linalg.qr(mat)  # r is an upper-triangular matrix
    d = np.diag(r)  # d is the diagonal of r
    ph = d/np.absolute(d)
    # it stores the element-wise product of q and ph in q
    q = np.multiply(q, ph)
    return q


def gram_schmidt_unitary_m_generation(_dim):
    '''It returns a random unitary matrix through Gram-Schmidt method'''
    mat = gen_random_sq_max_rank_matrix(_dim)
    # orthogonalization of the rows of the matrix mat
    u_list = iter_u_k_vector(mat)
    # normalization of the vectors in u_list
    e_list = _normalization(u_list)
    return np.vstack(e_list)


def iter_u_k_vector(mat):
    """Function updating the list of u"""
    _dim = mat.shape[0]
    u_list = []
    for k_index in range(_dim):
        u_list = u_k_vector(mat, u_list, k_index, _dim)
    return u_list


def u_k_vector(mat, u_list, k_index, _dim):
    """Evaluation of the new u"""
    v = mat[k_index]
    _sum = np.zeros(_dim, dtype=complex)
    if len(u_list) != 0:
        for j_index in range(k_index):
            _sum += _projection(v, u_list[j_index])
    u_list.append(v - _sum)
    return u_list


def _projection(v, u):
    """Projection of v on u"""
    _scalar = np.vdot(u, v)/np.vdot(u, u)
    return _scalar*u


def _normalization(u_list):
    """It performs the normalization on the vectors of u_list"""
    e_list = []
    for item in u_list:
        e_list.append(item/np.sqrt(np.vdot(item, item)))
    return np.vstack(e_list)

######################################################
# RANDOM STATE GENERATION
######################################################

def random_state_of_n_qubits_through_uniform_stat(n_qubits):
    """It generates a random n qubits state using uniform statistics in [0,1]"""
    _dim = 2**n_qubits
    psi = np.random.uniform(low=0, high=1, size=_dim) + 1j*np.random.uniform(low=0, high=1, size=_dim)
    one_item_list_psi = [psi]
    # one assumes that the vector can be normalized (almost certain)
    normalized_psi = _normalization(one_item_list_psi)[0]
    return normalized_psi

def random_state_of_n_qubits_through_normal_stat(n_qubits, rng=None):
    """Generate a normalized random pure state of `n_qubits` qubits.

    The amplitudes have independent real and imaginary parts drawn from
    a standard normal distribution, N(0, 1).
    """
    # If rng is a Generator, use it. If it is None or a seed (like an integer), initialize a Generator.
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    psi = rng.standard_normal(2**n_qubits) + 1j * rng.standard_normal(2**n_qubits)
    return psi / np.linalg.norm(psi)

def constructor_of_global_psi(psi, _dim_na):
    """It constructs the state as a tensor product of the state psi
    (of the n qubits) and the state ket{0}^tensor n_a"""
    zeros_dim_na = np.zeros(_dim_na)
    zeros_dim_na[0] = 1
    return np.kron(psi, zeros_dim_na)

######################################################
# RANDOM OBSERVABLE GENERATION
######################################################

def random_one_qubit_observable_gen_through_outer_pr(observable_rng):
    """It generate one qubit observable through outer product """
    return np.outer ( random_state_of_n_qubits_through_normal_stat(1, observable_rng) ,
                      random_state_of_n_qubits_through_normal_stat(1, observable_rng) )

######################################################
# TIME EVOLUTION
######################################################

def time_evolution_of_psi(psi, unitary_matrix):
    """It evaluates the unitary evolution of psi through unitary_matrix"""
    return unitary_matrix @ psi

def time_evolution_Phi(psi, unitary_matrix):
    """Return Phi(rho) = U rho U^dagger for a pure state |psi><psi|."""
    return unitary_matrix @ np.outer(psi, psi.conj()) @ unitary_matrix.conj().T

######################################################
# PROBABILITY VECTOR EVALUATION
######################################################

def probabilities_through_computational_povm(rho):
    """It evaluates the probabilities as the diagonal elements of the density matrix"""
    return np.real(np.diag(rho))

def probabilities_squared_mod_of_comp(psi):
    """It evaluates the probabilities as square modules of the elements of psi"""
    return np.abs(psi)**2

######################################################
# CHECKING
######################################################

def checking_prob_vec(prob_vec, verbose):
    """It checks if the deviation from 1 of the sum of the elements
    of the prob. vector is higher than a fixed threshold.
    It returns True (False) if the deviation is less than or equal to
    a threshold."""
    _difference = np.sum(prob_vec) - 1
    abs_difference = np.abs(_difference)
    is_valid = abs_difference <= prob_sum_threshold

    if verbose:
        if is_valid:
            print(
                f"Deviation from 1 less than or equal to {prob_sum_threshold}")
        else:
            print(f"Deviation from 1 greater than {prob_sum_threshold}")

    return is_valid

def checking_state_vec(psi):
    sq_norm = np.vdot(psi, psi)
    _norm = np.sqrt(sq_norm)
    _difference = _norm - 1
    abs_difference = np.abs(_difference)
    if abs_difference <= norm_threshold:
        print(f"Deviation from 1 less than or equal to {norm_threshold}")
    else:
        print(f"Deviation from 1 greater than {norm_threshold}")

######################################################
# EXPECTATION VALUE
######################################################

def exp_value_through_bracket(observable, state):
    """ Expectation value of a observable through <psi|op|psi> """
    return state.conj() @ observable @ state

def exp_value_through_trace(observable, state):
    """ Expectation value of a observable through trace of op * rho """
    return np.trace( np.outer(state, np.conj(state)) @ observable )

######################################################
# CONDITION NUMBER
######################################################

def condition_numb_eval(_matrix):
    s_trunc = np.linalg.svd(_matrix, compute_uv=False) [:n_input_dim] #singular values until n_input_dim in descending order
    return s_trunc[0] / s_trunc[-1] #s_max / s_min

######################################################
# EFFECTIVE POVM
######################################################

def evalual_element_of_eff_POVM(random_POVM_element, isometry):
    """It evaluates the effective POVM element as V^dagger (I_2 kron POVM element) V """
    return isometry.conj().T @ np.kron(np.eye(2), random_POVM_element) @ isometry

def evalual_eff_POVM(random_POVM, isometry):
    """It evaluates the effective POVM starting from a Random POVM and an isometry"""
    return [evalual_element_of_eff_POVM(item, isometry) for item in random_POVM]

def probabilities_through_eff_povm(eff_POVM, rho):
    """It evaluates the probability vector (list) as trace of eff POVM element and rho"""
    return [np.trace(element @ rho) for element in eff_POVM]

def random_POVM_calc(n):
    """It generates n POVM elements with dimension n x n"""
    unitary = haar_measure_unitary_m_generation(n)
    random_POVM = []
    for i in range(n):
        random_POVM.append( np.outer( unitary[i,:],unitary[i,:].conj() ) )
    return random_POVM

######################################################
# TRAINING AND TEST
######################################################

def train_and_test_states_generation(n_train_states, n_test_states, normal_stat_state_gen):
    """ It generates the training states and the test states """
    if normal_stat_state_gen:
        f_state_gen = random_state_of_n_qubits_through_normal_stat
    else:
        f_state_gen = random_state_of_n_qubits_through_uniform_stat
    def tr_or_test_gen(n_states):
        psi_list = []
        for _ in range(n_states):
            psi_list.append(f_state_gen(1)) #n_qubits_observable = 1
        return psi_list
    train_states = tr_or_test_gen(n_train_states)
    test_states = tr_or_test_gen(n_test_states)
    return train_states, test_states

def train_and_test_y_columns(operator, train_states, test_states):
    """ It evaluates the y columns of the training states and the test states """
    def y_column(operator, states):
        y = []
        for state in states:
            y.append( exp_value_through_bracket(operator, state) )
        return np.real(y)
    return y_column(operator, train_states), y_column(operator, test_states)

def train_and_test_p_matrices_eff_POVM_and_tr_cond_numb(train_states, test_states, eff_POVM, n_trials_tr, n_trials_test):
    """Evaluates the two p matrices through an effective POVM; it also evaluate the condition number of the train prob. matrix"""
    def prob_from_database_states(states, n_trials):
        probabilities_vector_list = []
        for state in states:
            rho = np.outer(state, state.conj())
            probabilities_vector = probabilities_through_eff_povm(eff_POVM, rho)
            if not checking_prob_vec(probabilities_vector, False):
                raise ValueError(
                    f"The sum of the prob. components deviates from 1 more than {prob_sum_threshold}"
                )
            probabilities_vector = np.clip(np.real(probabilities_vector), 0.0, 1.0)
            probabilities_vector /= np.sum(probabilities_vector)
            if n_trials != None:
                probabilities_vector = np.random.multinomial(
                    n_trials, probabilities_vector) / n_trials
            probabilities_vector_list.append(probabilities_vector)
        return np.vstack(probabilities_vector_list)
    
    tr_prob_matrix = prob_from_database_states(train_states, n_trials_tr)
    
    return tr_prob_matrix, prob_from_database_states(test_states, n_trials_test),condition_numb_eval(tr_prob_matrix)

def train_and_test_p_matrices_eff_POVM(train_states, test_states, eff_POVM, n_trials_tr, n_trials_test):
    """Evaluates the two p matrices through an effective POVM."""
    def prob_from_database_states(states, n_trials):
        probabilities_vector_list = []
        for state in states:
            rho = np.outer(state, state.conj())
            probabilities_vector = probabilities_through_eff_povm(eff_POVM, rho)
            if not checking_prob_vec(probabilities_vector, False):
                raise ValueError(
                    f"The sum of the prob. components deviates from 1 more than {prob_sum_threshold}"
                )
            probabilities_vector = np.clip(np.real(probabilities_vector), 0.0, 1.0)
            probabilities_vector /= np.sum(probabilities_vector)
            if n_trials != None:
                probabilities_vector = np.random.multinomial(
                    n_trials, probabilities_vector) / n_trials
            probabilities_vector_list.append(probabilities_vector)
        return np.vstack(probabilities_vector_list)

    return prob_from_database_states(train_states, n_trials_tr), prob_from_database_states(test_states, n_trials_test)


def train_and_test_p_matrices_and_tr_cond_numb(operator, train_states, test_states, _dim_na, unitary_matrix, n_trials_tr, n_trials_test):
    """ It evaluates the two p matrices of the training states and the test states """

    def prob_from_database_states(operator, states, _dim_na, unitary_matrix, n_trials):
        _dim = 2 * _dim_na
        probabilities_vector_list = []
        for state in states:
            global_state = constructor_of_global_psi(state, _dim_na)
            
            if rho_comp_POVM_formalism:
                #evalutation through POVM:
                probabilities_vector = probabilities_through_computational_povm(
                                       time_evolution_Phi(global_state, unitary_matrix))
            else:
                #evalutation through squared modules:
                evolved_global_state = time_evolution_of_psi(global_state, unitary_matrix)
                probabilities_vector = probabilities_squared_mod_of_comp(evolved_global_state)
            
            prob_checking_output = checking_prob_vec(
                probabilities_vector, False)
            if prob_checking_output == False:
                raise ValueError(
                    f"The sum of the prob. components deviates from 1 more than {prob_sum_threshold}")
            probabilities_vector = np.real(probabilities_vector)
            if n_trials != None:
                probabilities_vector = np.random.multinomial(
                    n_trials, probabilities_vector) / n_trials
            probabilities_vector_list.append(probabilities_vector)
        return np.vstack(probabilities_vector_list)

    train_prob_vector_np = prob_from_database_states(operator, train_states, _dim_na, unitary_matrix, n_trials_tr)

    return train_prob_vector_np, prob_from_database_states(operator, test_states, _dim_na, unitary_matrix, n_trials_test), condition_numb_eval(train_prob_vector_np)

def train_and_test_p_matrices(operator, train_states, test_states, _dim_na, unitary_matrix, n_trials_tr, n_trials_test):
    """ It evaluates the two p matrices of the training states and the test states """

    def prob_from_database_states(operator, states, _dim_na, unitary_matrix, n_trials):
        _dim = 2 * _dim_na
        probabilities_vector_list = []
        for state in states:
            global_state = constructor_of_global_psi(state, _dim_na)
            
            if rho_comp_POVM_formalism:
                #evalutation through POVM:
                probabilities_vector = probabilities_through_computational_povm(
                                       time_evolution_Phi(global_state, unitary_matrix))
            else:
                #evalutation through squared modules:
                evolved_global_state = time_evolution_of_psi(global_state, unitary_matrix)
                probabilities_vector = probabilities_squared_mod_of_comp(evolved_global_state)
            
            prob_checking_output = checking_prob_vec(
                probabilities_vector, False)
            if prob_checking_output == False:
                raise ValueError(
                    f"The sum of the prob. components deviates from 1 more than {prob_sum_threshold}")
            probabilities_vector = np.real(probabilities_vector)
            if n_trials != None:
                probabilities_vector = np.random.multinomial(
                    n_trials, probabilities_vector) / n_trials
            probabilities_vector_list.append(probabilities_vector)
        return np.vstack(probabilities_vector_list)

    train_prob_vector_np = prob_from_database_states(
        operator, train_states, _dim_na, unitary_matrix, n_trials_tr)
    test_prob_vector_np = prob_from_database_states(
        operator, test_states, _dim_na, unitary_matrix, n_trials_test)

    return train_prob_vector_np, test_prob_vector_np


######################################################
# ITERATION VS ANCILLAE / OUTCOMES
######################################################

def iterate_vs_numb_of_ancillae(n_range, n_train_states, n_test_states, observable_expr, unitar_method, n_trials_tr, n_trials_test):
    """ It evaluates the metrics for every iteration of the QELM. Each iteration has its own number of ancillae. """
    if unitar_method == 'ha':
        f = haar_measure_unitary_m_generation
    if unitar_method == 'gs':
        f = gram_schmidt_unitary_m_generation

    _msre_list = []; _sqrt_msre_list = []; _mse_list = []; _con_numb_list = []
    _index = 0
    for n_a in n_range:
        _dim_na = 2**n_a; _dim = 2**(n_a + 1)
        if _index == 0:
            train_states, test_states = train_and_test_states_generation(
                n_train_states, n_test_states, normal_stat_state_gen)
            y_train, y_test = train_and_test_y_columns(
                observable_expr, train_states, test_states)
            _index += 1 * unique_dataset
        p_train, p_test, tr_cond_numb = train_and_test_p_matrices_and_tr_cond_numb(
            observable_expr, train_states, test_states, _dim_na, f(_dim), n_trials_tr, n_trials_test)
        y_tilde = p_test @ ( np.linalg.pinv(p_train) @ y_train ) # y_tilde is the vector of predicted expectation values
        _msre, _sqrt_msre, _mse = metrics(y_tilde, y_test)
        _msre_list.append(_msre); _sqrt_msre_list.append(_sqrt_msre); _mse_list.append(_mse); _con_numb_list.append(tr_cond_numb)
    return _msre_list, _sqrt_msre_list, _mse_list, _con_numb_list


def iterate_vs_numb_of_outcomes(n_range, n_train_states, n_test_states, observable_expr, n_trials_tr, n_trials_test):
    """ It evaluates the metrics for every iteration of the QELM. Each iteration has its own number of outcomes. """
    _msre_list = []; _sqrt_msre_list = []; _mse_list = []; _con_numb_list = []
    _index = 0
    def isometry_generation(n):
        """It generates a 2n x 2 isometry"""
        return haar_measure_unitary_m_generation(2*n)[:,:2]
    for n in n_range:
        isometry = isometry_generation(n)
        random_POVM = random_POVM_calc(n)
        eff_POVM = evalual_eff_POVM(random_POVM, isometry)
        if _index == 0:
            train_states, test_states = train_and_test_states_generation(
                n_train_states, n_test_states, normal_stat_state_gen)
            y_train, y_test = train_and_test_y_columns(
                observable_expr, train_states, test_states)
            y_train = np.real(y_train)
            y_test = np.real(y_test)
            _index += 1 * unique_dataset
        p_train, p_test, tr_cond_numb = train_and_test_p_matrices_eff_POVM_and_tr_cond_numb(train_states, test_states, eff_POVM, n_trials_tr, n_trials_test)
        #p_plus = np.linalg.pinv(p_train)
        #w = p_plus @ y_train
        #y_tilde = p_test @ w  # y_tilde is the vector of predicted expectation values
        y_tilde = p_test @ ( np.linalg.pinv(p_train) @ y_train )
        _msre, _sqrt_msre, _mse = metrics(y_tilde, y_test)
        _msre_list.append(_msre); _sqrt_msre_list.append(_sqrt_msre); _mse_list.append(_mse); _con_numb_list.append(tr_cond_numb)
    return _msre_list, _sqrt_msre_list, _mse_list, _con_numb_list


######################################################
# SUPER ITERATIONS
######################################################

def super_iteration(super_iter, n_range, n_train_states, n_test_states,
                    observable_expr, unitar_method, eff_POVM_selection,
                    n_trials_tr, n_trials_test):
    if eff_POVM_selection:
        f_iteration = iterate_vs_numb_of_outcomes
        args = (n_range, n_train_states, n_test_states, observable_expr, n_trials_tr, n_trials_test)
    else:
        f_iteration = iterate_vs_numb_of_ancillae
        args = (n_range, n_train_states, n_test_states, observable_expr, unitar_method, n_trials_tr, n_trials_test)

    _mse_list_of_lists = []
    _cond_numb_list_of_lists = []
    for _index in range(super_iter):
        _msre_list, _sqrt_msre_list, _mse_list, _con_numb_list = f_iteration(*args)
        _mse_list_of_lists.append(_mse_list)
        _cond_numb_list_of_lists.append(_con_numb_list)

    return np.vstack(_mse_list_of_lists), np.vstack(_cond_numb_list_of_lists)

######################################################
# ITERATION VS FINITE STAT PRECISION
######################################################

def super_iteration_vs_trials_vec(super_iter, n_range, n_train_states,
                                     n_test_states, observable_expr,
                                     unitar_method, eff_POVM_selection,
                                     n_trials_tr_vec, n_trials_test_vec,):
    """Runs super_iteration across paired (train, test) shot-count configurations."""
    mse_np_list_of_lists = []
    cond_numb_list_of_lists = []
    start = perf_counter()
    for idx in tqdm(range(len(n_trials_tr_vec))):
        mse_np_list, cond_numb_list = super_iteration(super_iter, n_range, n_train_states,
                                      n_test_states, observable_expr,
                                      unitar_method, eff_POVM_selection,
                                      n_trials_tr_vec[idx], n_trials_test_vec[idx],)   
        mse_np_list_of_lists.append(mse_np_list)
        cond_numb_list_of_lists.append(cond_numb_list)
    end = perf_counter()
    print(f"execution time: {end - start}")
    return np.array(mse_np_list_of_lists), np.array(cond_numb_list_of_lists)


######################################################
# STATISTICS
######################################################


def metrics(y_tilde, y_test):
    """ It evaluates the mse, the msre and sqrt(msre) of y_tilde (predicted values) and y_test """
    deviation = y_test - y_tilde
    _msre = np.mean ( ( deviation / y_test ) ** 2 )
    _mse = np.mean ( deviation ** 2 )
    return float(_msre), float(np.sqrt(_msre)), float(_mse)

def reconstruction_metrics_alternative(y_pred, y_true):
    residual = np.asarray(y_pred) - np.asarray(y_true)
    mse = np.mean(np.abs(residual)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residual))
    max_abs_error = np.max(np.abs(residual))
    return mse, rmse, mae, max_abs_error

def super_iteration_stats(np_vec):
    _means = np.mean(np_vec, axis = 1)
    _stds = np.std(np_vec, axis = 1)
    _mins = np.min(np_vec, axis = 1)
    _maxs = np.max(np_vec, axis = 1)
    return _means, _stds, _mins, _maxs

######################################################
# PLOT
######################################################

def _plot_of_means_of_mse(super_iter, n_range, n_train_states, n_test_states,
                          operator_name, unitar_method, eff_POVM_selection,
                          n_trials_tr_vec, n_trials_test_vec, _means,
                          show_plot, save_plot, filename, end_of_name) :
    """It plots the means of the MSEs vs number of ancillae / number of outcomes """    
    labels = [str((tr, te)) for tr, te in zip(n_trials_tr_vec, n_trials_test_vec, strict=True)]
    for row, label in zip(_means, labels):
        plt.plot(n_range, row, marker='x', linewidth=1, label=label)
    plt.yscale('log')
    plt.xticks(n_range)
    plt.ylabel("mean of MSEs")
    plt.xlabel("numb. of outcomes") if eff_POVM_selection else plt.xlabel("numb. of ancillae")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(title="Numb. of trials (tr, te)") #, title_fontsize=12)
    t1 = f"reconstruction of {operator_name}, numb. of superit.: {super_iter}, "
    t2 = f"numb. of (tr.,test) states: ({n_train_states}, {n_test_states})\n"
    t3 = f"unitar. method: {unitar_method}, "
    t4 = "using qudits as a reservoir" if eff_POVM_selection else "using ancillae"
    title = t1 + t2 + t3 + t4
    plt.title(title)
    #plt.tight_layout()
    if show_plot: plt.show()
    if save_plot: plt.savefig(filename+f"{end_of_name}.png") 

def _plot_of_means_of_cond_numb(super_iter, n_range, n_train_states, n_test_states,
                                operator_name, unitar_method, eff_POVM_selection,
                                n_trials_tr_vec, n_trials_test_vec, _means_cond_numb, ):
    """It plots the means of the condition number vs number of ancillae / number of outcomes """    
    labels = [str((tr, te)) for tr, te in zip(n_trials_tr_vec, n_trials_test_vec, strict=True)]
    for row, label in zip(_means_cond_numb, labels):
        plt.plot(n_range, row, marker='x', linewidth=1, label=label)
    plt.yscale('log')
    plt.xticks(n_range)
    plt.ylabel("mean of condition numbers")
    plt.xlabel("numb. of outcomes") if eff_POVM_selection else plt.xlabel("numb. of ancillae")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(title="Numb. of trials (tr, te)") #, title_fontsize=12)
    t1 = f"reconstruction of {operator_name}, numb. of superit.: {super_iter}, "
    t2 = f"numb. of (tr.,test) states: ({n_train_states}, {n_test_states})\n"
    t3 = f"unitar. method: {unitar_method}, "
    t4 = "using qudits as a reservoir" if eff_POVM_selection else "using ancillae"
    title = t1 + t2 + t3 + t4
    plt.title(title)
    #plt.tight_layout()
    plt.show()
    #if show_plot: plt.show()
    #if save_plot: plt.savefig(filename+f"{end_of_name}.png") 

######################################################
# OUTPUT
######################################################

def recap_of_functions():
    print("functions defined in this .py")
    list_of_functions = [k for k, v in list(globals().items())
                         if callable(v) and getattr(v, '__module__', None) == __name__]

    for item in list_of_functions:
        print(item)

def output_print(_means, _stds, _mins, _maxs):
    print("means of sqrt of MSREs:")
    print(_means)
    print("stds of sqrt of MSREs:")
    print(_stds)
    print("mins of sqrt of MSREs:")
    print(_mins)
    print("maxs of sqrt of MSREs:")
    print(_maxs)

# FILE CREATION AND WRITING

def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

######################################################
# MAIN
######################################################

def main():
    validate_input(observable_expr, unitar_method, unique_dataset)
    filename = timeStamped(end_of_name) # in case of datafile / plotfile generation
    operator_name, operator = match_observable_or_generate_random(observable_expr)
    print_inputs()
    mse_np, cond_numb_np = super_iteration_vs_trials_vec(super_iter, n_range, n_train_states,
                                              n_test_states, operator,
                                              unitar_method, eff_POVM_selection,
                                              n_trials_tr_vec, n_trials_test_vec,)
    
    _means_mse, _stds_mse, _mins_mse, _maxs_mse = super_iteration_stats(mse_np)
    _means_cond_numb, _stds_cond_numb, _mins_cond_numb, _maxs_cond_numb = super_iteration_stats(cond_numb_np)

    _plot_of_means_of_mse(super_iter, n_range, n_train_states, n_test_states,
                          operator_name, unitar_method, eff_POVM_selection,
                          n_trials_tr_vec, n_trials_test_vec, _means_mse,
                          show_plot, save_plot, filename, end_of_name)
    
    _plot_of_means_of_cond_numb(super_iter, n_range, n_train_states, n_test_states,
                                operator_name, unitar_method, eff_POVM_selection,
                                n_trials_tr_vec, n_trials_test_vec, _means_cond_numb, )

if __name__ == "__main__":
    main()


