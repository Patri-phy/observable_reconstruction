import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from tqdm import tqdm

######################################################
# CONSTANTS
######################################################

s_x = [[0, 1], [1, 0]]
s_y = [[0, -1j], [1j, 0]]
s_z = [[1, 0], [0, -1]]
c_n = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
c_not = np.array(c_n, dtype=complex)
sigma_x = np.array(s_x, dtype=complex)
sigma_y = np.array(s_y, dtype=complex)
sigma_z = np.array(s_z, dtype=complex)
OPERATORS = [(sigma_x, "sigma_x"), (sigma_y, "sigma_y"), (sigma_z, "sigma_z")]
n = 1   #number of initial qubits (related to the dimension of the operator)

######################################################
# INPUT
######################################################

operator_choice = sigma_z #one can choose among sigma_x, sigma_y, sigma_z
# selection of the method of unitarization
# use "ha" for Haar measure method, "gs" for Gram-Schmidt method
select_method = "ha"
n_a_range = range(2,6)
prob_sum_threshold = 0.001
norm_threshold = 0.001
n_train_states = 2000
n_test_states = 700
finite_statistics = "no"
n_trials = 200000 #it works only if finite_statistics is set to "yes"
new_dataset = 0 # 0 if the dataset must be new for every iteration vs numb. of ancillae, 1 otherwise

######################################################
# INPUT RECOGNITION
######################################################

def f_name_of_op(operator):
    for matrix, name in OPERATORS:
        if np.allclose(operator, matrix):
            return name
    raise ValueError("Unrecognized operator matrix.")

######################################################
# FUNCTIONS FOR QUANTUM CIRCUIT
######################################################

# RANDOM MATRIX GENERATION AND UNITARIZATION

def gen_random_sq_max_rank_matrix(_dim):
    """It is defined to randomly generate a _dim x _dim
    matrix with complex entries. It call itself until
    the rank is equal to _dim"""
    real_p = np.random.uniform(low = 0, high = 1, size = [_dim,_dim])
    imag_p = np.random.uniform(low = 0, high = 1, size = [_dim,_dim])
    mat = real_p + 1j*imag_p
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
    q, r = np.linalg.qr(mat) # r is an upper-triangular matrix
    d = np.diag(r) #d is the diagonal of r
    ph = d/np.absolute(d)
    q = np.multiply(q, ph) #it stores the element-wise product of q and ph in q
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
    e_list = np.vstack(e_list)
    return e_list

# RANDOM STATE GENERATION

def random_state_of_n_qubits_through_uniform_stat(n):
    """It generates a random n qubits state using uniform statistics in [0,1]"""
    _dim = 2**n
    psi_real_part = np.random.uniform(low = 0, high = 1, size = _dim)
    psi_imag_part = np.random.uniform(low = 0, high = 1, size = _dim)
    psi = psi_real_part + 1j*psi_imag_part
    one_item_list_psi = [psi]
    # one assumes that the vector can be normalized (almost certain)
    normalized_psi = _normalization(one_item_list_psi)[0]
    return normalized_psi

def random_state_of_n_qubits_through_normal_stat(n):
    """It generates a random n qubits state using normal distribution
    with mean 0 and standard deviation 1"""
    _dim_n = 2**n
    psi_real_part = np.random.randn(_dim_n)
    psi_imag_part = np.random.randn(_dim_n)
    psi = psi_real_part + 1j*psi_imag_part    
    one_item_list_psi = [psi]
    # one assumes that the vector can be normalized (almost certain)
    normalized_psi = _normalization(one_item_list_psi)[0]
    return normalized_psi

def constructor_of_global_psi(psi,_dim_na):
    """It constructs the state as a tensor product of the state psi
    (of the n qubits) and the state ket{0}^tensor n_a"""
    zeros_dim_na = np.zeros(_dim_na)
    zeros_dim_na[0] = 1
    global_psi = np.kron(psi, zeros_dim_na)
    return global_psi

def checking_state_vec(psi):
    sq_norm = np.vdot(psi,psi)
    _norm = np.sqrt(sq_norm)
    _difference = _norm - 1
    abs_difference = np.abs(_difference)
    if abs_difference <= norm_threshold:
        print(f"Deviation from 1 less than or equal to {norm_threshold}")
    else:
        print(f"Deviation from 1 greater than {norm_threshold}")

# TIME EVOLUTION

def time_evolution_of_psi(psi, unitary_matrix):
    """It evaluates the unitary evolution of psi through unitary_matrix"""
    return unitary_matrix @ psi

def time_evolution_Phi(psi, unitary_matrix):
    """It evaluates the evolution of the density matrix through unitary_matrix"""
    rho_0 = np.outer(psi,np.conj(psi)) #density matrix
    unitary_matrix_tr = unitary_matrix.transpose()
    unitary_matrix_tr_conj = unitary_matrix_tr.conj()
    rho =  unitary_matrix @ rho_0 @ unitary_matrix_tr_conj #Phi(rho)
    return rho

# PROBABILITY VECTOR EVALUATION

def probabilities_through_povm(rho,_dim):
    """It evaluates the probability vector relative to the rho through POVM"""
    probabilities_vector = []
    
    def ith_proj(i,_dim):
        """It evaluates the i-th projector of a _dim-dimensional Hilbert space"""
        matrix = np.zeros((_dim,_dim))
        matrix[i, i] = 1
        return matrix
    
    for _dim_index in range(_dim):
        _dim_index_proj = ith_proj(_dim_index,_dim) #evaluation of the i-th projector
        _dim_index_prob = np.trace(rho @ _dim_index_proj) #evaluation of the i-th probability
        probabilities_vector.append(_dim_index_prob)
    return probabilities_vector

def probabilities_squared_mod_of_comp(psi):
    """It evaluates the probabilities as square modules of the elements of psi"""
    psi_mod = np.abs(psi)
    psi_sq_mod = psi_mod**2
    return psi_sq_mod

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
            print(f"Deviation from 1 less than or equal to {prob_sum_threshold}")
        else:
            print(f"Deviation from 1 greater than {prob_sum_threshold}")
            
    return is_valid

def exp_value_through_bracket(operator,state):
	""" Expectation value of a operator through <psi|op|psi> """
	return state.conj() @ operator @ state

def exp_value_through_trace(operator,state):
    """ Expectation value of a operator through trace of op * rho """
    rho = np.outer(state,np.conj(state))
    _product = rho @ operator
    return np.trace(_product)

######################################################
# ITERATION
######################################################

def train_and_test_states_generation(n, n_train_states, n_test_states):
    """ It generates the training states and the test states """
    train_states = []
    test_states = []
    for _ in range(n_train_states):
        train_psi = random_state_of_n_qubits_through_uniform_stat(n)
        train_states.append(train_psi)
    for _ in range(n_test_states):
        test_psi = random_state_of_n_qubits_through_uniform_stat(n)
        test_states.append(test_psi)
    return train_states, test_states

def train_and_test_y_columns(operator, train_states, test_states):
    """ It evaluates the y columns of the training states and the test states """
    y_train_exp_val = []
    y_test_exp_val = []
    for train_state in train_states:
        _exp_value = exp_value_through_bracket(operator, train_state)
        y_train_exp_val.append(_exp_value)
    for test_state in test_states:
        _exp_value = exp_value_through_bracket(operator, test_state)
        y_test_exp_val.append(_exp_value)
    y_train = np.array(y_train_exp_val)
    y_test = np.array(y_test_exp_val)
    return y_train, y_test

def train_and_test_p_matrices(operator, train_states, test_states, _dim_na, unitary_matrix):
    """ It evaluates the two p matrices of the training states and the test states """

    def probabilities_from_database_states(operator, states, _dim_na, unitary_matrix):
        _dim = 2 *_dim_na
        probabilities_vector_list = []
        for state in states:
            global_state = constructor_of_global_psi(state,_dim_na)
            rho = time_evolution_Phi(global_state, unitary_matrix)
            probabilities_vector = probabilities_through_povm(rho,_dim)
            prob_checking_output = checking_prob_vec(probabilities_vector, False)
            if prob_checking_output == False:
                raise ValueError(f"The sum of the prob. components deviates from 1 more than {prob_sum_threshold}")
            probabilities_vector = np.real(probabilities_vector)
            if finite_statistics == "yes":
                probabilities_vector = np.random.multinomial(n_trials, probabilities_vector) / n_trials
            probabilities_vector_list.append(probabilities_vector)
            probabilities_vector_np = np.vstack(probabilities_vector_list)
        return probabilities_vector_np
        
    train_prob_vector_np = probabilities_from_database_states(operator, train_states, _dim_na, unitary_matrix)
    test_prob_vector_np = probabilities_from_database_states(operator, test_states, _dim_na, unitary_matrix)

    return train_prob_vector_np, test_prob_vector_np

def iterate_vs_numb_of_ancillae(n_a_range, n, n_train_states, n_test_states, operator_choice, select_method):
    """ It evaluates the MSRE and its square root for every iteration of the QELM. Each iteration has its own number of ancillae. """
    if select_method == "ha":
        f = haar_measure_unitary_m_generation
    elif select_method == "gs":
        f = gram_schmidt_unitary_m_generation
    else:
        raise ValueError(
        f"Invalid select_method: '{select_method}'. Expected 'ha' or 'gs'")
    _msre_list = []
    _sqrt_msre_list = []
    _index = 0
    for n_a in tqdm(n_a_range):
        _dim_na = 2**n_a
        _dim = 2**(n_a +1)
        if _index == 0:
            train_states, test_states = train_and_test_states_generation(n, n_train_states, n_test_states)
            y_train, y_test = train_and_test_y_columns(operator_choice, train_states, test_states)
            y_train = np.real(y_train)
            y_test = np.real(y_test)
            _index += 1 * new_dataset
        unitary_matrix = f(_dim)
        p_train, p_test = train_and_test_p_matrices(operator_choice, train_states, test_states,_dim_na, unitary_matrix)
        p_plus = np.linalg.pinv(p_train)
        w = p_plus @ y_train
        y_tilde = p_test @ w # y_tilde is the vector of predicted expectation values
        _msre, _sqrt_msre = msre_and_sqrt_of_msre_funct(y_tilde, y_test)
        _msre_list.append(float(_msre))
        _sqrt_msre_list.append(float(_sqrt_msre))
    return _msre_list, _sqrt_msre_list

######################################################
# STATISTICS
######################################################

def msre_and_sqrt_of_msre_funct(y_tilde, y_test):
    """ It evaluates the msre and its square root of y_tilde (predicted values) and y_test """
    _eps = ( y_test - y_tilde ) / y_test
    _sq_eps = _eps**2
    _msre = np.mean(_sq_eps)
    _sqrt_msre = np.sqrt(_msre)
    return _msre, _sqrt_msre

######################################################
# PLOT
######################################################

def _plot_of_sqrt_msre(
    n_a_range,
    sqrt_msre,
    n_train_states,
    n_test_states,
    select_method,
    operator_choice,
    finite_statistics,
    n_trials
    ):
    """It plots the sqrt of msre as a function of the number of ancillae """
    plt.xticks(n_a_range)
    plt.plot(n_a_range, sqrt_msre, marker='x', linewidth=1)
    plt.xlabel("number of ancillae")
    plt.grid()
    plt.ylabel("square root of MSRE")
    operator_name = f_name_of_op(operator_choice)
    title_0 = f"reconstruction of {operator_name}, "
    title_1 = f"finite statistics: {finite_statistics}, "
    if finite_statistics == "yes":
        title_1 += f"numb. of trials: {n_trials}, "
    title_2 = f"numb. of tr. and test states: ({n_train_states}, {n_test_states}), "
    title_3 = f"method of unitar.: {select_method.upper()}"
    title = title_0 + title_1 + title_2 + title_3
    plt.title(title)
    plt.show()

######################################################
# OUTPUT
######################################################

# RECAP
def output_recap(n, n_a_range, select_method, norm_threshold,
                 prob_sum_threshold, n_train_states, n_test_states,
                 operator_choice, finite_statistics, n_trials):
    print(f"Recap of inputs:")
    list_of_numb_of_ancillae = [ _ for _ in n_a_range]
    print(f"range of ancilla variables: {list_of_numb_of_ancillae}")
    print(f"selected method for unitarization: {select_method}")
    print(f"threshold of the deviation from 1 (norm): {norm_threshold}")
    print(f"threshold of the deviation from 1 (prob): {prob_sum_threshold}")
    print(f"number of training states: {n_train_states}")
    print(f"number of training states: {n_test_states}")
    chosen_operator = f_name_of_op(operator_choice)
    print(f"operator to reconstruct: {chosen_operator}")
    print(f"finite statistics: {finite_statistics}")
    if finite_statistics == "yes":
        print(f"Number of trials: {n_trials}")
    print()

def recap_of_functions():
    list_of_functions = [k for k, v in list(globals().items())
         if callable(v) and getattr(v, '__module__', None) == __name__]

    for item in list_of_functions:
        print(item)

def main():
    output_recap(n, n_a_range, select_method, norm_threshold,
                 prob_sum_threshold, n_train_states, n_test_states,
                 operator_choice, finite_statistics, n_trials)
    start = perf_counter()
    _msre_list, _sqrt_msre_list = iterate_vs_numb_of_ancillae(n_a_range, n, n_train_states, n_test_states, operator_choice, select_method)
    end = perf_counter()
    print("seconds to run the iterated QELM:")
    print(end-start)
    list_of_numb_of_ancillae = [ _ for _ in n_a_range]
    print("numb. of ancillae vs mean squared relative errors (MSREs):")
    zip_1 = zip(list_of_numb_of_ancillae,_msre_list)
    for item in zip_1:
        print(item[0])
        print(item[1])
    
    print("numb. of ancillae vs the sqrt of MSREs:")
    zip_2 = zip(list_of_numb_of_ancillae,_sqrt_msre_list)
    for item in zip_2:
        print(item[0])
        print(item[1])
    _plot_of_sqrt_msre(
    n_a_range,
    _sqrt_msre_list,
    n_train_states,
    n_test_states,
    select_method,
    operator_choice,
    finite_statistics,
    n_trials
    )

if __name__ == "__main__":
    main()



