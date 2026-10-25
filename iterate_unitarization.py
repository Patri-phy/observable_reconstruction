import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from tqdm import tqdm
######################################################
# INPUT
######################################################
initial_dim = 3
final_dim_plus_one = 100
dim_range = range(initial_dim, final_dim_plus_one)
n_iter = 300
n_super_iter = 30
_threshold = 4e-11
# use "yes" (or "no") to check (avoid checking) if,
# at every iteration, the obtained matrix is unitary
select_check = "yes"

# use "ha" for Haar measure method, "gs" for Gram-Schmidt method
select_method = "ha"

######################################################
# FUNCTIONS FOR GENERATION AND UNITARIZATION
######################################################
def gen_random_sq_max_rank_matrix(_dim):
    """It is defined to randomly generate a _dim x _dim
    matrix with complex entries. It call itself until the rank is equal to _dim"""
    real_p = np.random.uniform(low = 0, high = 1, size = [_dim,_dim])
    imag_p = np.random.uniform(low = 0, high = 1, size = [_dim,_dim])
    mat = (real_p + 1j*imag_p)/np.sqrt(2.0)
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
    q, r = np.linalg.qr(mat)
    d = np.diag(r)
    ph = d/np.absolute(d)
    q = np.multiply(q, ph, q)
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

######################################################
# FUNCTION FOR CHECKING IF A MATRIX IS UNITARY
######################################################

def unitary_m_check(mat, show_plot):
    """It returns True if mat is unitary, False otherwise.
    It shows the plots of u u dagger and u dagger u if show_plot is True."""
    _shape = mat.shape
    if _shape[0] != _shape[1]:
        raise TypeError("Nonsquare matrix")
    mat_tr = mat.transpose()
    mat_tr_conj = mat_tr.conj()
    mat_times_mat_dagger = mat @ mat_tr_conj
    mat_times_mat_dagger_real = np.real(mat_times_mat_dagger)
    mat_times_mat_dagger_imag = np.imag(mat_times_mat_dagger)
    if show_plot:
        plt.matshow(mat_times_mat_dagger_real)
        plt.title(r"$\Re(U U^\dagger)$")
        plt.show()
        plt.matshow(mat_times_mat_dagger_imag)
        plt.title(r"$\Im(U U^\dagger)$")
        plt.show()
    mat_dagger_times_mat = mat_tr_conj @ mat
    mat_dagger_times_mat_real = np.real(mat_dagger_times_mat)
    mat_dagger_times_mat_imag = np.imag(mat_dagger_times_mat)
    if show_plot:
        plt.matshow(mat_dagger_times_mat_real)
        plt.title(r"$\Re(U^\dagger U)$")
        plt.show()
        plt.matshow(mat_dagger_times_mat_imag)
        plt.title(r"$\Im(U^\dagger U)$")
        plt.show()

    u_u_dag_minus_I_real = np.real(mat_times_mat_dagger - np.eye(_shape[0]))
    u_u_dag_imag = np.imag(mat_times_mat_dagger)
    u_dag_u_minus_I_real = np.real(mat_dagger_times_mat - np.eye(_shape[0]))
    u_dag_u_imag = np.imag(mat_dagger_times_mat)

    ch_1 = np.abs(u_u_dag_minus_I_real) <= _threshold
    ch_2 = np.abs(u_u_dag_imag) <= _threshold
    ch_3 = np.abs(u_dag_u_minus_I_real) <= _threshold
    ch_4 = np.abs(u_dag_u_imag) <= _threshold

    ch = [ch_1,ch_2,ch_3,ch_4]

    ch = np.vstack(ch)
    return ch.all()

######################################################
# FUNCTIONS ITERATION
######################################################

def avg_exe_time_vs_dims_without_check(dim_range, n_iter, f):
    """For each dimension it evaluates the average time of execution in seconds"""
    avg_exe_times = []
    for dim_index in dim_range:
        exe_times = []
        for i_iter in range(n_iter):
            start = perf_counter()
            mat = f(dim_index)
            end = perf_counter()
            exe_times.append(end - start)
        avg_exe_time = np.mean(exe_times)
        avg_exe_times.append(avg_exe_time)
    return avg_exe_times

def avg_exe_time_vs_dims_with_check(dim_range, n_iter, f):
    """For each dimension it evaluates the average time of execution in seconds.
    At each iteration it checks if the matrix is unitary."""
    avg_exe_times = []
    for dim_index in dim_range:
        exe_times = []
        for i_iter in range(n_iter):
            start = perf_counter()
            mat = f(dim_index)
            end = perf_counter()
            check_value = unitary_m_check(mat, False)
            if check_value == False:
                print(f"Nonunitary matrix; dimension: {dim_index}; iteration {i_iter}")
            exe_times.append(end - start)
        avg_exe_time = np.mean(exe_times)
        avg_exe_times.append(avg_exe_time)
    return avg_exe_times

def iterate_avg_exe_time_vs_dims_with_check(dim_range, n_iter, f, n_super_iter):
    """It iterated the calculations of the averages. It includes the check (unitary or not)"""
    avg_exe_times_list_of_list = []
    for _ in tqdm(range(n_super_iter)):
        avg_exe_times = avg_exe_time_vs_dims_with_check(dim_range, n_iter, f)
        avg_exe_times_list_of_list.append(avg_exe_times)
    avg_times_np = np.vstack(avg_exe_times_list_of_list)
    averages = avg_times_np.sum(axis=0)/n_super_iter
    return averages

def iterate_avg_exe_time_vs_dims_without_check(dim_range, n_iter, f, n_super_iter):
    """It iterated the calculations of the averages"""
    avg_exe_times_list_of_list = []
    for _ in tqdm(range(n_super_iter)):
        avg_exe_times = avg_exe_time_vs_dims_without_check(dim_range, n_iter, f)
        avg_exe_times_list_of_list.append(avg_exe_times)
    avg_times_np = np.vstack(avg_exe_times_list_of_list)
    averages = avg_times_np.sum(axis=0)/n_super_iter
    return averages

######################################################
# PLOT
######################################################

def _plot(dim_range, exe_time_avg, select_method, select_check, n_iter, n_super_iter):
    """It plots the averages vs dimensions """
    plt.plot(dim_range, exe_time_avg, marker='x')
    plt.grid()
    plt.xlabel("Dimension")
    plt.ylabel("Average execution time (s)")
    plt.title(f"method: {select_method.upper()}, n_iter: {n_iter}, check: {select_check}, n_super_iter: {n_super_iter}")
    plt.show()

######################################################
# OUTPUT
######################################################
leftmost = dim_range[0]
rightmost = dim_range[-1]
print(f"Dimension range: {leftmost} - {rightmost}")
print(f"Number of dimensions: {len(dim_range)}")
print("Number of iterations:", n_iter)
print("Number of super iterations:", n_super_iter)
print("Method:", select_method)
print("Threshold for unitary check:", _threshold)
print()

if select_check == "yes":
    print("It checks if the matrix is unitary at every iteration")
    print(f"Threshold: {_threshold}")
else:
    print("It does not check if the matrix is unitary")

if select_method == "ha":
    f = haar_measure_unitary_m_generation
elif select_method == "gs":
    f = gram_schmidt_unitary_m_generation
else:
    raise ValueError(
        f"Invalid select_method: '{select_method}'. Expected 'ha' or 'gs')")

if select_check == "yes":
    averages = iterate_avg_exe_time_vs_dims_with_check(dim_range, n_iter, f, n_super_iter)
else:
    averages = iterate_avg_exe_time_vs_dims_without_check(dim_range, n_iter, f, n_super_iter)



print("averages:", averages)
print()

_plot(dim_range, averages, select_method, select_check, n_iter, n_super_iter)
