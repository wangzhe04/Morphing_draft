# from mimetypes import init
# from re import T
import numpy as np
# from scipy import linalg

n1 = 0 # couplings equal in production and decay vetex
n2 = 1 # coupling only in production vertex
n3 = 2 # coupling only in decay vertex

# get the minimal number of indepedent samples
def get_Nmin(n1, n2, n3):
    res1 = (n1*(n1+1) * (n1+2) * ((n1+3) + 4 * (n2+n3)))/24
    res2 = (n1*(n1+1) * n2*(n2+1) + n1*(n1+1)*n3*(n3+1) + n2*(n2+1)*n3*(n3+1))/4 
    res3 = n1*n2*n3*(n1+n2+n3+3)/2
    return res1 + res2 + res3

def invert_matrix(Matrix): 
    return np.linalg.inv(Matrix) 

def pseudo_invert_matrix(matrix): # use when nbase >= nmin
    return np.dot(np.linalg.inv(np.dot(matrix.T, matrix)), matrix.T) # (A^T * A)^-1 * A^T 


def pseudo_invert_matrix_QR_decomposition(matrix): # use when nbase >= nmin
    q, r= np.linalg.qr(matrix, 'complete') # ‘complete’ : returns q, r with dimensions (…, M, M), (…, M, N), the pdf shows [m*n] = QR = [m*n][n*n], thus thought we need to change the order of Q and R
    return np.dot(np.linalg.inv(q), r.transpose()) # QR Decomposition of (A^T * A)^-1 * A^T 

def get_inv_matrix(n_benchmarks, components, n_components, basis, n_parameters):
    inv_morphing_submatrix = np.zeros([n_benchmarks, n_components])
    for b in range(n_benchmarks):
         for c in range(n_components):
            factor = 1.0
            for p in range(n_parameters): # n_parameters == 2
                factor *= float(basis[b, p] ** components[c, p]) # get value of each g1^n * g2^n
            inv_morphing_submatrix[b, c] = factor
    morphing_submatrix = inv_morphing_submatrix.T
    return morphing_submatrix


if __name__=="__main__":
    this_n_benchmarks_this_basis = 5
    this_components = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]) #powers of g1 and g2
    this_n_components = 5
    this_basis = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]])
    this_n_parameters = 2

    morphing_submatrix = get_inv_matrix(this_n_benchmarks_this_basis, this_components, this_n_components, this_basis, this_n_parameters)

    # The results comparison when n_base == n_min with QR decomposition
    print("Pseudo_inverse and standard inverse when nbase == nmin:\nPsudo:")
    print(pseudo_invert_matrix_QR_decomposition(morphing_submatrix))
    print("Standard:")
    print(invert_matrix(morphing_submatrix)) 
    print()

    # The results when n_base ==7 >= n_min with psudo_inverse 
    print("Pseudo_inverse when nbase >= nmin:\n Psudo:")
    this_n_benchmarks_this_basis = 7
    this_basis = np.array([[1, -7], [1, -6], [1, -5], [1, -4], [1, -3], [1, -2], [1, -1]])
    morphing_submatrix = get_inv_matrix(this_n_benchmarks_this_basis, this_components, this_n_components, this_basis, this_n_parameters)
    print(pseudo_invert_matrix(morphing_submatrix))
    print()
    print("Psudo with QR decomposition:")
    print(pseudo_invert_matrix_QR_decomposition(morphing_submatrix))
    # Raised Error: numpy.linalg.LinAlgError: Last 2 dimensions of the array must be square

    
    

