import numpy as np
from sympy import Poly, symbols, expand

class Morpher:

    def __init__(self, n_parameters=2):
        self.components = None
        self.n_components = None
        self.morphing_matrix = None
        self.n_benchmarks = None
        self.n_parameters = n_parameters
        self.gp = None # production
        self.gd = None # decay
        self.gc =None # combine

    # basis, each row is a benchmark, Example: g1_1 = basis[0, 0], g2_1 = basis[0, 1]
    def set_basis(self, basis_p = None, basis_d = None, basis_c = None):

        if basis_p is None and basis_d is None and basis_c is None:
            raise Exception('No basis is given')

        if basis_c is not None:  
            self.basis = basis_c
            self.gc = basis_c
            self.n_benchmarks = len(basis_c[0])

        if basis_p is not None:
            self.gp = basis_p
            self.n_benchmarks = len(basis_p[0])
        
        if basis_d is not None:
            self.gd = basis_d
            self.n_benchmarks = len(basis_d[0])

        if basis_c is not None and basis_p is not None and basis_d is not None:
            assert len(basis_p[0]) == len(basis_d[0]) == len(basis_c[0]), "the number of basis points in production, decay and combine should be the same"
            self.n_benchmarks = len(basis_p[0])
            return
        elif basis_c is not None and basis_p is not None:
            assert len(basis_p[0]) == len(basis_c[0]), "the number of basis points in production and combine should be the same"
            self.n_benchmarks = len(basis_p[0])
            return
        elif basis_c is not None and basis_d is not None:
            assert len(basis_d[0]) == len(basis_c[0]), "the number of basis points in decay and combine should be the same"
            self.n_benchmarks = len(basis_d[0])
            return
        elif basis_p is not None and basis_d is not None:
            assert len(basis_p[0]) == len(basis_d[0]), "the number of each basis points in production and decay should be the same"
            self.n_benchmarks = len(basis_p[0])
            return

    def calculate_morphing_matrix_multiple_coupling(self):
        n_gp = 0
        n_gd = 0
        n_gc = 0

        if self.gp is not None:
            n_gp = len(self.gp) # n_gp == n for total of gp_1 ... gp_n
        if self.gd is not None:
            n_gd = len(self.gd)
        if self.gc is not None:
            n_gc = len(self.gc)
        
        assert self.components is not None, "No components are given"

        # the first n_gd components are for gd, the next n_gp components in self.compoents are for gp, the last n_gc components are for gc
        assert (n_gp + n_gd + n_gc) == len(self.components[0]), "The number of coupling parameters in basis is not equal to the number of components"

        inv_morphing_submatrix = np.zeros([self.n_benchmarks, self.n_components])

        for b in range(self.n_benchmarks):
            for c in range(self.n_components):
                factor = 1.0
                if n_gd != 0: # if gd coupling exists
                    for j in range(n_gd):
                        factor *= float(self.gd[j, b] ** self.components[c, j])
                if n_gp != 0: # if gp coupling exists
                    for i in range(n_gp):
                        if n_gd != 0:
                            factor *= float(self.gp[i,b] ** self.components[c,i+n_gd] )
                        else:
                            factor *= float(self.gp[i,b] ** self.components[c,i])
                if n_gc != 0: # if gc coupling exists
                    for k in range(n_gc):
                        if n_gd != 0 and n_gp != 0: # add the length of gd and gp to index if they are not none
                            factor *= float(self.gc[k,b] ** self.components[c,k+n_gd+n_gp])
                        elif n_gd != 0:
                            factor *= float(self.gc[k,b] ** self.components[c,k+n_gd])
                        elif n_gp != 0:
                            factor *= float(self.gc[k,b] ** self.components[c,k+n_gp])
                        else:
                            factor *= float(self.gc[k,b] ** self.components[c,k])
                inv_morphing_submatrix[b, c] = factor
        # print("inv_morphing_submatrix:\n", inv_morphing_submatrix.T)
        morphing_submatrix = inv_morphing_submatrix.T
        self.matrix_before_invertion = morphing_submatrix
        # QR factorization
        q, r= np.linalg.qr(morphing_submatrix, 'complete')
        self.morphing_matrix = np.dot(np.linalg.pinv(r), q.T)
        return self.morphing_matrix


    def find_components_with_multiple_couplings(self, Nd = 0, Np = 0, Ns = 0):
        lst = []

        if Np < 0 or Nd < 0 or Ns < 0:
            print("Np, Nd, Ns must be non_negative integers")
            exit()
        elif Np == 0 and Nd == 0 and Ns == 0:
            print("both Np and Nd, or Ns alone must be positive integers")
            exit()

        gp = symbols('gp:15')
        gd = symbols('gd:15')
        gs = symbols('gs:15')

        prod = sum(gp[:Np] + gs[:Ns])  #sum of couplings in production
        dec = sum(gd[:Nd] + gs[:Ns])   #sum of couplings in decay

        if (Nd==0 and Ns==0):
            dec = 1;
        if (Np==0 and Ns==0):
            prod = 1;

        f = expand(prod**2*dec**2)  #contribution to matrix element squared
        # print("Poly:\n", f)
        mono=Poly(f).terms(); #list of tuples containing monomials

        for i in range(0, len(mono)):
            lst.append(mono[i][0])

        #array of coupligs powers in the alphabetic order gd0, gd1, ..., gp0, gp1, ..., gs0, gs1, ...
        arr = np.array(lst)

        self.components = arr
        self.n_components = len(arr)

        return arr

        



if __name__=="__main__":

    # In the order of gd, gp, gc, the code will determine the number of each coupling parameter based on gd, gp, gc...
    n_d = 0
    n_p = 0
    n_s = 2

    # specify gd, gp, gc separately
    gd = None       # np.array([[1,1,1,1,1,1]])
    gp = None       # np.array([[0.7071, 0.7071, 0.7071, 0.7071, 0.7071, 0.7071], [0, 4.2426, 0, 4.2426, -4.2426, 0], [0, 0, 4.2426, 4.2426, 0, -4.2426]])
    gs = np.array([[1,1,1,1,1], [-5, -4, -3, -2, -1]])

    # n_parameters here should equal to n_d + n_p + n_c
    morpher = Morpher(n_parameters=2)

    this_components = morpher.find_components_with_multiple_couplings(Nd = n_d, Np = n_p, Ns = n_s)
    print("Powers of components:\n", this_components)
    morpher.set_basis( basis_p=gp, basis_d=gd, basis_c = gs)
    print("Matrix:\n",morpher.calculate_morphing_matrix_multiple_coupling())

 

