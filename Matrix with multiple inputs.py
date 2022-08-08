import numpy as np
from sympy import Poly, symbols, expand
from numpy import linalg as la
import itertools

class Morpher:

    def __init__(self, n_parameters=2):
        self.components = None
        self.n_components = None
        self.morphing_matrix = None
        self.n_benchmarks = None
        self.n_parameters = n_parameters
        self.gp = None # production
        self.gd = None # decay
        self.gs =None # combine

    # basis, each row is a benchmark, Example: g1_1 = basis[0, 0], g2_1 = basis[0, 1]
    def set_basis(self, basis_p = None, basis_d = None, basis_s = None):

        if basis_p is None and basis_d is None and basis_s is None:
            raise Exception('No basis is given')

        if basis_s is not None:  
            self.basis = basis_s
            self.gs = basis_s
            self.n_benchmarks = len(basis_s[0])

        if basis_p is not None:
            self.gp = basis_p
            self.n_benchmarks = len(basis_p[0])
        
        if basis_d is not None:
            self.gd = basis_d
            self.n_benchmarks = len(basis_d[0])

        if basis_s is not None and basis_p is not None and basis_d is not None:
            assert len(basis_p[0]) == len(basis_d[0]) == len(basis_s[0]), "the number of basis points in production, decay and combine should be the same"
            self.n_benchmarks = len(basis_p[0])
            return
        elif basis_s is not None and basis_p is not None:
            assert len(basis_p[0]) == len(basis_s[0]), "the number of basis points in production and combine should be the same"
            self.n_benchmarks = len(basis_p[0])
            return
        elif basis_s is not None and basis_d is not None:
            assert len(basis_d[0]) == len(basis_s[0]), "the number of basis points in decay and combine should be the same"
            self.n_benchmarks = len(basis_d[0])
            return
        elif basis_p is not None and basis_d is not None:
            assert len(basis_p[0]) == len(basis_d[0]), "the number of each basis points in production and decay should be the same"
            self.n_benchmarks = len(basis_p[0])
            return

    # get the minimal number of indepedent samples
    def get_Nmin(self, ns, np, nd):
        res1 = (ns*(ns+1) * (ns+2) * ((ns+3) + 4 * (np+nd)))/24
        res2 = (ns*(ns+1) * np*(np+1) + ns*(ns+1)*nd*(nd+1) + np*(np+1)*nd*(nd+1))/4 
        res3 = ns*np*nd*(ns+np+nd+3)/2
        return res1 + res2 + res3

    def calculate_morphing_matrix_multiple_coupling(self):
        n_gp = 0
        n_gd = 0
        n_gs = 0

        if self.gp is not None:
            n_gp = len(self.gp) # n_gp == n for total of gp_1 ... gp_n
        if self.gd is not None:
            n_gd = len(self.gd)
        if self.gs is not None:
            n_gs = len(self.gs)
        
        assert self.components is not None, "No components are given"

        # the first n_gd components are for gd, the next n_gp components in self.compoents are for gp, the last n_gc components are for gc
        assert (n_gp + n_gd + n_gs) == len(self.components[0]), "The number of coupling parameters in basis is not equal to the number of components"

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
                if n_gs != 0: # if gc coupling exists
                    for k in range(n_gs):
                        if n_gd != 0 and n_gp != 0: # add the length of gd and gp to index if they are not none
                            factor *= float(self.gs[k,b] ** self.components[c,k+n_gd+n_gp])
                        elif n_gd != 0:
                            factor *= float(self.gs[k,b] ** self.components[c,k+n_gd])
                        elif n_gp != 0:
                            factor *= float(self.gs[k,b] ** self.components[c,k+n_gp])
                        else:
                            factor *= float(self.gs[k,b] ** self.components[c,k])
                inv_morphing_submatrix[b, c] = factor
        # print("inv_morphing_submatrix:\n", inv_morphing_submatrix.T)
        morphing_submatrix = inv_morphing_submatrix.T
        self.matrix_before_invertion = morphing_submatrix
        # QR factorization
        q, r= np.linalg.qr(morphing_submatrix, 'complete')
        self.morphing_matrix = np.dot(np.linalg.pinv(r), q.T)
        return self.morphing_matrix


    def find_components_with_multiple_couplings(self, max_overall_power = 0, parameter_max_power = 0, Nd = 0, Np = 0, Ns = 0):
        lst = []

        if max_overall_power != 0 and parameter_max_power != 0:
            powers_each_component = [range(max_power + 1) for max_power in parameter_max_power]
            # print(powers_each_component)

            # Go through regions and finds components for each
            components = []
            for powers in itertools.product(*powers_each_component):
                powers = np.array(powers, dtype=int)
                
                if np.sum(powers) > max_overall_power:
                    continue

                if not any((powers == x).all() for x in components):
                    components.append(np.copy(powers))

            arr = np.array(components, dtype=int)

        else:
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

    # Print minimum number of samples needed
    print("minimum number of samples required:\n", morpher.get_Nmin(n_s, n_p, n_d))

    # Print the couplings
    if gd is not None:
        print("gd:\n", gd.T)
    if gp is not None:
        print("gp:\n", gp.T)
    if gs is not None:
        print("gs:\n", gs.T)

    this_components = morpher.find_components_with_multiple_couplings(Nd = n_d, Np = n_p, Ns = n_s)
    print("Powers of components:\n", this_components)
    morpher.set_basis( basis_p=gp, basis_d=gd, basis_s = gs)
    print("Matrix:\n",morpher.calculate_morphing_matrix_multiple_coupling())
    print("Condition number:\n", la.cond(morpher.morphing_matrix, 1))

 

