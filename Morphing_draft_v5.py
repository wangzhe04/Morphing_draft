#%%
import numpy as np
import matplotlib.pyplot as plt
import random
from prettytable import PrettyTable
import sympy as sm
import re


class Morpher:

    def __init__(self, n_parameters=2):
        self.components = None
        self.n_components = None
        self.basis = None
        self.morphing_matrix = None
        self.n_benchmarks = None
        self.n_parameters = n_parameters
        self.morphing_matrix_component_weights = None
        self.morphing_weights = None
        self.W_i = None
        self.matrix_before_invertion = None
        self.this_xsec = None
        self.gp = None # production
        self.gd = None # decay
        self.gc =None # combine

    # the power of each g1^n * g2^n, n_components is the number of components/coupling pairs
    def set_components(self, components):
        self.components = components
        self.n_components = len(components)

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
        # the first n_gp components in self.compoents are for gp, the next n_gd components are for gd, the last n_gc components are for gc
        assert self.components is not None, "No components are given"
        assert (n_gp + n_gd + n_gc) == len(self.components[0]), "The number of coupling parameters in basis is not equal to the number of components"

        inv_morphing_submatrix = np.zeros([self.n_benchmarks, self.n_components])
        # print(self.n_benchmarks, self.n_components, self.n_parameters)
        for b in range(self.n_benchmarks):
            for c in range(self.n_components):
                factor = 1.0
                if n_gd != 0: # if there is gd coupling
                    for j in range(n_gd):
                        factor *= float(self.gd[j, b] ** self.components[c, j])
                if n_gp != 0: # if there is gp coupling
                    for i in range(n_gp):
                        if n_gd != 0:
                            factor *= float(self.gp[i,b] ** self.components[c,i+n_gd] )
                        else:
                            factor *= float(self.gp[i,b] ** self.components[c,i])
                if n_gc != 0: # if there is gc coupling
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

    # calculate the morphing weights
    def calculate_morphing_weights(self, theta):
        component_weights = np.zeros(self.n_components)
        for c in range(self.n_components):
            factor = 1.0
            for p in range(self.n_parameters):
                factor *= float(theta[p] ** self.components[c, p])
            component_weights[c] = factor

        component_weights = np.array(component_weights)
        self.morphing_matrix_component_weights = component_weights

        self.morphing_weights = np.dot(self.morphing_matrix, component_weights)
        return self.morphing_weights

    # calculate W_i with W_i = w_i*sigma_i
    def calculate_weights_times_crossection(self, xsec):
        index = len(self.morphing_weights)
        if len(xsec) < index:
            raise Exception('The number of xsec values is smaller than the number of morphing weights')
        
        # Get the corresponding xsec values for the morphing weights    
        this_xsec = xsec[:index]
        self.this_xsec = this_xsec
        self.W_i = np.multiply(this_xsec, self.morphing_weights, dtype=np.float32)
        return self.W_i

    # calculate Neff = sum(W_i)
    def calculate_Neff(self):
        return sum(self.W_i)
    
    # calculate N_tot = sum(abs(W_i))
    def calculate_Ntot(self):
        return sum(np.abs(self.W_i))

    def calculate_Ntot_squared(self):
        return sum(self.W_i * self.W_i)

    # get basis points with ranges
    def get_predict_points(self, g1 = 1, g2_ranges = [-13, 13]):
        g2_list = list(range(g2_ranges[0], g2_ranges[1] + 1))
        
        n_basis= len(g2_list)
        basis_points = []
        for i in range(n_basis):
            g2 = g2_list[i]
            basis_point = [g1, g2]
            basis_points.append(basis_point)
        return np.array(basis_points)

    def get_predict_xsec(self, predict_points, know_xsec, this_components):
        morpher = Morpher(self.n_parameters)
        morpher.set_components(this_components)
        morpher.set_basis(basis_c= self.gc, basis_d= self.gd, basis_p= self.gp)
        morpher.calculate_morphing_matrix_multiple_coupling()

        res_xsec = []
        for i in range(len(predict_points)):
            this_point = predict_points[i]
            morpher.calculate_morphing_weights(this_point)
            this_xsec = morpher.calculate_weights_times_crossection(know_xsec)
            this_xsec = morpher.calculate_Neff()
            res_xsec.append(this_xsec)
        return np.array(res_xsec)

    # return in a list with the order corresponging to the predict points [small -> large]
    def get_Neff_Ntot(self, predict_points, know_xsec, gc, gd, gp, this_components):
        morpher = Morpher(self.n_parameters)
        morpher.set_components(this_components)
        morpher.set_basis(basis_c= gc, basis_d= gd, basis_p= gp)
        morpher.calculate_morphing_matrix_multiple_coupling()

        res_Neff_Ntot = []
        for i in range(len(predict_points)):
            this_point = predict_points[i]
            morpher.calculate_morphing_weights(this_point)
            this_xsec = morpher.calculate_weights_times_crossection(know_xsec)
            this_Neff = morpher.calculate_Neff()
            this_Ntot = morpher.calculate_Ntot()
            res_Neff_Ntot.append(this_Neff/this_Ntot)
        # print(res_Neff_Ntot)

        return np.array(res_Neff_Ntot)

    def get_Neff_Ntot_squared(self, predict_points, know_xsec, this_components):
        morpher = Morpher(self.n_parameters)
        morpher.set_components(this_components)
        morpher.set_basis(basis_c= self.gc, basis_d= self.gd, basis_p= self.gp)
        morpher.calculate_morphing_matrix_multiple_coupling()

        res_Neff_Ntot_squared = []
        for i in range(len(predict_points)):
            this_point = predict_points[i]
            morpher.calculate_morphing_weights(this_point)
            this_xsec = morpher.calculate_weights_times_crossection(know_xsec)
            this_Neff = morpher.calculate_Neff()
            this_Neff_squared = this_Neff * this_Neff
            this_Ntot_Squared = morpher.calculate_Ntot_squared()
            res_Neff_Ntot_squared.append(this_Neff_squared/this_Ntot_Squared)

        return np.array(res_Neff_Ntot_squared)

    def get_predict_points_with_range(self, sample_size = 1000, g1 = 1, g2_range = [-10, 10]):
        g2_list = []
        for i in range(sample_size):
            g2 = random.uniform(g2_range[0], g2_range[1])
            g2_list.append(g2)
        g2_list.sort()
        n_basis= len(g2_list)
        basis_points = []
        for i in range(n_basis):
            g2 = g2_list[i]
            basis_point = [g1, g2]
            basis_points.append(basis_point)
        return g2_list, np.array(basis_points)

    # add a xsec value to the input xsec list, which will not affect self.xsec.
    # This method is mainly used for adding non consecutive neff values to the xsec list.
    def add_predict_Neff(self, changing_xsec, input_xsec, add_g2_values, this_basis, n_components):

        if type(add_g2_values) == list or type(add_g2_values) == np.ndarray:
            for i in add_g2_values:
                g2_ranges = [i, i]
                changing_xsec = np.append(changing_xsec, self.get_predict_xsec(self.get_predict_points(g1 = 1, g2_ranges = g2_ranges), input_xsec, this_basis, n_components))
            res = np.array(changing_xsec)
        elif(type(add_g2_values) == int):
            g2_ranges = [add_g2_values, add_g2_values]
            res = np.append(changing_xsec, self.get_predict_xsec(self.get_predict_points(g1 = 1, g2_ranges = g2_ranges), input_xsec, this_basis, n_components))
        else:
            print("add_g2_values type error")
            res = np.array(changing_xsec)
        return res


    def expand_poly(self, lstp, lstd, lstc):
        return sm.expand((lstp+lstc)**2 * (lstd+lstc)**2)

    def expand_g(self, lst):
        length = len(lst)
        res = str(lst[0])
        for i in range(length-1):
            res += " + " + str(lst[i+1])
        a = sm.expand(res)
        return a

    def extract_powers(self, lst, n_p, n_c, n_d):
        components = np.zeros([len(lst), n_p+n_c+n_d])
        length = len(lst)

        for component in range(length):
            for i in range(n_d):
                if str("g"+str(i+1)+"_d**") in str(lst[component]):
                    m = re.search(r'g'+str(i+1)+'_d\*\*(\d+)', str(lst[component]))
                    components[component, i] = int(m.group(1))
                elif str("g"+str(i+1)+"_d") in str(lst[component]):
                    components[component, i] = 1
                else:
                    continue
            for i in range(n_p):
                if str("g"+str(i+1)+"_p**") in str(lst[component]):
                    m = re.search(r'g'+str(i+1)+'_p\*\*(\d+)', str(lst[component]))
                    components[component, i+n_d] = int(m.group(1))
                elif str("g"+str(i+1)+"_p") in str(lst[component]):
                    components[component, i+n_d] = 1
                else:
                    continue
            for i in range(n_c):
                if str("g"+str(i+1)+"_c**") in str(lst[component]):
                    m = re.search(r'g'+str(i+1)+'_c\*\*(\d+)', str(lst[component]))
                    components[component, n_p+n_d+i] = int(m.group(1))
                elif str("g"+str(i+1)+"_c") in str(lst[component]):
                    components[component, n_p+n_d+i] = 1
                else:
                    continue
        return components

    def calculate_components(self, n_d = 0, n_p = 0, n_c = 0):

        list_p = []
        list_d = []
        list_c = []

        for i in range(n_p):
            globals()["g" + str(i+1)+"_p"] = sm.Symbol("g" + str(i+1)+"_p")
            list_p.append(globals()["g" + str(i+1)+"_p"])
        for i in range(n_d):
            globals()["g" + str(i+1)+"_d"] = sm.Symbol("g" + str(i+1)+"_d")
            list_d.append(globals()["g" + str(i+1)+"_d"])
        for i in range(n_c):
            globals()["g" + str(i+1)+"_c"] = sm.Symbol("g" + str(i+1)+"_c")
            list_c.append(globals()["g" + str(i+1)+"_c"])

        if list_p == []:
            list_p = [0]
        if list_d == []:
            list_d = [0]
        if list_c == []:
            list_c = [0]

        expression_list = str(self.expand_poly(self.expand_g(list_p), self.expand_g(list_d), self.expand_g(list_c)))

        expression_list = expression_list.split(" + ")
    
        return self.extract_powers(expression_list, n_p, n_c, n_d)


                


        



if __name__=="__main__":

    n_p = 0
    n_d = 0
    n_c = 2

    morpher = Morpher(n_parameters = 2)

    this_components = morpher.calculate_components(n_d = n_d, n_p = n_p, n_c = n_c)
    print(this_components)
    xsec = np.array([0.759, 0.53, 0.4, 0.335, 0.316, 0.316, 0.328]) 
    predict_point = np.array([1, 1])

    gc=np.array([[1,1,1,1,1], [-5, -4, -3, -2, -1]])

    morpher.set_basis(basis_c=gc)
    morpher.set_components(components=this_components)
    morpher.calculate_morphing_matrix_multiple_coupling()
    morpher.calculate_morphing_weights(predict_point)
    morpher.calculate_weights_times_crossection(xsec)

    xsec_5 = morpher.get_predict_xsec(morpher.get_predict_points(g2_ranges=[-5,5]), xsec,this_components)



"""
    # this_components_1 = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]) #powers of g1 and g2
    # this_basis = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]]) # basis
    # xsec = np.array([0.759, 0.53, 0.4, 0.335, 0.316, 0.316, 0.328])

    # In the order of gd, gp, gc, the code will determine the number of each coupling parameter based on gd, gp, gc..
    n_p = 3
    n_d = 1
    n_c = 0



    gd = np.array([[1,1,1,1,1,1]])
    gp = np.array([[0.7071, 0.7071, 0.7071, 0.7071, 0.7071, 0.7071], [0, 4.2426, 0, 4.2426, -4.2426, 0], [0, 0, 4.2426, 4.2426, 0, -4.2426]])
    gc = None # np.array([[1,1,1,1,1, 1], [-5, -4, -3, -2, -1, 0]])
    
    xsec = np.array([0.515, 0.732, 0.527, 0.742, 0.354, 0.527, 0.364, 0.742, 0.364, 0.621, 0.432, 0.621, 0.432]) # define once, the code will take the corresponding xsec values for the morphing weights
    predict_point = np.array([1, 1, 1, 1] )  # change the point to predict

    morpher = Morpher(n_parameters=4)
    this_components = morpher.calculate_components(n_d = n_d, n_p = n_p, n_c = n_c)
    print(this_components)
    morpher.set_components(this_components)
    morpher.set_basis( basis_p=gp, basis_d=gd, basis_c = gc)
    print("Matrix:\n",morpher.calculate_morphing_matrix_multiple_coupling())
    print("Weights:\n", morpher.calculate_morphing_weights(predict_point))
    print("Weights Times Xsex:\n",morpher.calculate_weights_times_crossection(xsec))

    print("Neff: \n", morpher.calculate_Neff())
    print("Ntot:\n", morpher.calculate_Ntot())
    # The code below is the previous example, this_basis == gc

"""

 

#%%