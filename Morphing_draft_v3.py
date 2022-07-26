import numpy as np
import matplotlib.pyplot as plt
import random
from prettytable import PrettyTable


class Morpher:

    def __init__(self, n_parameters):
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
        self.input_xsec = None

    # the power of each g1^n * g2^n, n_components is the number of components/coupling pairs
    def set_components(self, components):
        self.components = components
        self.n_components = len(components)

    # basis, each row is a benchmark, Example: g1_1 = basis[0, 0], g2_1 = basis[0, 1]
    def set_basis(self, basis):
        self.basis = basis
        self.n_benchmarks = len(basis)

    def set_xsec(self, xsec):
        self.input_xsec = xsec

    def calculate_morphing_matrix(self):
        inv_morphing_submatrix = np.zeros([self.n_benchmarks, self.n_components])
        for b in range(self.n_benchmarks):
            for c in range(self.n_components):
                factor = 1.0
                for p in range(self.n_parameters): # n_parameters == 2
                    factor *= float(self.basis[b, p] ** self.components[c, p]) # get value of each g1^n * g2^n
                inv_morphing_submatrix[b, c] = factor
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


    """
        The code below involve calling one or more functions above
        each function creates a new object and takes in all inputs in order to be flexible with changing inputs and won't affect the original object values
        Can be modifed to use the original object values and reduce inputs
    """

    def get_predict_xsec(self, predict_points, know_xsec, known_basis, this_components):
        morpher = Morpher(self.n_parameters)
        morpher.set_components(this_components)
        morpher.set_basis(known_basis)
        morpher.calculate_morphing_matrix()

        if(type(predict_points != np.ndarray) or type(predict_points != list)):
            predict_points = np.array(predict_points)

        res_xsec = []
        for i in range(len(predict_points)):
            this_point = predict_points[i]
            morpher.calculate_morphing_weights(this_point)
            this_xsec = morpher.calculate_weights_times_crossection(know_xsec)
            this_xsec = morpher.calculate_Neff()
            res_xsec.append(this_xsec)
        return np.array(res_xsec)

    # return in a list with the order corresponging to the predict points [small -> large]
    def get_Neff_Ntot(self, predict_points):

        res_Neff_Ntot = []
        for i in range(len(predict_points)):
            this_point = predict_points[i]
            self.calculate_morphing_weights(this_point)
            this_xsec = self.calculate_weights_times_crossection(self.input_xsec)
            this_Neff = self.calculate_Neff()
            this_Ntot = self.calculate_Ntot()
            res_Neff_Ntot.append(this_Neff/this_Ntot)
        # print(res_Neff_Ntot)

        return np.array(res_Neff_Ntot)

    def get_Neff_Ntot_squared(self, predict_points):

        res_Neff_Ntot_squared = []
        for i in range(len(predict_points)):
            this_point = predict_points[i]
            morpher.calculate_morphing_weights(this_point)
            this_xsec = morpher.calculate_weights_times_crossection(self.input_xsec)
            this_Neff = morpher.calculate_Neff()
            this_Neff_squared = this_Neff * this_Neff
            this_Ntot_Squared = morpher.calculate_Ntot_squared()
            res_Neff_Ntot_squared.append(this_Neff_squared/this_Ntot_Squared)

        return np.array(res_Neff_Ntot_squared)

    def multiple_coupling_input_neff(self, list_of_couplings):

        n_couplings = len(list_of_couplings)
        res = []

        for i in range(n_couplings):
            couplings = list_of_couplings[i]
            neff_i = []
            for point in couplings:
                self.calculate_morphing_weights(point)
                this_xsec = self.calculate_weights_times_crossection(self.input_xsec)
                this_Neff = self.calculate_Neff()
                this_Ntot = self.calculate_Ntot()
                neff_i.append(tuple(("Point: ", point, "Neff: ", this_Neff)))
            res.append(neff_i)

        return np.array(res, dtype=object)

    def multiple_coupling_input_ntot(self, list_of_couplings):

        n_couplings = len(list_of_couplings)
        res = []

        for i in range(n_couplings):
            couplings = list_of_couplings[i]
            ntot_i = []
            for point in couplings:
                self.calculate_morphing_weights(point)
                this_xsec = self.calculate_weights_times_crossection(self.input_xsec)
                this_Neff = self.calculate_Neff()
                this_Ntot = self.calculate_Ntot()
                ntot_i.append(tuple(("point: ", point, "Ntot: ",this_Ntot)))
            res.append(ntot_i)

        return np.array(res)



if __name__=="__main__":
    """
    The code blow compare xsex value with n_base=5 and n_base=7 as well as simulated values. 
    """

    # The code below shows teh w_i, xsec, and W_i for the given range of parameters
    this_components = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]) #powers of g1 and g2
    this_basis_5 = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]]) # basis
    this_basis_7 = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1], [1, 0], [1,1]]) # basis
    xsec = np.array([0.759, 0.53, 0.4, 0.335, 0.316, 0.316, 0.328]) # define once, the code will take the corresponding xsec values for the morphing weights
    predict_point = np.array([1, -10] ) # change the point to predict
    predict_points_list = [[1, -5], [1, -4], [1, -3], [1, -2], [1, -1], [1, 0], [1, 1]], [[2, -5], [2, -4], [2, -3], [2, -2], [2, -1], [1,0], [1,1]]# define the points to predict

    morpher = Morpher(n_parameters=2)
    morpher.set_basis(this_basis_7)
    morpher.set_components(this_components)
    morpher.set_xsec(xsec)
    morpher.calculate_morphing_matrix()
    morpher.calculate_morphing_weights(predict_point)
    morpher.calculate_weights_times_crossection(xsec)

    print(morpher.multiple_coupling_input_neff(predict_points_list))
    # print(xsec_7)

    # print(np.array(predict_points_list))

    # xsec_7 = morpher.get_predict_xsec(predict_points_list, xsec, this_basis_7, this_components)

    # print(morpher.get_Neff_Ntot(predict_points_list))
    # print(morpher.get_Neff_Ntot_squared(predict_points_list))