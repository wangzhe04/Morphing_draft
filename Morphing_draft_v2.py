#%%
import numpy as np
import matplotlib.pyplot as plt
import random
from prettytable import PrettyTable


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

    # the power of each g1^n * g2^n, n_components is the number of components/coupling pairs
    def set_components(self, components):
        self.components = components
        self.n_components = len(components)

    # basis, each row is a benchmark, Example: g1_1 = basis[0, 0], g2_1 = basis[0, 1]
    def set_basis(self, basis):
        self.basis = basis
        self.n_benchmarks = len(basis)

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
        # print(self.W_i)
        return sum(np.abs(self.W_i))

    def calculate_Ntot_squared(self):
        return sum(self.W_i * self.W_i)

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

    def get_predict_xsec(self, predict_points, know_xsec, known_basis, this_components):
        morpher = Morpher()
        morpher.set_components(this_components)
        morpher.set_basis(known_basis)
        morpher.calculate_morphing_matrix()

        res_xsec = []
        for i in range(len(predict_points)):
            this_point = predict_points[i]
            morpher.calculate_morphing_weights(this_point)
            this_xsec = morpher.calculate_weights_times_crossection(know_xsec)
            this_xsec = morpher.calculate_Neff()
            res_xsec.append(this_xsec)
        return np.array(res_xsec)

    # return in a list with the order corresponging to the predict points [small -> large]
    def get_Neff_Ntot(self, predict_points, know_xsec, known_basis, this_components):
        morpher = Morpher()
        morpher.set_components(this_components)
        morpher.set_basis(known_basis)
        morpher.calculate_morphing_matrix()

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

    def get_Neff_Ntot_squared(self, predict_points, know_xsec, known_basis, this_components):
        morpher = Morpher()
        morpher.set_components(this_components)
        morpher.set_basis(known_basis)
        morpher.calculate_morphing_matrix()

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





if __name__=="__main__":
    # The code below shows teh w_i, xsec, and W_i for the given
    this_components = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]) #powers of g1 and g2
    this_basis = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]]) # basis, each row is a benchmark
    xsec = np.array([0.759, 0.53, 0.4, 0.335, 0.316, 0.316, 0.328]) # define once, the code will take the corresponding xsec values for the morphing weights
    predict_point = np.array([1, -4] ) # change the point to predict

    morpher = Morpher()
    morpher.set_components(this_components)
    morpher.set_basis(this_basis)
    morpher.calculate_morphing_matrix()
    morpher.calculate_morphing_weights(predict_point)

    # Predict cross-section values with nbase = 5
    xsec_5 = morpher.get_predict_xsec(morpher.get_predict_points(), xsec, this_basis, this_components)

    # used to check if neff/ntot is correct
    morpher.calculate_weights_times_crossection(xsec)
    # print(morpher.calculate_Neff())
    # print(morpher.calculate_Ntot())
    # print(morpher.calculate_Ntot_squared())

    # print("Predict cross-section values with nbase = 5 \n", xsec_5)


    # Change the basis to 7 points, [1, -5], [1, -4], [1, -3], [1, -2], [1, -1], [1, 0], [1, 1]
    this_components = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [0, 4]]) #powers of g1 and g2
    this_basis_7 = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1], [1, 0], [1, 1]]) # basis, each row is a benchmark
    this_basis_5 = np.array([[1, -5], [1, -4], [1, -3], [1, -2], [1, -1]])
    
    # Predict cross-section values with nbase = 7
    xsec_7 = morpher.get_predict_xsec(morpher.get_predict_points(), xsec, this_basis_7, this_components)
    # print("Predict cross-section values with nbase = 7 \n", xsec_7)

    xsec_simulated = np.zeros(len(xsec_7))

    xsec_difference_5_7 = abs(xsec_7 - xsec_5)

    # Specify the Column Names while initializing the Table
    myTable = PrettyTable(["g2", "simulated", "n_base = 5", "n_base = 7"])
    g2_list = list(range(-13, 14))
    for i in range(len(xsec_7)):
        myTable.add_row([g2_list[i], xsec_simulated[i], xsec_5[i], xsec_7[i]])

    print(myTable)



    """
    # Code below plots the Neff/Ntot vs g2 for the 5 and 7 basis points

    """
    g2_ranges = [-13, 13]
    predict_points_g2, g2_points = morpher.get_predict_points_with_range(sample_size = 10000, g1 = 1, g2_range = g2_ranges)
    neff_ntot_5 = morpher.get_Neff_Ntot(g2_points, xsec, this_basis_5, this_components)
    neff_ntot_7 = morpher.get_Neff_Ntot(g2_points, xsec, this_basis_7, this_components)

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,10))

    ax1.plot(predict_points_g2, neff_ntot_7, color = 'red', label = 'nbase = 7')
    ax1.plot(predict_points_g2, neff_ntot_5, color = 'blue', label = 'nbase = 5')
    ax1.legend()
    ax1.set_xticks(range(-13, 13, 1))
    ax1.set_yticks(np.linspace(0.0, 1.0, 11))
    ax1.set_title("Comparison of Neff/Ntot with nbase = 5 and nbase = 7, nsample = 10000")

    ax1.set_xlabel('g2')
    ax1.set_ylabel("Neff/Ntot")

    ax2.plot(predict_points_g2, neff_ntot_7, color = 'red', label = 'nbase = 7')
    ax2.legend()
    ax2.set_xticks(range(-13, 13, 1))
    ax2.set_yticks(np.linspace(0.0, 1.0, 11))
    ax2.set_xlabel('g2, nbase = 7')
    ax2.set_ylabel("Neff/Ntot")

    ax3.plot(predict_points_g2, neff_ntot_5, color='blue', label='nbase = 5')
    ax3.legend()
    ax3.set_xticks(range(-13, 13, 1))
    ax3.set_yticks(np.linspace(0.0, 1.0, 11))
    ax3.set_xlabel('g2, nbase = 5')
    ax3.set_ylabel("Neff/Ntot")
    plt.show()


    """
    # Code below plots the Neff^2/sum(W_i^2) vs g2 for the 5 and 7 basis points

    """

    g2_ranges = [-13, 13]
    predict_points_g2, g2_points = morpher.get_predict_points_with_range(sample_size = 10000, g1 = 1, g2_range = g2_ranges)
    neff_ntot_5 = morpher.get_Neff_Ntot_squared(g2_points, xsec, this_basis_5, this_components)
    neff_ntot_7 = morpher.get_Neff_Ntot_squared(g2_points, xsec, this_basis_7, this_components)

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,10))

    ax1.plot(predict_points_g2, neff_ntot_7, color = 'red', label = 'nbase = 7')
    ax1.plot(predict_points_g2, neff_ntot_5, color = 'blue', label = 'nbase = 5')
    ax1.legend()
    ax1.set_xticks(range(-13, 13, 1))
    ax1.set_yticks(np.linspace(0.0, 2.0, 11))
    ax1.set_title("Comparison of Neff^2/sum(W_i^2) with nbase = 5 and nbase = 7, nsample = 10000")

    ax1.set_xlabel('g2')
    ax1.set_ylabel("Neff/Ntot")

    ax2.plot(predict_points_g2, neff_ntot_7, color = 'red', label = 'nbase = 7')
    ax2.legend()
    ax2.set_xticks(range(-13, 13, 1))
    ax2.set_yticks(np.linspace(0.0, 2.0, 11))
    ax2.set_xlabel('g2, nbase = 7')
    ax2.set_ylabel("Neff/Ntot")

    ax3.plot(predict_points_g2, neff_ntot_5, color='blue', label='nbase = 5')
    ax3.legend()
    ax3.set_xticks(range(-13, 13, 1))
    ax3.set_yticks(np.linspace(0.0, 2.0, 11))
    ax3.set_xlabel('g2, nbase = 5')
    ax3.set_ylabel("Neff/Ntot")
    plt.show()

#%%
