For Morphing_draft_v5

# base functions
function set_components(self, components):
    set the self.compoents and the n_components based on the components input. 
    components should be an ndarray with shape [n_components, nd+np+nc], elements in each sub-array is in the order of gd, gp, gc. 
    eg. g1d * g1p^2 * g2p^2 * g1c^2 = [[1, 2, 2, 2]]

function set_basis(self, basis_p = None, basis_d = None, basis_c = None):
    set self.gp, self.gd, self.gc with to input basis resepctively
    each basis should be an ndarray with a shape of [nd/np/nc, n_components]

function calculate_morphing_matrix_multiple_coupling(self):
    calculate the morphing matrix based on self.np, self.nd, self.nc
    n_gd, n_gp, n_gc are the number of couplings in of each type, eg, n_gd == 2 means there are 2 decay couplings. 
    The function determined the position of of each coupling types in self.components by n_gd, n_gp, and n_gc, no need to specify. 
    The function will first calculate a matrix with self.gp, self.gp, self.gc with corresponding powers of each component. 
    Then the function will return a pseudo inverse qr factorized matrix for the purpose of overdetermined morphing. 

function calculate_weights_times_crossection(self, xsec):
    calculate weights times corresponding xsec value, get W_i values that uses for calculate neff, ntot
    use self.morphing_weights * correspongdin index of xsec
    get the corresponding xsec with the length of the morphing weights.
    set the value to self.W_i
    self.W_i is a list with length of n_components, same as weights
    return self.W_i

function get_predict_xsec(self, predict_points, know_xsec, known_basis, this_components):
    calculate the neff/xsec values of the input list of predict points
    return a list of xsec values of the range of input

# The following three functions are helper functions that used for calculate Neff/Ntot, Neff^2/sum(W_i^2) for each predict point. 
function calculate_Neff(self):
    calculate and return Neff = sum(W_i)

function calculate_Ntot(self);
    calculate and return Ntot = sum(abs(W_i))

function calculate_Ntot_Squared(self)
    calculate and return sum(W_i^2)

# The following functions are used to calculate powers of polynomials based on n_d, n_p, n_c

function expand_poly(self, lstp, lstd, lstc):
    helper function that expand the mathmatical functions to get polynomials
    couplings = Expand[(Total[lstp] + Total[lstc])^2*(Total[lstd] + Total[lstc])^2];

function expand_g(self, lst):
    helper function that expand and add up each element in the list

function extract_powers(self, lst, n_p, n_c, n_d):
    helper function extract powers from each polynomials, determine the element through n_p, n_c, n_d
    return the ndarray in [n_components, gd+gp+gc]







