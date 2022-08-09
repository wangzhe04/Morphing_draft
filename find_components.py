import numpy as np
import itertools

def find_components(max_overall_power=4, parameter_max_power=2):
    """
    Finds the components, i.e. the individual terms contributing to the squared matrix element.

    Parameters
    ----------
    max_overall_power : int, optional
        The maximal sum of powers of all parameters contributing to the squared matrix element.
        Typically, if parameters can affect the couplings at n vertices, this number is 2n. Default value: 4.

    Returns
    -------
    components : ndarray
        Array with shape (n_components, n_parameters), where each entry gives the power with which a parameter
        scales a given component.
    """


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

    res_components = np.array(components, dtype=int)

    return res_components


if __name__=="__main__":
    print(find_components(max_overall_power=4,parameter_max_power=[4,4]))