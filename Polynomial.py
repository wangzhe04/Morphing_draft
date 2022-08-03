import numpy as np
import sympy as sm
import re

# Function = (elements_gp + elements_gc) ** 2 * (elements_gd+elements_gc) ** 2
# eg. user specify np = 2, nc = 1, nd = 1, then gp = gp1, gp2; gc = gc1; gd = gd1
# Function = (gp1 + gp2 + gc1) ** 2 * (gd1 + gc1) ** 2

# Should work now

def expand_poly(lstp, lstd, lstc):
    return sm.expand((lstp+lstc)**2 * (lstd+lstc)**2)

def expand_g(lst):
    length = len(lst)
    res = str(lst[0])
    for i in range(length-1):
        res += " + " + str(lst[i+1])
    a = sm.expand(res)
    return a

def extract_powers(lst, n_p, n_c, n_d):

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

if __name__ == "__main__":

    n_p = 3
    n_d = 1
    n_c = 0

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

    expression_list = str(expand_poly(expand_g(list_p), expand_g(list_d), expand_g(list_c)))
    print(expression_list)

    expression_list = expression_list.split(" + ")

    print(extract_powers(expression_list, n_p, n_c, n_d))
    