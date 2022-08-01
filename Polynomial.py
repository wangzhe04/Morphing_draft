import numpy as np
import sympy as sm

# Function = (elements_gp + elements_gc) ** 2 * (elements_gd+elements_gc) ** 2
# eg. user specify np = 2, nc = 1, nd = 1, then gp = gp1, gp2; gc = gc1; gd = gd1
# Function = (gp1 + gp2 + gc1) ** 2 * (gd1 + gc1) ** 2

def expand_poly(lstp, lstd, lstc):
    return sm.expand((lstp+lstc)**2 * (lstd+lstc)**2)

def expand_g(lst):
    length = len(lst)
    res = str(lst[0])
    for i in range(length-1):
        res += " + " + str(lst[i+1])
    a = sm.expand(res)
    return a

if __name__ == "__main__":

    n_p = 3
    n_d = 1
    n_c = 1

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

    print(expand_poly(expand_g(list_p), expand_g(list_d), expand_g(list_c)))