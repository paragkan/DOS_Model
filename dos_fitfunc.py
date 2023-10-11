import numpy as np


def F0(X, a, b, c, e, f):
    T, t = X
    value = np.exp(a + b*T + c*t + e*(T*T) + f*t*t)
    return value
#...................................................................
def dT_F0(X, a, b, c, e, f):
    T, t = X
    value = (b + 2*e*T)*np.exp(a + b*T + c*t + e*(T*T) + f*t*t)
    return value
#...................................................................
def dt__F0(X, a, b, c, e, f):
    T, t = X
    value = (c + 2*f*t)*np.exp(a + b*T + c*t + e*(T*T) + f*t*t)
    return value
#...................................................................
def F1(X, a, b, c, d, e, f):
    T, t = X
    value = np.exp(a + b*(T) + c*t + d*(T*t) + + e*T*T + f*t*t)
    return value
#...................................................................
def F2(X, a, b, c, d, e, f, g):
    T, t = X
    value = np.exp(a + b*(T) + c*t + d*(T*t) + + e*T*T + f*t*t + g*t*t*T)
    return value
#...................................................................
def F3(X, a, b, c, d, e, f, g):
    T, t = X
    value = np.exp(a + b*(T) + c*t + d*(T*t) + + e*T*T + f*t*t + g*t*T*T)
    return value
#...................................................................
def dT_F1(X, a, b, c, d, e, f):
    T, t = X
    value = (b + d*t + 2*e*T)*np.exp(a + b*(T) + c*t + d*(T*t) + + e*T*T + f*t*t)
    return value
#...................................................................
def dt__F1(X, a, b, c, d, e, f):
    T, t = X
    value = (c + d*T + 2*f*t)*np.exp(a + b*(T) + c*t + d*(T*t) + + e*T*T + f*t*t)
    return value
#...................................................................
def G0(X, A, a, b, c, d, e, f):
    T, t = X
    vector = a*T + b*t + c + d*t*T + e*t*t + f*T*T
    val = np.exp(vector)
    value = 4*A*val/((1+val)*(1+val))
    return value
#...................................................................
def G1(X, A, a, b, c, k, j):
    T, t = X
    vector = a*(T + b*t + c)
    vector = np.exp(-k*vector)
    value = A*vector*np.exp(-j*vector)
    return value
def G2(X, A, w, xc, v, yc, u, k):
    T, t = X
    value = w/((T-xc)**2 + w**2) + v/((t-yc)**2 + v**2) + u*T + k*t
    z =  (2*A/np.pi)*(value) 
    return z
fx_list = []
fx_list.append(F0)
fx_list.append(F1)
# fx_list.append(G0)
# fx_list.append(G1)
# fx_list.append(G2)
