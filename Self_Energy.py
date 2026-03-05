import numpy as np
from collections import Counter
# from GF_1D import G0, E_tb, Sigma_2, E_free, Sigma_4, Sigma_6
# import matplotlib.pyplot as plt

def generate_permutations(N=6):
    bin_array = np.zeros((2**N, N))
    for i in range(int(2**N)):
        b = np.array(list(np.binary_repr(i)), dtype=np.int16)
        if b.size < N:
            b = np.concatenate((np.zeros(N-b.size), b))
        bin_array[i,:] = b
    pm_array = (-1)**bin_array
    # print(bin_array)
    # print(pm_array)
    return pm_array.astype(np.int16)

# arr = generate_permutations(N=2)
# print(arr)

def select_permutations(N=6):
    pm_array = generate_permutations(N=N)
    selected = np.zeros((1, N), dtype=np.int16)
    for i in range(pm_array.shape[0]):
        p = pm_array[i,:]
        fail = False
        for j in range(2,N):
            if np.sum(p[:j]) == 0:
                fail = True
        if np.sum(p) != 0:
            fail = True
        if not fail:
            selected = np.vstack((selected, p))
    return selected[1:,:]

# arr = select_permutations(N=2)
# print(arr)

# print(np.array([np.sum(arr[:,:i], axis=1, keepdims=False) for i in range(1,2)]).T)

def sigma_contributions(permutations):
    N = permutations.shape[1]
    k_sum = np.array([np.sum(permutations[:,:i], axis=1, keepdims=False) for i in range(1,N)]).T
    # counters = [frozenset(Counter(row).items()) for row in k_sum]
    counter_counts = Counter(frozenset(Counter(row).items()) for row in k_sum)
    result = [(dict(k), v) for k, v in counter_counts.items()]
    # print(result)
    return result

# terms = sigma_contributions(arr)

def build_sigma_N(terms, N, G0, q=1., V=0.05, **kwargs):
    compiled = []
    for d, s in terms:
        ns = np.array(list(d.keys()))
        ms = np.array(list(d.values()))
        compiled.append((s, ns, ms))
    # print(compiled)

    if 'beta' in kwargs and 'a' in kwargs:
        q = 2 * np.pi * kwargs['beta'] / kwargs['a']

    def calc_sigma(k, E, **kwargs):
        S = 0.
        for s, ns, ms in compiled:
            # Vectorised: compute G0 for all n in this term at once
            G0_vals = np.real(np.array([G0(k + n * q, E, eps=0, **kwargs) for n in ns]))
            # print(G0_vals.shape)
            S += s * np.prod(G0_vals ** ms[:, None], axis=0)
        S *= V ** N
        return S

    return calc_sigma

def build_full_sigma(N, G0, **kwargs):
    funcs = []
    for n in range(2, N+1, 2):
        # print(n)
        perms = select_permutations(n)
        # print(perms)
        terms = sigma_contributions(perms)
        # print(terms)
        funcs.append(build_sigma_N(terms, n, G0, **kwargs))
    # print(funcs)
    
    def sigma(k, E, **kwargs):
        S = 0.
        for f in funcs:
            S += f(k, E, **kwargs)
        return S
    
    return sigma


# E = 0.5
# k_vals = np.linspace(-np.pi, np.pi, 1000)
# V = 0.3

# # sigma_new = build_sigma_N(terms, N=2, G0=G0, V=V, E0=E_free)
# sigma_new = build_full_sigma(N=6, G0=G0, E0=E_free, V=V)
# S = sigma_new(k_vals, E)
# # print(S)

# S_old = np.real(Sigma_6(k_vals, E, V=V, E0=E_free))
# # print(S.dtype)
# # print(S_old.dtype)

# fig, ax = plt.subplots()
# ax.plot(k_vals, S, color='b', ls='-', label='New')
# ax.plot(k_vals, S_old, color='r', ls=':', label='Old')
# ax.set_xlabel(r'$k$')
# ax.set_ylabel(r'$\Sigma (k, E)$')
# ax.set_title(r'$E = 0.5$, $V = 0.3$')
# ax.legend()
# plt.show()