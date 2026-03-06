import numpy as np
from collections import Counter
# from GF_1D import G0, E_tb, Sigma_2, E_free, Sigma_4, Sigma_6
# import matplotlib.pyplot as plt

def generate_permutations(N=6):
    '''
    Generate all permutations [+,-,+,+,-,...] for N binary choices,
    corresponding to a momentum transfer by either +q or -q.
    '''
    bin_array = np.zeros((2**N, N))
    for i in range(int(2**N)):
        b = np.array(list(np.binary_repr(i)), dtype=np.int16)
        if b.size < N:
            b = np.concatenate((np.zeros(N-b.size), b))
        bin_array[i,:] = b
    pm_array = (-1)**bin_array
    return pm_array.astype(np.int16)


def select_permutations(N=6):
    '''
    Generate all permutations +,-,+,+,-,...] for N binary choices,
    and then select only those which are 'valid': those which never
    return to zero momentum in the middle of the sequence, but do 
    return to zero momentum at the end.
    '''
    selected = []
    for n in range(2, N+1, 2):
        pm_array = generate_permutations(N=n)
        for i in range(pm_array.shape[0]):
            p = pm_array[i,:]
            fail = False
            for j in range(2,n):
                if np.sum(p[:j]) == 0:
                    fail = True
            if np.sum(p) != 0:
                fail = True
            if not fail:
                selected.append(p)
    return selected


def sigma_contributions(terms):
    '''
    Generate concise representations of the terms contributing to the self-energy
    from the set of relevant permutations, corresponding to different diagrams.

    Returns:
    scalars: [s1, s2, ...]      Scalar multiplicity of each term
    ns_mat: [ns1, ns2, ...]     Indices in G0_stored of relevant terms to select. Indices are related to momentum by:
                                n        | 0  1  2   3 ...
                                momentum | q -q 2q -2q ...                 
    ms_mat: [ms1, ms2, ...]     Powers of each term. Row index m in G0_stored contains propagators to the power (m+1),
                                so the correct row is selected by slicing with (ms_mat-1). Default value of ms_mat is 0 
                                if that propagator does not appear (i.e., G0(k-nq)^0 = 1); final row of G0_stored is set 
                                to 1s so that this case is handled correctly when slicing with index -1.
                                The order of each term is given by the sum of ms_mat along axis 1 (i.e. the number of bare propagators), plus 1.
    Example: s[i] = 2, ns_mat[i,:] = [0, 2, 4, 6], ms_mat[i,:] = [3, 3, 1, 0], order N = (3 + 3 + 1) + 1 = 8
    Corresponds to self-energy contribution:
        S = 2 x V^8 x G0(k+q)^3 x G0(k+2q)^3 x G0(k+3q)
    These terms can be accessed by:
        S = 2 x V^8 x G0_stored[2,0] x G0_stored[2,2] x G0_stored[0,4] (x G0_stored[-1,6])
    '''
    k_sum = [np.cumsum(v[:-1]) for v in terms]
    counter_counts = Counter(frozenset(Counter(row).items()) for row in k_sum)

    def process(k, v):
        ns = np.array(list(dict(k).keys()))
        ms = np.array(list(dict(k).values()))
        return v, np.where(ns > 0, 2*ns - 2, -2*ns - 1), ms
    
    entries = [process(k, v) for k, v in counter_counts.items()]
    
    max_len = max(len(ns) for _, ns, _ in entries)
    num = len(entries)
    
    scalars = np.array([s for s, _, _ in entries])
    ns_mat = np.zeros((num, max_len), dtype=int)
    ms_mat = np.zeros((num, max_len), dtype=int)
    # mask   = np.zeros((num, max_len), dtype=bool)
    
    for i, (_, ns, ms) in enumerate(entries):
        L = len(ns)
        ns_mat[i, :L] = ns
        ms_mat[i, :L] = ms
        # mask[i, :L] = True

    return scalars, ns_mat, ms_mat#, mask


def build_full_sigma(N, **kwargs):
    terms = select_permutations(N)
    scalars, ns_mat, ms_mat = sigma_contributions(terms)

    def sigma(G0_stored, V=0.1, **kwargs):
        gathered = G0_stored[ms_mat-1, ns_mat, :]
        products = np.prod(gathered, axis=1)
        V_powers = V ** (ms_mat.sum(axis=1)+1)
        return np.dot(scalars * V_powers, products)
    
    return sigma

# N = 6
# perms = generate_permutations(N=N)
# print(perms)
# terms = select_permutations(N=N)
# print(terms)
# conts = sigma_contributions(terms)
# print(conts)

# sigma_new = build_full_sigma(N=N)



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