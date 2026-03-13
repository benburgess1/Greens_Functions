import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def E_free(k, **kwargs):
    return k**2


def v_free(E, **kwargs):
    return 2*np.sqrt(np.abs(E))


def E_tb(k, t=1, a=1, **kwargs):
    return -2*t*np.cos(k*a)


def v_tb(E, t=1, a=1, **kwargs):
    if np.abs(E) > 2*t:
        return 0
    k = np.arccos(E/(-2*t)) / a
    return 2*t*a*np.sin(k*a)


def G0(E, E0_vals,
    #    k, E0=E_free, 
       eps=1e-4, **kwargs):
    # if E0_vals is None:
    #     E0_vals = E0(k, **kwargs)
    return 1 / (E - E0_vals + 1j*eps)
    # return 1 / (E - E0(k, **kwargs) + 1j*eps)


def G(E, E0_vals, G0_stored=None, Sigma=None, Sigma_func=None, eps=1e-4, 
    #   k,
      **kwargs):
    if Sigma is None:
        if Sigma_func is None:
            Sigma = 0
        else:
            Sigma = Sigma_func(G0_stored, **kwargs)
    return G0(E - Sigma, E0_vals, eps=eps, **kwargs)


def Sigma_2(G0_stored, V=0.05, 
            # q=1, G0_stored=None, 
            **kwargs):
    # if G0_stored is None:
    #     # print('triggered')
    #     if 'beta' in kwargs and 'a' in kwargs:
    #         q = 2 * np.pi * kwargs['beta'] / kwargs['a']
    #     G0_p1 = G0(k+q, E, eps=0, **kwargs)
    #     G0_m1 = G0(k-q, E, eps=0, **kwargs)
    # else:
    #     # print('Triggered')
    G0_p1, G0_m1 = G0_stored[0,:2,:]
    return np.abs(V)**2 * (G0_p1 + G0_m1)


def Sigma_4(G0_stored, V=0.05, **kwargs):
    # if 'beta' in kwargs and 'a' in kwargs:
    #     q = 2 * np.pi * kwargs['beta'] / kwargs['a']
    # S2 = Sigma_2(k, E, V=V, q=q, **kwargs)
    # S4 = np.abs(V)**4 * (G0(k-q, E, eps=0, **kwargs)**2 * G0(k-2*q, E, eps=0, **kwargs)
    #                      + G0(k+q, E, eps=0, **kwargs)**2 * G0(k+2*q, E, eps=0, **kwargs))
    # S4 += np.abs(V)**4 * (G0(k-G, E, eps=0)**2 * G0(k, E, eps=0)
    #                       + G0(k+G, E, eps=0)**2 * G0(k, E, eps=0))
    S2 = Sigma_2(G0_stored, V=V, **kwargs)
    G0_p1, G0_m1, G0_p2, G0_m2 = G0_stored[0,:4,:]
    G0_p12, G0_m12 = G0_stored[1,:2,:]
    S4 = np.abs(V)**4 * (G0_p12 * G0_p2 + G0_m12 * G0_m2)
    return S2 + S4


def Sigma_6(G0_stored, V=0.05, **kwargs):
    # if 'beta' in kwargs and 'a' in kwargs:
    #     q = 2 * np.pi * kwargs['beta'] / kwargs['a']
    # S4 = Sigma_4(k, E, V=V, q=q, **kwargs)
    # S6 = np.abs(V)**6 * (G0(k-q, E, eps=0, **kwargs)**2 * G0(k-2*q, E, eps=0, **kwargs)**2 * G0(k-3*q, E, eps=0, **kwargs)
    #                      + G0(k+q, E, eps=0, **kwargs)**2 * G0(k+2*q, E, eps=0, **kwargs)**2 * G0(k+3*q, E, eps=0, **kwargs)
    #                      + G0(k-q, E, eps=0, **kwargs)**3 * G0(k-2*q, E, eps=0, **kwargs)**2
    #                      + G0(k+q, E, eps=0, **kwargs)**3 * G0(k+2*q, E, eps=0, **kwargs)**2)
    S4 = Sigma_4(G0_stored, V=V, **kwargs)
    G0_p1, G0_m1, G0_p2, G0_m2, G0_p3, G0_m3 = G0_stored[0,:6,:]
    G0_p12, G0_m12, G0_p22, G0_m22 = G0_stored[1,:4,:]
    G0_p13, G0_m13 = G0_stored[2,:2,:]
    S6 = np.abs(V)**6 * (G0_p12 * G0_p22 * G0_p3
                         + G0_p13 * G0_p22 
                         + G0_m12 * G0_m22 * G0_m3
                         + G0_m13 * G0_m22)
    return S4 + S6


def calc_dos(E_vals, k_vals, save=False, save_filename='Data.npz', **kwargs):
    G_vals = G(k_vals[:, None], E_vals[None, :], **kwargs)
    dos_vals = np.sum(np.imag(G_vals), axis=0) * (-1/np.pi)
    if save:
        np.savez(save_filename, E_vals=E_vals, dos_vals=dos_vals, k_vals=k_vals, **kwargs)
    return dos_vals


def Sigma_2_2Component(k, E, V1=0.05, V2=0.025, q1=1, q2=1/np.sqrt(2), **kwargs):
    S2_1 = np.abs(V1)**2 * (G0(k-q1, E, eps=0) + G0(k+q1, E, eps=0))
    S2_2 = np.abs(V2)**2 * (G0(k-q2, E, eps=0) + G0(k+q2, E, eps=0))
    return S2_1 + S2_2


def Sigma_4_2Component(k, E, V1=0.05, V2=0.025, q1=1, q2=1/np.sqrt(2), **kwargs):
    # Second order terms
    S2 = Sigma_2_2Component(k, E, V1=V1, V2=V2, q1=q1, q2=q2, **kwargs)
    S4 = 0
    # Cross-coupling terms
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            S4 += (np.abs(V1)**2 * np.abs(V2)**2 
                   * G0(k+s1*q1, E, eps=0)**2 * G0(k+s1*q1+s2*q2, E, eps=0))
            S4 += (np.abs(V1)**2 * np.abs(V2)**2 
                   * G0(k+s1*q2, E, eps=0)**2 * G0(k+s1*q2+s2*q1, E, eps=0))
            S4 += (2 * np.abs(V1)**2 * np.abs(V2)**2 
                   * G0(k+s1*q1, E, eps=0) * G0(k+s1*q1+s2*q2, E, eps=0) * G0(k+s2*q2, E, eps=0))
    # 2G coupling terms
    S4 += np.abs(V1)**4 * (G0(k-q1, E, eps=0)**2 * G0(k-2*q1, E, eps=0)
                         + G0(k+q1, E, eps=0)**2 * G0(k+2*q1, E, eps=0))
    S4 += np.abs(V2)**4 * (G0(k-q2, E, eps=0)**2 * G0(k-2*q2, E, eps=0)
                         + G0(k+q2, E, eps=0)**2 * G0(k+2*q2, E, eps=0))
    return S2 + S4


def adaptive_params(E, v0=v_free, A=0.1, B=1, eps_min=1e-5, L_max=1e5, L_min=1e4, 
                    print_warnings=True, fix_epsilon=False, eps_fix=1e-4,
                    calc_L=True, **kwargs):
    # E = np.abs(E)
    # Choose epsilon based on upper bound set by E
    if not fix_epsilon:
        eps = A * np.abs(E)
    else:
        # print('fixed')
        eps = eps_fix
    if eps < eps_min:
        if print_warnings:
            print(f'\nWarning: eps = {eps:.3g} < eps_min = {eps_min:.3g}. Setting eps = eps_min.')
        eps = eps_min
    if not calc_L:
        return eps
    # Choose L based on lower bound set by E and epsilon
    L = B * 2 * np.pi * np.abs(v0(E, **kwargs)) / eps
    if L > L_max:
        if print_warnings:
            print(f'Warning: L = {L:.3g} > L_max = {L_max:.3g}. Setting L = L_max.')
        L = L_max
    elif L < L_min:
        if print_warnings:
            print(f'Warning: low value L = {L} < L_min = {L_min:.3g}. Setting L = L_min.')
        L = L_min
    return eps, L
        

def generate_k_vals(L, k_max=None, n=2, C=0.1, q=1, **kwargs):
    dk = 2 * np.pi / L
    if k_max is None:
        k_max = (n/2 + C) * q
    k_vals = np.arange(-k_max, k_max, dk)
    return k_vals


def calc_dos_adaptive(E_vals, save=False, save_filename='Data.npz', E0=E_tb, 
                      q=1., N=2, **kwargs):
    dos_vals = np.zeros_like(E_vals)
    if 'beta' in kwargs and 'a' in kwargs:
            q = 2 * np.pi * kwargs['beta'] / kwargs['a']
    for i, E in enumerate(tqdm(E_vals)):
        eps, L = adaptive_params(E, **kwargs)       # Set adaptive parameters
        k_vals = generate_k_vals(L, **kwargs)
        E0_vals = E0(k_vals, **kwargs)
        # Generate 'building blocks' G0(k+nq)^m (powers of bare propagators)
        idxs = []
        for j in range(1, int(N/2)+1):
            idxs.append(j)
            idxs.append(-j)
        idxs = np.array(idxs)
        G0_stored = np.array([np.real(G0(E, E0(k_vals[None,:] + idxs[:,None] * q, **kwargs), eps=0, **kwargs))])
        for n in range(2, N+1, 2):
            G0_new = np.zeros((1,*G0_stored.shape[1:]))
            G0_new[0,:N-n,:] = G0_stored[-1,:N-n,:] * G0_stored[0,:N-n,:]
            G0_stored = np.concatenate((G0_stored, G0_new))
        # Set final row to ones; corresponds to G0(k+nq)^0, i.e. gives a location to point to when a 
        # particular bare propagator is not involved in a given self-energy term
        G0_stored = np.concatenate((G0_stored, np.ones((1, *G0_stored.shape[1:]))))
        G_vals = G(E, E0_vals, G0_stored=G0_stored, eps=eps, **kwargs)        # Calculate GF; self-energy passed in kwargs as sigma_func
        dos_vals[i] = np.sum(np.imag(G_vals)) * (-1/np.pi) / L      # Calculate DoS from GF
    if save:
        print('Saving -> ' + save_filename)
        save_kwargs = {k: v for k, v in kwargs.items() if callable(v) is False}
        np.savez(save_filename, E_vals=E_vals, dos_vals=dos_vals, k_vals=k_vals, **save_kwargs)
    return dos_vals


def calc_Sigma_recursive(g_mat, V=0.1, **kwargs):
    N = int((g_mat.shape[0] - 1) / 2)
    v = np.abs(V)**2
    Sigma_plus = v / g_mat[2*N,:]
    for i in range(N-1):
        Sigma_plus = v / (g_mat[2*N-i-1,:] - Sigma_plus)
    Sigma_minus = v / g_mat[0,:]
    for i in range(N-1):
        Sigma_minus = v / (g_mat[i+1,:] - Sigma_minus)
    return Sigma_plus + Sigma_minus


def calc_dos_recursive(E_vals, save=True, save_filename='Data.npz', E0=E_tb, 
                       q=1., N=2, L=1e6, fix_k=False, **kwargs):
    dos_vals = np.zeros_like(E_vals)
    if 'beta' in kwargs and 'a' in kwargs:
            q = 2 * np.pi * kwargs['beta'] / kwargs['a']
    k_vals = generate_k_vals(L, **kwargs)
    Ns = np.arange(-N, N+1)
    E0_mat = E0(k_vals[None,  :] + Ns[:, None]*q, **kwargs)
    if not fix_k:
    # precompute all strides upfront
        possible_strides = set(calc_stride(E, adaptive_params(E, calc_L=False, **kwargs), L, **kwargs) for E in E_vals)
        E0_sliced = {s: np.ascontiguousarray(E0_mat[:, ::s]) for s in possible_strides}
        g_bufs = {s: np.empty_like(arr) for s, arr in E0_sliced.items()}
    for i, E in enumerate(tqdm(E_vals)):
        eps = adaptive_params(E, calc_L=False, **kwargs)       # Set adaptive parameters
        if not fix_k:
            stride = calc_stride(E, eps, L, **kwargs)
            E0_view = E0_sliced[stride]
            np.subtract(E, E0_view, out=g_bufs[stride])
            g_mat = g_bufs[stride]
            E0_vals = E0_view[N]
        else:
            g_mat = E - E0_mat
            E0_vals = E0_mat[N,:]
            stride = 1.
        # g_mat = E - E0_mat[:, ::stride]
        # g_mat = np.ascontiguousarray(E - E0_mat[:, ::stride])     Maybe implement this if running into memory issues in Sigma function
        Sigma = calc_Sigma_recursive(g_mat, **kwargs)
        # E0_vals = E0_mat[N, ::stride]
        G_vals = G(E, E0_vals, Sigma=Sigma, eps=eps)
        dos_vals[i] = np.sum(np.imag(G_vals)) * (-1/np.pi) * stride / L      # Calculate DoS from GF
    if save:
        print('Saving -> ' + save_filename)
        save_kwargs = {k: v for k, v in kwargs.items() if callable(v) is False}
        np.savez(save_filename, E_vals=E_vals, dos_vals=dos_vals, k_vals=k_vals, L=L, q=q, N=N, **save_kwargs)
    return dos_vals


def calc_stride(E, eps, L, v0=v_free, B=1, print_warnings=True, 
                stride_max=100, **kwargs):
    dk = 2 * np.pi / L
    Dk = eps / (v0(E, **kwargs) + 1e-8)
    stride = int(np.floor(Dk / (dk*B)))
    if stride > stride_max:
        if print_warnings:
            print(f'Warning: stride = {stride:.3g} > stride_max = {stride_max:.3g}. Setting stride = stride_max.')
        stride = stride_max
    elif stride < 1:
        if print_warnings:
            print(f'Warning: stride = {stride:.3g} < 1. Setting stride = 1.')
        stride = 1
    return stride



if __name__ == '__main__':
    # k_max = 1.1 * np.pi
    # L = 1e5
    # dk = 2*np.pi / L
    # k_vals = np.arange(-k_max, k_max, dk)
    # k_vals = np.array([0])
    # f = 'Green_Function/Data/DoS_1D_AA_S4_V10.05_V20.025_GF.npz'
    # calc_dos(E_vals, k_vals, Sigma_func=Sigma_4_AA, L=L, V1=0.05, V2=0.025, eps=1e-4, save=True, save_filename=f)
    # f = 'Green_Function/Data/1D/DoS_1D_free_GF_adaptive_updated.npz'
    # f = 'Green_Function/Data/1D/DoS_1D_S2_V0.1_GF_adaptive_updated.npz'
    # f = 'Green_Function/Data/1D/DoS_1D_S2_V0.01_GF_adaptive.npz'
    # calc_dos_adaptive(E_vals=E_vals, Sigma_func=Sigma_2, save=True, save_filename=f, A=0.001, B=10, C=0.1, n=2, 
    #                   V=0.01, k_max=1.1, L_max=2e5)
    E_vals = np.linspace(-2, 2, 100)[1:-1]
    beta = 1 / np.sqrt(2)
    a = 1
    t = 1
    V = 0.9
    f = 'Data/1D/TB/DoS_1D_TB_t1_GF_test.npz'
    calc_dos_adaptive(E_vals=E_vals, 
                      E0=E_tb, v0=v_tb, t=t, a=a, 
                      Sigma_func=None, beta=beta,
                      save=True, save_filename=f, 
                      A=0.001, B=20, k_max=1.*np.pi,
                      L_max=3e6, L_min=2e4,
                      print_warnings=True, fix_epsilon=True, eps_fix=1e-3
                      )