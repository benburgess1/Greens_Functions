import numpy as np
from scipy.linalg import eigh_tridiagonal

def build_H(N=1000, t=1, V=0.1, a=1, beta=1/np.sqrt(2), phi=0, full_H=False,
            **kwargs):
    x = np.arange(N) * a
    if full_H:
        H = np.diag(2*V*np.cos(2*np.pi*beta*x/a + phi))
        for i in range(N-1):
            H[i, i+1] = t
            H[i+1, i] = t
        return H
    else:
        d = 2 * V * np.cos(2*np.pi*beta*x/a + phi)
        e = t * np.ones(N-1)
        return d, e

def build_H_DAA(N=1000, t=1, V1=0.1, V2=0.1, a=1, beta1=1/np.sqrt(2), beta2=1/np.sqrt(3),
                phi1=0, phi2=0, full_H=False, **kwargs):
    x = np.arange(N) * a
    if full_H:
        H = np.diag(2*V1*np.cos(2*np.pi*beta1*x/a + phi1) + 2*V2*np.cos(2*np.pi*beta2*x/a + phi2))
        for i in range(N-1):
            H[i, i+1] = t
            H[i+1, i] = t
        return H
    else:
        d = 2*V1*np.cos(2*np.pi*beta1*x/a + phi1) + 2*V2*np.cos(2*np.pi*beta2*x/a + phi2)
        e = t * np.ones(N-1)
        return d, e

def calc_spectrum(calc_dos=True, calc_IPR=True, save_evects=False,
                  save=True, save_filename='Data.npz', 
                  H_func=build_H, **kwargs):
    save_dict = {k:v for k, v in kwargs.items()}
    d, e = H_func(full_H=False, **kwargs)
    print('Performing diagonalisation... ', end='', flush=True)
    if calc_IPR:
        evals, evects = eigh_tridiagonal(d, e, eigvals_only=False, lapack_driver='stemr')
        if save_evects:
            save_dict['evects'] = evects
    else:
        evals = eigh_tridiagonal(d, e, eigvals_only=True, lapack_driver='sterf')
    save_dict['evals'] = evals
    print('Done')
    if calc_dos:
        print('Calculating DoS... ', end='', flush=True)
        E_vals, dos_vals = dos_gaussian_stream(evals, **kwargs)
        save_dict['E_vals'] = E_vals
        save_dict['dos_vals'] = dos_vals
        print('Done')
    if calc_IPR:
        print('Calculating IPR... ', end='', flush=True)
        ipr_vals = calc_ipr(evects)
        save_dict['ipr_vals'] = ipr_vals
        print('Done')
    if save:
        np.savez(save_filename, **save_dict)
    return evals


def dos_gaussian_stream(evals, eta=None, n_points=1000, energy_window=None, **kwargs):
    N = evals.size

    if eta is None:
        eta = 5 * (np.max(evals) - np.min(evals)) / N

    if energy_window is None:
        Emin, Emax = evals.min(), evals.max()
        padding = 5 * eta
        Emin -= padding
        Emax += padding
    else:
        Emin, Emax = energy_window

    E_grid = np.linspace(Emin, Emax, n_points)

    diff = E_grid[:, None] - evals[None, :]
    rho = np.exp(-0.5 * (diff / eta)**2).sum(axis=1)
    rho /= (N * np.sqrt(2 * np.pi) * eta)

    return E_grid, rho

def calc_ipr(evects):
    return np.sum(np.abs(evects)**4, axis=0)

def update_dos(filename, **kwargs):
    data = np.load(filename)
    save_dict = {k:v for k, v in data.items()}
    evals = data['evals']
    E_vals, dos_vals = dos_gaussian_stream(evals, **kwargs)
    save_dict['E_vals'] = E_vals
    save_dict['dos_vals'] = dos_vals
    np.savez(filename, **save_dict)


if __name__ == '__main__':
    N_vals = np.array([1e3, 3e3, 1e4, 3e4, 1e5], dtype=np.int32)
    V = 1.05
    t = 1
    a = 1
    beta = 1/np.sqrt(2)
    # for i, N in enumerate(N_vals):
    #     print(f'Evaluating N = {N:.3g} ({i+1}/{len(N_vals)})')
    #     f = f'Data/ED/Spectrum_V{V:.3g}_N{N:.3g}.npz'
    #     calc_spectrum(calc_dos=True, calc_IPR=False, 
    #                 N=N, V=V, t=t, a=a, beta=beta, 
    #                 save=True, save_filename=f,
    #                 eta=0.0005, n_points=5000,
    #                 energy_window=(-3.2, 3.2))
    # f1 = 'Data/ED/Spectrum_V0.5_N20000.npz'
    # f1 = 'Data/ED/Spectrum_V0.95_N3e+05.npz'
    # update_dos(f1, eta=8e-5, n_points=5000, energy_window=(-2.7,2.7))
    N = int(1e4)
    t = 1
    a = 1
    beta1 = 1/np.sqrt(2)
    beta2 = 1/np.sqrt(3)
    V2 = 0.5
    V1_vals = np.array([0., 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.])
    for V1 in V1_vals:
        f = f'Data/ED/Spectrum_DAA_V1{V1:.3g}_V2{V2:.3g}_N{N:.3g}.npz'
        calc_spectrum(calc_dos=True, calc_IPR=True, 
                        N=N, t=t, a=a, 
                        H_func=build_H_DAA,
                        beta1=beta1, beta2=beta2,
                        V1=V1, V2=V2,
                        save=True, save_filename=f,
                        eta=0.001, n_points=5000)
