import numpy as np
from GF_1D import E_tb

def calc_dos_1D(k_vals, t=1, a=1, L=1, 
                save=True, save_filename='Data.npz', 
                eps=1e-6, **kwargs):
    E_vals = E_tb(k_vals, t=t, a=a, **kwargs)
    dos_vals = 1/(np.abs(2*t*a*np.sin(k_vals*a)) + eps) * L / np.pi
    if save:
        np.savez(save_filename, E_vals=E_vals, dos_vals=dos_vals, k_vals=k_vals, L=L, **kwargs)
    else:
        return dos_vals
    

if __name__ == '__main__':
    f = 'Green_Function/Data/1D/TB/DoS_1D_t1_theory.npz'
    k_vals = np.linspace(0, np.pi, 100)
    calc_dos_1D(k_vals=k_vals, t=1, a=1, L=1, save_filename=f)