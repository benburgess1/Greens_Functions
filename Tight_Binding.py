import numpy as np
from GF_1D import E_tb

def calc_dos_1D(E_vals, t=1, a=1, L=1, 
                save=True, save_filename='Data.npz', 
                eps=1e-6, **kwargs):
    dos_vals =  L / (np.pi * a * np.sqrt((2*t)**2 - E_vals**2 + eps))
    if save:
        np.savez(save_filename, E_vals=E_vals, dos_vals=dos_vals, L=L, t=t, a=a, **kwargs)
    else:
        return dos_vals
    

if __name__ == '__main__':
    f = 'Data/1D/TB/DoS_1D_t1_theory.npz'
    # k_vals = np.linspace(0, np.pi, 100)
    E_vals = np.linspace(-2,2,100)[1:-1]
    calc_dos_1D(E_vals=E_vals, t=1, a=1, L=1, save_filename=f)