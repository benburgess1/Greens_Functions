import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.ticker import ScalarFormatter
import AA_ED


def plot_dos(filenames, colors=None, labels=None, plot_theory=False, filename_theory='', 
             normalise_E=False, ylim=None, xlim=None,
             plot_title=True, title_params={}, plot_legend=True,
             xlab=r'$E/E_R$'):
    fig, ax = plt.subplots()
    if colors is None:
        colors = ['b' for i in range(len(filenames))]
    if labels is None:
        labels = ['Numerics' for i in range(len(filenames))]
    for i,f in enumerate(filenames):
        data = np.load(f)
        E_vals = data['E_vals']
        dos_vals = data['dos_vals']
        if normalise_E:
            if 'V' in data.files:
                V = data['V']
                if V > 1:
                    E_vals /= V
                    dos_vals *= V
        ax.plot(E_vals, dos_vals, color=colors[i], ls='-', marker=None, label=labels[i])
    ax.set_xlabel(xlab)
    ax.set_ylabel(r'$\rho$ (arb.)')
    if plot_theory:
        data_theory = np.load(filename_theory)
        E_vals = data_theory['E_vals']
        dos_vals = data_theory['dos_vals']
        ax.plot(E_vals, dos_vals, color='r', ls='', marker='.', ms=2, label='Theory')
    if plot_legend:
        ax.legend()

    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    
    formatter = ScalarFormatter(useOffset=True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)

    if plot_title:
        title_str = make_title_str(title_params, data, base_str=r'$\rho (E)$')
        ax.set_title(title_str)


    plt.show()


def make_title_str(title_params, data, base_str='', dp=5):
    for k, v in title_params.items():
        val = data[k]
        if len(base_str) > 0:
            base_str += ', '
        base_str += v + r'$=$' + f'{val:.{dp}g}'
    return base_str


def plot_dos_ED(filename, calc_new=False, plot_eigenvalues=False, xlab=r'$E$', xlim=None, ylim=None, 
                plot_title=True, title_params={}, plot_theory=False, filename_theory='', **kwargs):
    data = np.load(filename)
    fig, ax = plt.subplots()
    E_vals = data['E_vals']
    dos_vals = data['dos_vals']
    if calc_new:
        E_vals, dos_vals = AA_ED.dos_gaussian_stream(data['evals'], **kwargs)
    ax.plot(E_vals, dos_vals, color='b', ls='-', label='Numerics')
    if plot_eigenvalues:
        evals = data['evals']
        ax.plot(evals, np.zeros_like(evals), color='g', marker='x', ls='', ms=1, label='Eigenvalues')
    if plot_theory:
        data_theory = np.load(filename_theory)
        E_vals = data_theory['E_vals']
        dos_vals = data_theory['dos_vals']
        ax.plot(E_vals, dos_vals, color='r', ls=':', label='Theory')

    ax.legend()
    ax.set_xlabel(xlab)
    ax.set_ylabel(r'$\rho$ (arb.)')

    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    
    formatter = ScalarFormatter(useOffset=True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)

    if plot_title:
        title_str = make_title_str(title_params, data, base_str=r'$\rho (E)$')
        ax.set_title(title_str)
    
    plt.show()


if __name__ == '__main__':
    # f = 'Green_Function/Data/DoS_1D_S4_V0.15_GF.npz'
    # f2 = 'Green_Function/Data/DoS_1D_S2_V0.15_theory_extended.npz'
    # f = 'Green_Function/Data/DoS_1D_AA_S4_V10.05_V20.025_GF.npz'
    # f = 'Green_Function/Data/2D/DoS_2D_free_GF_adaptive.npz'
    # f = 'Green_Function/Data/2D/DoS_2D_free_GF_test.npz'
    # f2 = 'Green_Function/Data/2D/DoS_2D_free_theory.npz'
    # f = 'Green_Function/Data/1D/DoS_1D_S2_V0.1_GF_adaptive_updated.npz'
    f1 = 'Green_Function/Data/1D/DoS_1D_S2_V0.01_GF_adaptive.npz'
    f2 = 'Green_Function/Data/1D/DoS_1D_S4_V0.05_GF_adaptive.npz'
    f3 = 'Green_Function/Data/1D/DoS_1D_S6_V0.05_GF_adaptive.npz'
    f4 = 'Green_Function/Data/1D/DoS_1D_iterative_n4_V0.05.npz'
    filenames = [f1, f2, f3]
    filenames = [f3]
    filenames = [f1, f4]
    # ft = 'Green_Function/Data/1D/Dos_1D_S4_V0.05_theory.npz'
    ft = 'Green_Function/Data/1D/Dos_1D_S2_V0.01_theory_normalised.npz'
    # f2 = 'Green_Function/Data/1D/DoS_1D_S2_V0.02_theory_normalised.npz'
    # f = 'Green_Function/Data/1D/DoS_1D_free_GF_adaptive_updated.npz'
    # f2 = 'Green_Function/Data/1D/DoS_1D_free_theory_normalised.npz'
    # plot_dos(filenames=filenames, 
    #         #  colors=['b','r','cyan'], labels=['S2', 'S4', 'S6'], 
    #         colors = ['b', 'cyan'], labels=['Self-energy', 'Iterative'],
    #          ylim=(-0.5,10), xlim=(-0.1,1), plot_title=True, 
    #         #  title_params = {'a':r'$a$', 'b':r'$b$', 'kmax':r'$k_{max}$'},
    #         #  title_params={'L':r'$L$', 'eps':r'$\epsilon$'},
    #         title_params = {'V':r'$|V|$'},
    #          plot_theory=True, filename_theory=ft)
    f1 = 'Data/1D/TB/DoS_1D_TB_t1_GF.npz'
    f2 = 'Data/1D/TB/DoS_1D_AA_S2_t1_V2.0_GF.npz'
    f3 = 'Data/1D/TB/DoS_1D_AA_S4_t1_V2.0_GF.npz'
    f4 = 'Data/1D/TB/DoS_1D_AA_S6_t1_V2.0_GF.npz'
    f5 = 'Data/1D/TB/DoS_1D_AA_S8_t1_V2_GF.npz'
    # f6 = 'Data/1D/TB/DoS_1D_AA_S8_t1_V0.5_GF_precise.npz'
    # f2 = 'Data/1D/TB/DoS_1D_AA_S2_t1_V0.9_GF.npz'
    # f3 = 'Data/1D/TB/DoS_1D_AA_S4_t1_V0.9_GF.npz'
    # f4 = 'Data/1D/TB/DoS_1D_AA_S6_t1_V0.9_GF.npz'
    # f4 = 'Data/1D/TB/DoS_1D_AA_S6_t1_V2.0_GF.npz'
    # f3 = 'Data/1D/TB/DoS_1D_AA_S6_t1_V1.0_GF.npz'
    # f4 = 'Data/1D/TB/DoS_1D_AA_S6_t1_V1.5_GF.npz'
    filenames = [f1, f2, f3, f4, f5]
    # filenames = [f1, f4, f5]
    # filenames = [f1, f5, f6]
    ft = 'Data/1D/TB/DoS_1D_t1_theory.npz'
    # plot_dos(filenames=filenames, 
    #          colors = ['b', 'gold', 'limegreen', 'cyan', 'm'], labels=['TB Chain', 'S2', 'S4', 'S6', 'S8'],
    #         #  colors = ['b', 'cyan', 'm'], labels=['TB Chain', 'S6', 'S8'],
    #         #  colors = ['b', 'gold', 'cyan'], labels=['TB Chain', 'S8', 'S8 Precise'],
    #          normalise_E=False, ylim=(-0.25,8.5), xlim=None, 
    #          plot_title=True, title_params = {'t':r'$t$', 'a':r'$a$', 'V':r'$V$'},
    #          xlab=r'$E$',
    #          plot_theory=True, filename_theory=ft)

    f = 'Data/ED/Spectrum_V0.5_N10000.npz'
    # plot_dos_ED(f, calc_new=True, eta=0.0003,
    #             plot_eigenvalues=True,
    #             plot_theory=True, filename_theory=ft,
    #             xlab=r'$E$ / $t$', xlim=None, ylim=None, 
    #             plot_title=True, title_params={'V':r'$V$', 'N':r'$N$'})
    f1 = 'Data/ED/Spectrum_V0.75_N10000.npz'
    f2 = 'Data/1D/TB/DoS_1D_AA_S8_t1_V0.75_GF.npz'
    f3 = 'Data/1D/TB/DoS_1D_AA_S10_t1_V0.75_GF.npz'
    # f2 = 'Data/ED/Spectrum_V2_N10000.npz'
    # f3 = 'Data/1D/TB/DoS_1D_AA_S8_t1_V0.5_GF.npz'
    filenames = [f1, f2, f3]
    # plot_dos(filenames=filenames, 
    #         #  colors = ['c', 'm', 'b'], labels=['S6', 'S8', 'ED N=10000'],
    #          colors = ['b', 'gold', 'c'], labels=['ED N=10000', 'GF S8', 'GF S10'],
    #         #  colors = ['b', 'cyan', 'm'], labels=['TB Chain', 'S6', 'S8'],
    #         #  colors = ['b', 'gold', 'cyan'], labels=['TB Chain', 'S8', 'S8 Precise'],
    #          normalise_E=False, ylim=None, xlim=None, 
    #          plot_title=True, title_params = {'V':r'$V$'},
    #          xlab=r'$E$ / $t$',
    #          plot_theory=False)
    
    f1 = 'Data/ED/Spectrum_V0.5_N10000.npz'
    f2 = 'Data/1D/TB/DoS_1D_AA_S4_t1_V0.5_GF.npz'
    f3 = 'Data/1D/TB/DoS_1D_AA_S4_t1_V0.5_GF_eps0.01.npz'
    f4 = 'Data/1D/TB/DoS_1D_AA_S4_t1_V0.5_GF_eps0.001.npz'
    f5 = 'Data/1D/TB/DoS_1D_AA_S4_t1_V0.5_GF_eps0.0001.npz'
    filenames = [f1, f2, f3, f4, f5]
    # plot_dos(filenames=filenames, 
    #         #  colors = ['c', 'm', 'b'], labels=['S6', 'S8', 'ED N=10000'],
    #          colors = ['r', 'b', 'gold', 'limegreen', 'c'], labels=['ED N=10000', 'Adaptive', r'$\epsilon=0.01$', r'$\epsilon=0.001$', r'$\epsilon=0.0001$'],
    #         #  colors = ['b', 'cyan', 'm'], labels=['TB Chain', 'S6', 'S8'],
    #         #  colors = ['b', 'gold', 'cyan'], labels=['TB Chain', 'S8', 'S8 Precise'],
    #          normalise_E=False, ylim=None, xlim=None, 
    #          plot_title=True, title_params = {'V':r'$V$'},
    #          xlab=r'$E$ / $t$',
    #          plot_theory=False)
    # f1 = 'Data/1D/TB/DoS_1D_AA_S2_t1_V0.3_GF_test.npz'
    # f2 = 'Data/1D/TB/DoS_1D_AA_S2_t1_V0.3_GF.npz'
    # f1 = 'Data/1D/TB/DoS_1D_TB_t1_GF_test.npz'
    # f2 = 'Data/1D/TB/DoS_1D_TB_t1_GF.npz'
    f1 = 'Data/1D/TB/DoS_1D_AA_S4_t1_V0.3_GF_test3.npz'
    f2 = 'Data/1D/TB/DoS_1D_AA_S4_t1_V0.3_GF_test.npz'
    f3 = 'Data/1D/TB/DoS_1D_AA_S4_t1_V0.3_GF.npz'
    filenames = [f1,f2,f3]
    # f1 = 'Data/1D/TB/DoS_1D_AA_S6_t1_V0.3_GF_test2.npz'
    # f2 = 'Data/1D/TB/DoS_1D_AA_S6_t1_V0.3_GF_test.npz'
    # f3 = 'Data/1D/TB/DoS_1D_AA_S6_t1_V0.3_GF.npz'
    f1 = 'Data/1D/TB/DoS_1D_AA_S8_t1_V0.3_GF_test3.npz'
    # f2 = 'Data/1D/TB/DoS_1D_AA_S8_t1_V0.3_GF_test2.npz'
    f3 = 'Data/1D/TB/DoS_1D_AA_S8_t1_V0.3_GF.npz'
    filenames = [f1, f3]
    # plot_dos(filenames=filenames, 
    #         #  colors = ['c', 'm', 'b'], labels=['S6', 'S8', 'ED N=10000'],
    #         #  colors = ['c', 'r', 'b'], labels=['New updated', 'New', 'Old'],
    #          colors = ['r', 'b'], labels=['New', 'Old'],
    #         #  colors = ['b', 'cyan', 'm'], labels=['TB Chain', 'S6', 'S8'],
    #         #  colors = ['b', 'gold', 'cyan'], labels=['TB Chain', 'S8', 'S8 Precise'],
    #          normalise_E=False, ylim=None, xlim=None, 
    #          plot_title=True, title_params = {},
    #          xlab=r'$E$ / $t$',
    #          plot_theory=False)
    f1 = 'Data/ED/Spectrum_V0.5_N20000.npz'
    f2 = 'Data/1D/TB/DoS_1D_AA_N30_t1_V0.5_GF_recursive.npz'
    f3 = 'Data/1D/TB/DoS_1D_AA_N30_t1_V0.5_GF_recursive_test4.npz'
    # f2 = 'Data/1D/TB/DoS_1D_AA_N2_t1_V0.5_GF_recursive.npz'
    # f3 = 'Data/1D/TB/DoS_1D_AA_N3_t1_V0.5_GF_recursive.npz'
    # f3 = 'Data/1D/TB/DoS_1D_AA_S8_t1_V0.5_GF.npz'
    filenames = [f1, f2, f3]
    plot_dos(filenames=filenames, 
            #  colors=['b', 'r', 'c', 'gold'], labels=['Recursive N=1', 'Recursive N=2', 'Recursive N=3', 'Explicit S4'],
             colors=['b', 'r', 'c'], labels=['ED N=20000', 'Recursive N=30', 'Recursive N=30 Updated'],
             normalise_E=False, ylim=None, xlim=None, 
             plot_title=True, title_params={'V':r'$V$'},
             xlab=r'$E$ / $t$',
             plot_theory=False)
    


    
