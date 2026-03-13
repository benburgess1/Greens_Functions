import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.ticker import ScalarFormatter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import AA_ED


def plot_dos(filenames, colors=None, labels=None, plot_theory=False, filename_theory='', 
             normalise_E=False, ylim=None, xlim=None,
             plot_title=True, title_params={}, plot_legend=True,
             xlab=r'$E$ / $t$'):
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


def plot_eigenspectra(filenames, cmap="plasma", marker_size=40, figsize=None,
                      ipr_min=None, ipr_max=None, ax=None, plot_title=True, 
                      title_params={}, x_param='V1', xlab=r'$V_1$ / $t$'):
    """
    Plot energy eigenspectra with IPR-coloured markers for multiple datasets.
 
    Each dataset is plotted as a vertical column of circular markers at x = V1,
    where the colour of each marker encodes the inverse participation ratio (IPR)
    of the corresponding eigenstate.
 
    Parameters
    ----------
    filenames : list of str
        Paths to .npz files. Each file must contain:
            - 'evals'    : 1-D array of energy eigenvalues
            - 'ipr_vals' : 1-D array of IPR values (same length as evals)
            - 'V1'       : scalar giving the x-position for that dataset
    cmap : str or Colormap, optional
        Matplotlib colormap used to encode IPR. Default: 'plasma'.
    marker_size : float, optional
        Size of scatter markers. Default: 40.
    figsize : tuple of (float, float), optional
        Figure size in inches. Inferred from the number of datasets if None.
    ipr_min : float, optional
        Lower bound for the IPR colour scale. Uses global minimum if None.
    ipr_max : float, optional
        Upper bound for the IPR colour scale. Uses global maximum if None.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. A new figure/axes is created if None.
 
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """
    if not filenames:
        raise ValueError("filenames list is empty.")
 
    # ------------------------------------------------------------------
    # Load all data first so we can establish a shared colour scale
    # ------------------------------------------------------------------
    datasets = []
    for fname in filenames:
        data = np.load(fname)
        evals = data["evals"]
        ipr   = data["ipr_vals"]
        v1    = float(data[x_param])
 
        if evals.shape != ipr.shape:
            raise ValueError(
                f"'evals' and 'ipr_vals' have different lengths in {fname}."
            )
        datasets.append({"evals": evals, "ipr": ipr, x_param: v1, "fname": fname})
 
    # Global IPR colour scale
    all_ipr = np.concatenate([d["ipr"] for d in datasets])
    vmin = ipr_min if ipr_min is not None else all_ipr.min()
    vmax = ipr_max if ipr_max is not None else all_ipr.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
 
    # ------------------------------------------------------------------
    # Set up figure
    # ------------------------------------------------------------------
    if ax is None:
        if figsize is None:
            width  = max(6, len(filenames) * 1.5)
            height = 7
            figsize = (width, height)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
 
    colormap = plt.colormaps[cmap] if isinstance(cmap, str) else cmap
 
    # ------------------------------------------------------------------
    # Plot each dataset
    # ------------------------------------------------------------------
    sc = None
    for d in datasets:
        x_positions = np.full_like(d["evals"], d[x_param])
        sc = ax.scatter(
            x_positions,
            d["evals"],
            c=d["ipr"],
            cmap=colormap,
            norm=norm,
            s=marker_size,
            marker="o",
            linewidths=0,
            zorder=3,
        )
 
    # ------------------------------------------------------------------
    # Colourbar
    # ------------------------------------------------------------------
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Inverse Participation Ratio (IPR)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
 
    # ------------------------------------------------------------------
    # Axis labels and formatting
    # ------------------------------------------------------------------
    ax.set_xlabel(xlab, fontsize=13)
    ax.set_ylabel(r"$E$ / $t$", fontsize=13)
    # ax.set_title("Eigenspectra", fontsize=14)
    if plot_title:
        title_str = make_title_str(title_params, data, base_str='Eigenspectra', dp=4)
        ax.set_title(title_str)
    ax.tick_params(labelsize=10)
 
    # Set x-ticks at each V1 value
    v1_values = sorted({d[x_param] for d in datasets})
    ax.set_xticks(v1_values)
    ax.set_xticklabels([f"{v:.3g}" for v in v1_values], rotation=45, ha="right")
 
    # Light horizontal gridlines for readability
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
 
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_ipr_vs_energy(filename, log_ipr=True, marker_size=20,
                       figsize=(7, 5), ax=None, plot_title=True, title_params={}):
    """
    Plot IPR versus energy eigenvalue for a single dataset.
 
    Parameters
    ----------
    filename : str
        Path to a .npz file containing:
            - 'evals'    : 1-D array of energy eigenvalues
            - 'ipr_vals' : 1-D array of IPR values (same length as evals)
            - 'V1'       : scalar used in the plot title
    log_ipr : bool, optional
        If True, use a logarithmic y-axis for the IPR. Default: True.
    cmap : str or Colormap, optional
        Matplotlib colormap used to colour points by energy. Default: 'plasma'.
    marker_size : float, optional
        Size of scatter markers. Default: 20.
    figsize : tuple of (float, float), optional
        Figure size in inches. Default: (7, 5).
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. A new figure/axes is created if None.
 
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    data    = np.load(filename)
    evals   = data["evals"]
    ipr     = data["ipr_vals"]
    # v1      = float(data["V1"])
 
    if evals.shape != ipr.shape:
        raise ValueError("'evals' and 'ipr_vals' have different lengths.")
 
    # ------------------------------------------------------------------
    # Set up figure
    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
 
    # ------------------------------------------------------------------
    # Scatter: x = energy, y = IPR, colour = energy
    # ------------------------------------------------------------------
    sc = ax.scatter(
        evals,
        ipr,
        c='b',
        # cmap=cmap,
        s=marker_size,
        marker="o",
        linewidths=0,
        zorder=3,
    )
 
    # ------------------------------------------------------------------
    # Optional log scale
    # ------------------------------------------------------------------
    if log_ipr:
        ax.set_yscale("log")
 
    # ------------------------------------------------------------------
    # Axes labels, title, and grid
    # ------------------------------------------------------------------
    ax.set_xlabel(r'$E$ / $t$', fontsize=13)
    ipr_label = "IPR" + (" (log scale)" if log_ipr else "")
    ax.set_ylabel(ipr_label, fontsize=13)
    # ax.set_title(rf"IPR vs Energy  |  $V_1 = {v1:.3g}$", fontsize=14)
    if plot_title:
        title_str = make_title_str(title_params, data, base_str='IPR across spectrum', dp=4)
        ax.set_title(title_str)
    ax.tick_params(labelsize=10)
 
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
 
    fig.tight_layout()
    plt.show()
    return fig, ax


if __name__ == '__main__':
    # V_vals = np.arange(0., 2.1, 0.2)
    # filenames = [f'Data/ED/Spectrum_AA_V{V:.3g}_N1e+04.npz' for V in V_vals]
    # plot_eigenspectra(filenames, x_param='V', xlab=r'$V$ / $t$', title_params={'N':r'$N$'}, ipr_min=0, ipr_max=1)
    # V1_vals = np.array([0., 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1., 1.2, 1.4, 1.6, 1.8, 2.])
    V1_vals = np.arange(0., 2.1, 0.2)
    filenames = [f'Data/ED/Spectrum_DAA_V1{V1:.3g}_V20.5_N1e+04.npz' for V1 in V1_vals]
    # plot_eigenspectra(filenames, title_params={'V2':r'$V_2$', 'N':r'$N$'}, ipr_min=0, ipr_max=1)
    # f = 'Data/ED/Spectrum_DAA_V10.6_V20.5_N1e+04.npz'
    # plot_ipr_vs_energy(f, plot_title=True, title_params={'V1':r'$V_1$', 'V2':r'$V_2$', 'N':r'$N$'})
    f = 'Data/ED/Spectrum_AA_V1.2_N1e+04.npz'
    plot_ipr_vs_energy(f, plot_title=True, title_params={'V':r'$V$', 'N':r'$N$'})
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
    f1 = 'Data/ED/Spectrum_V1_N10000.npz'
    f2 = 'Data/ED/Spectrum_V1_N20000.npz'
    # f2 = 'Data/1D/TB/DoS_1D_AA_N10_t1_V0.75_GF_recursive.npz'
    # f2 = 'Data/1D/TB/DoS_1D_AA_N30_t1_V1_GF_recursive.npz'
    f3 = 'Data/1D/TB/DoS_1D_AA_N100_t1_V1_GF_recursive_test2.npz'
    # f2 = 'Data/1D/TB/DoS_1D_AA_N2_t1_V0.5_GF_recursive.npz'
    # f3 = 'Data/1D/TB/DoS_1D_AA_N3_t1_V0.5_GF_recursive.npz'
    # f3 = 'Data/1D/TB/DoS_1D_AA_S8_t1_V0.5_GF.npz'
    filenames = [f1, f2, f3]
    filenames = [f1, f3]
    f_rec = 'Data/1D/TB/DoS_1D_AA_N3000_t1_V0.95_L3e+04_GF_recursive.npz'
    f_ED = 'Data/ED/Spectrum_V0.95_N3e+05.npz'
    filenames = [f_ED, f_rec]
    # plot_dos(filenames=filenames, 
    #         #  colors=['b', 'r', 'c', 'gold'], labels=['Recursive N=1', 'Recursive N=2', 'Recursive N=3', 'Explicit S4'],
    #         #  colors=['b', 'c', 'r'], labels=['ED N=10000', 'Recursive N=30', 'Recursive N=100, fixed L'],
    #          colors=['b', 'r'], labels=['ED N=3e+05', r'Recursive N=3000'],
    #         #  colors=['b', 'c'], labels=['ED N=10000', 'Recursive N=30'],
    #          normalise_E=False, ylim=None, xlim=None, 
    #          plot_title=True, title_params={'V':r'$V$'},
    #          xlab=r'$E$ / $t$',
    #          plot_theory=False)
    N_vals = np.array([30, 100, 300, 1e3, 3e3], dtype=np.int16)
    filenames = [f'Data/1D/TB/DoS_1D_AA_N{N}_t1_V0.95_L3e+04_GF_recursive.npz' for N in N_vals]
    # N_vals = np.array([1e3, 3e3, 1e4, 3e4, 1e5], dtype=np.int32)
    # filenames = [f'Data/ED/Spectrum_V0.95_N{N:.3g}.npz' for N in N_vals]
    labels = [f'N = {N:.3g}' for N in N_vals]
    colors = ['b', 'r', 'c', 'gold', 'limegreen']
    # plot_dos(filenames=filenames, colors=colors, labels=labels,
    #          plot_title=True, title_params={'V':r'$V$'})


    


    
