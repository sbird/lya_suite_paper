"""Figures for the simulation suite paper."""

import os.path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from lyaemu.likelihood import LikelihoodClass
import lyaemu.distinct_colours_py3 as dc

def make_box_convergence(convfile):
    """Make a plot showing the convergence of the flux power spectrum with box size."""
    hh = h5py.File(convfile)
    #Low-res is current.
    flux_powers_lr = hh["flux_powers"]["L120n1024"][:]
    flux_powers_hr = hh["flux_powers"]["L60n512"][:]
    kfkms_vhr = hh["kfkms"][:]
    redshifts = hh["zout"][:]
    nk = np.shape(kfkms_vhr)[0]
    flux_powers_lr = flux_powers_lr.reshape((nk,-1))
    flux_powers_hr = flux_powers_hr.reshape((nk,-1))
    fig = plt.figure()
    axes = []
    index = 1
    for ii, zz in enumerate(redshifts):
        if zz > 4.4 or zz < 2.2:
            continue
        sharex=None
        sharey=None
        if index > 3:
            sharex = axes[(index-1) % 3]
        #if (index-1) % 3 > 0:
            #sharey = axes[index -1 - ((index-1) % 3)]
        ax = fig.add_subplot(4,3, index, sharex=sharex, sharey=sharey)
        if sharex is None:
            ax.set_xlim(1.5e-3, 2e-2)
        ax.semilogx(kfkms_vhr[ii], flux_powers_lr[ii]/flux_powers_hr[ii], color="blue", ls="-")
        ax.text(2.5e-3, 0.97, "z=%.2g" % zz)
        ax.set_ylim(0.95, 1.05)
        ax.grid(visible=True, axis='y')
        if (index-1) % 3 > 0:
            ax.set_yticks([0.98,1.0, 1.02]) #, [str(0.97), str(1.0), str(1.03)])
            ax.set_yticklabels([])
        else:
            ax.set_yticks([0.98,1.0, 1.02]) #, [str(0.97), str(1.0), str(1.03)])
            plt.ylabel(r"$P_F / P_F^{ref}$")
#         if index > 8:
#             ax.set_xlim(2e-3, 2e-2)
#         if index > 8:
#             ax.set_xticklabels([])
#             ax.set_xticks([2e-3,5e-3,0.01,2e-2])
        plt.xlabel("k (s/km)")
        axes.append(ax)
        index += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("../figures/box-convergence.pdf")

def make_res_convergence(convfile="fluxpower_converge.hdf5", convfile2="res_converge_nomf.hdf5"):
    """Make a plot showing the convergence of the flux power spectrum with resolution."""
    with h5py.File(convfile2) as hh:
        flux_powers_lr2 = hh["flux_powers"]["L120n1536"][:]
        flux_powers_hr2 = hh["flux_powers"]["L120n3072"][:]
        kfkms_vhr2 = hh["kfkms"][:]
        redshifts2 = hh["zout"][:]
        nk = np.shape(kfkms_vhr2)[0]
        flux_powers_lr2 = flux_powers_lr2.reshape((nk,-1))
        flux_powers_hr2 = flux_powers_hr2.reshape((nk,-1))
    with h5py.File(convfile) as hh:
        flux_powers_hr = hh["flux_vectors"]["L15n384"][:]
        flux_powers_vhr = hh["flux_vectors"]["L15n512"][:]
        kfkms_vhr = hh["kfkms"]["L15n512"][:]
        redshifts = hh["zout"][:]
    assert np.size(redshifts) == np.size(redshifts2)
    fig = plt.figure()
    axes = []
    index = 1
    for ii, zz in enumerate(redshifts):
        if zz > 4.4 or zz < 2.2:
            continue
        sharex=None
        sharey=None
        if index > 3:
            sharex = axes[(index-1) % 3]
        #if (index-1) % 3 > 0:
            #sharey = axes[index -1 - ((index-1) % 3)]
        ax = fig.add_subplot(4,3, index, sharex=sharex, sharey=sharey)
        ax.semilogx(kfkms_vhr2[ii], flux_powers_lr2[ii]/flux_powers_hr2[ii], label="%.2g kpc/h" % (120000./1536.), color="blue", ls="-")
        ax.semilogx(kfkms_vhr[ii], flux_powers_hr[ii]/flux_powers_vhr[ii], label="%.2g kpc/h" % (15000./384.), color="grey", ls="--")
        ax.text(0.023, 1.04, "z="+str(zz))
        ax.set_ylim(0.99, 1.05)
        if (index-1) % 3 > 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticks([1.0, 1.02, 1.04], [str(1.0), str(1.02), str(1.04)])
            plt.ylabel(r"$P_F / P_F^{ref}$")

        if index == 1:
            ax.legend()
        axes.append(ax)
        index += 1
        plt.xlabel("k (s/km)")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("../figures/resolution-convergence.pdf")

def make_temperature_variation(tempfile, ex=5, gkfile="Gaikwad_2020b_T0_Evolution_All_Statistics.txt"):
    """Make a plot of the possible temperature variations over time."""
    obs = np.loadtxt(gkfile)
    plt.xlim(4.5, 2)
    hh = h5py.File(tempfile)
    redshift = hh["zout"][:]
    ii = np.where(redshift <= 4.4)
    mint = np.min(hh["meanT"][:], axis=0)
    maxt = np.max(hh["meanT"][:], axis=0)
    plt.fill_between(redshift[ii], mint[ii]/1e4, maxt[ii]/1e4, color="grey", alpha=0.3)
    plt.errorbar(obs[:,0], obs[:,9]/1e4, fmt='o', xerr=0.1, yerr=obs[:,10]/1e4)
    plt.plot(redshift[ii], hh["meanT"][:][ex][ii]/1e4, color="black", ls="-")
    print(hh["params"][:][ex])
    plt.xlabel("z")
    plt.ylabel(r"$T_0$ ($10^4$ K)")
    plt.savefig("../figures/mean-temperature.pdf")

def save_fig(name, plotdir):
    """Format and save a figure"""
    plt.xlim(1e-3,2e-2)
    #plt.ylim(bottom=0.9, top=1.1)
    plt.xlabel(r"$k_F$")
    plt.ylabel(r"$\Delta P_F(k)$ ($%s$)" % name[1])
    plt.legend(loc="lower left", ncol=1,fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir,"single_param_%s.pdf" % name[0]))
    plt.clf()

def single_parameter_plot(zzs=None, plotdir='../figures'):
    """Plot change in each parameter of an emulator from direct simulations."""
    emulatordir = os.path.join(os.path.dirname(__file__), "emu_full_extend")
    like = LikelihoodClass(basedir=emulatordir, data_corr=False, tau_thresh=1e6)
    plimits = like.param_limits
    means = np.mean(plimits, axis=1)
    okf, defaultfv, _ = like.get_predicted(means)
    pnames = like.get_pnames()
    if zzs is None:
        zzs = np.array([2.2, 3.2, 4.4])
    assert len(pnames) == np.size(means)
    dist_col = dc.get_distinct(12)
    for (i, name) in enumerate(pnames):
        upper = np.array(means)
        upper[i] = plimits[i,1]
        okf2, upperfv, _ = like.get_predicted(upper)
        assert np.all(np.abs(okf[0] / okf2[0] -1) < 1e-3)
        lower = np.array(means)
        lower[i] = plimits[i,0]
        okf2, lowerfv, _ = like.get_predicted(lower)
        assert np.all(np.abs(okf[-1] / okf2[-1] -1) < 1e-3)
        for (j,zz) in enumerate(zzs):
            zind = np.argmin(np.abs(like.zout - zz))
            plt.semilogx(okf[zind], upperfv[zind]/defaultfv[zind], label=r"$%s=%.2g$, $z=%.2g$" % (name[1], upper[i], zz), color=dist_col[2*j % 12])
            plt.semilogx(okf[zind], lowerfv[zind]/defaultfv[zind], label=r"$%s=%.2g$, $z=%.2g$" % (name[1], lower[i], zz), ls="--", color=dist_col[(2*j+1) %12])
        save_fig(name, plotdir)
    return like

def three_panel():
    fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(24,6), sharey=True)
    plt.setp(ax, xticks=[5,4,3,2], xlim=[5.5,1.9], yticks=[0.8,0.9,1,1.1,1.2], ylim=[0.75,1.25])

    for i in range(3):
        ax[i].plot([5.6,1.8],[1,1],'k-')
        ax[i].plot(zbins, l15n192[:, 10**i]/l15n512[:, 10**i], 'yo-', label='Full 192/512'*(1-i), alpha=0.7)
        ax[i].plot(zbins, l15n256[:, 10**i]/l15n512[:, 10**i], 'ro-', label='Full 256/512'*(1-i), alpha=0.7)
        ax[i].plot(zbins, l15n384[:, 10**i]/l15n512[:, 10**i], 'bo-', label='Full 384/512'*(1-i), alpha=0.7)
        ax[i].legend(loc='best', fontsize=18, title=str(10**i)+r'$\times$Mean Density', title_fontsize=20)

    ax[1].set_xlabel("Redshift", fontsize=18)
    ax[0].set_ylabel(r"$T_{low}/T_{high}$", fontsize=20)
    fig.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(figbase+'comp-temps.pdf')
    plt.show()

if __name__ == "__main__":
#    make_temperature_variation("emulator_meanT.hdf5-40")
    make_res_convergence()
#    make_box_convergence("box_converge.hdf5")
  #  single_parameter_plot()
