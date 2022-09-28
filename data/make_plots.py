"""Figures for the simulation suite paper."""

import os.path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from lyaemu.likelihood import LikelihoodClass
import lyaemu.distinct_colours_py3 as dc

def make_res_convergence(convfile):
    """Make a plot showing the convergence of the flux power spectrum with resolution."""
    hh = h5py.File(convfile)
    #Low-res is current.
    flux_powers_lr = hh["flux_vectors"]["L15n192"][:]
    flux_powers_hr = hh["flux_vectors"]["L15n384"][:]
    flux_powers_vhr = hh["flux_vectors"]["L15n512"][:]
    kfkms_vhr = hh["kfkms"]["L15n512"][:]
    redshifts = hh["zout"][:]
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
        ax.semilogx(kfkms_vhr[ii], flux_powers_lr[ii]/flux_powers_vhr[ii], label="%.2g kpc/h" % (15000./192.), color="blue", ls="-")
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


def make_box_convergence(convfile):
    """Make a plot showing the convergence of the flux power spectrum with box size."""
    hh = h5py.File(convfile)
    #Low-res is current.
    flux_powers_lr = hh["flux_vectors"]["L15n192"][:]
    kfkms_lr = hh["kfkms"]["L15n192"][:]
    flux_powers_hr = hh["flux_vectors"]["L15n384"][:]
    kfkms_hr = hh["kfkms"]["L15n384"][:]
    flux_powers_vhr = hh["flux_vectors"]["L15n512"][:]
    kfkms_vhr = hh["kfkms"]["L15n512"][:]
    redshifts = hh["zout"][:]
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
        ax.semilogx(kfkms_vhr[ii], flux_powers_lr[ii]/flux_powers_vhr[ii], label="%.2g kpc/h" % (15000./192.), color="blue", ls="-")
        ax.semilogx(kfkms_vhr[ii], flux_powers_hr[ii]/flux_powers_vhr[ii], label="%.2g kpc/h" % (15000./384.), color="grey", ls="--")
        ax.text(kfkms_vhr[ii][0]*1.3, 1.02, "z="+str(zz))
        ax.set_ylim(1.0, 1.05)
        if (index-1) % 3 > 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticks([1.0, 1.02, 1.04], [str(1.0), str(1.02), str(1.04)])
        if index == 1:
            ax.legend()
        axes.append(ax)
        index += 1
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

def save_fig(name, zz, plotdir):
    """Format and save a figure"""
    plt.xlim(1e-3,2e-2)
    plt.ylim(bottom=0.9, top=1.1)
    plt.xlabel(r"$k_F$")
    plt.ylabel(r"$\Delta P_F(k)$ ($z = %.1f$)" % zz)
    plt.legend(loc="lower left", ncol=2,fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir,"single_param_%.2g_%s.pdf" % (zz, name[0])))
    plt.clf()

def single_parameter_plot(zz=2.2, plotdir='../figures'):
    """Plot change in each parameter of an emulator from direct simulations."""
    emulatordir = os.path.join(os.path.dirname(__file__), "emu_full_extend")
    like = LikelihoodClass(basedir=emulatordir, data_corr=False, tau_thresh=1e6)
    plimits = like.param_limits
    means = np.mean(plimits, axis=1)
    okf, defaultfv, _ = like.get_predicted(means)
    pnames = like.get_pnames()
    zind = np.argmin(np.abs(like.zout - zz))
    okf = okf[zind]
    defaultfv = defaultfv[zind]
    assert len(pnames) == np.size(means)
    dist_col = dc.get_distinct(12)
    for (i, name) in enumerate(pnames):
        upper = np.array(means)
        upper[i] = plimits[i,1]
        okf2, upperfv, _ = like.get_predicted(upper)
        assert np.all(np.abs(okf / okf2[zind] -1) < 1e-3)
        lower = np.array(means)
        lower[i] = plimits[i,0]
        okf2, lowerfv, _ = like.get_predicted(lower)
        assert np.all(np.abs(okf / okf2[zind] -1) < 1e-3)
        plt.semilogx(okf, upperfv[zind]/defaultfv, label=r"$%s=%.2g$" % (name[1], upper[i]), color=dist_col[2*i % 12])
        plt.semilogx(okf, lowerfv[zind]/defaultfv, label=r"$%s=%.2g$" % (name[1], lower[i]), ls="--", color=dist_col[(2*i+1) %12])
        if i in (1, 3, 6):
            save_fig(name, zz, plotdir)
    save_fig(pnames[-1], zz, plotdir)

if __name__ == "__main__":
    make_temperature_variation("emulator_meanT.hdf5-40")
    make_res_convergence("fluxpower_converge.hdf5")
