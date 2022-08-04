"""Figures for the simulation suite paper."""

import h5py
import numpy as np
import matplotlib.pyplot as plt

def make_res_convergence(convfile):
    """Make a plot showing the convergence of the flux power spectrum with resolution."""
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

if __name__ == "__main__":
    make_temperature_variation("emulator_meanT.hdf5-40")
    make_res_convergence("fluxpower_converge.hdf5")
