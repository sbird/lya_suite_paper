"""Figures for the simulation suite paper."""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from lyaemu.coarse_grid import Emulator

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

def get_predicted(params, gpemu):
    """Helper function to get the predicted flux power spectrum and error, rebinned to match the desired kbins."""
    nparams = params
    # tau_0_i[z] @dtau_0 / tau_0_i[z] @[dtau_0 = 0]
    # Divided by lowest redshift case
    tau0_fac = mflux.mean_flux_slope_to_factor(self.zout, params[0])
    nparams = params[1:] #Keep only t0 sampling parameter (of mean flux parameters)
    # .predict should take [{list of parameters: t0; cosmo.; thermal},]
    # Here: emulating @ cosmo.; thermal; sampled t0 * [tau0_fac from above]
    predicted_nat, std_nat = gpemu.predict(np.array(nparams).reshape(1, -1), tau0_factors=tau0_fac)
    return gpemu.kf, predicted_nat, std_nat

def single_parameter_plot(plotdir='plots'):
    """Plot change in each parameter of an emulator from direct simulations."""
    emulatordir = path.join(path.dirname(__file__), "emu_full_extend")
    mf = MeanFluxFactor()
    dist_col = dc.get_distinct(12)
    emu = Emulator(emulatordir, mf=mf)
    emu.load()
    gpemu = emu.get_emulator(max_z=3, min_z=2.2)
    plimits = emu.get_param_limits(include_dense=True)
    means = np.mean(plimits, axis=1)
    defaultfv = get_predicted(means, gpemu)
    for (name, index) in mf.dense_param_names.items():
        ind = np.where(par[:,index] != defpar[index])
        for i in np.ravel(ind):
            if i % 2 == 0 or i > 7:
                continue
            tp = par[i,index]
            fp = (flux_vectors[i]/deffv)
            plt.semilogx(kfs[i][0], fp[0:np.size(kfs[i][0])], label=r"$\tau_0=%.2g$ ($z=4.2$)" % tp, color=dist_col[i])
            plt.semilogx(kfs[i][-1], fp[np.size(kfs[i][-1]):2*np.size(kfs[i][-1])], label=r"$\tau_0=%.2g$ ($z=2.2$)" % tp, ls="--", color=dist_col[i+1])
        plt.xlim(1e-3,2e-2)
        plt.ylim(bottom=0.2, top=1.3)
        plt.xlabel(r"$k_F$")
        plt.ylabel(r"$\Delta P_F(k)$")
        plt.legend(loc="lower left", ncol=2,fontsize=10)
        plt.tight_layout()
        plt.savefig(path.join(plotdir,"single_param_"+name+".pdf"))
        plt.clf()
    pnames = [r"n_s", r"A_\mathrm{P}", r"H_S", r"H_A", r"h"]
    for (name, index) in emu.param_names.items():
        dn = len(mf.dense_param_names)
        index += dn
        ind = np.where(par[:,index] != defpar[index])
        cc = 0
        for i in np.ravel(ind):
            tp = par[i,index]
            fp = (flux_vectors[i]/deffv)
            plt.semilogx(kfs[i][0], fp[0:np.size(kfs[i][0])], label=r"$%s=%.2g$ ($z=4.2$)" % (pnames[index-dn], tp), color=dist_col[2*cc])
            plt.semilogx(kfs[i][-1], fp[np.size(kfs[i][-1]):2*np.size(kfs[i][-1])], label=r"$%s=%.2g$ ($z=2.2$)" % (pnames[index-dn], tp), ls="--", color=dist_col[2*cc+1])
            cc+=1
        plt.xlim(1e-3,2e-2)
        plt.ylim(bottom=0.8, top=1.1)
        plt.xlabel(r"$k_F$")
        plt.ylabel(r"$\Delta P_F(k)$")
        plt.legend(loc="lower left", ncol=2,fontsize=10)
        plt.tight_layout()
        plt.savefig(path.join(plotdir,"single_param_"+name+".pdf"))
        plt.clf()

if __name__ == "__main__":
    make_temperature_variation("emulator_meanT.hdf5-40")
    make_res_convergence("fluxpower_converge.hdf5")
