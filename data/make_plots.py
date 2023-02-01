"""Figures for the simulation suite paper."""

import os.path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from lyaemu.likelihood import LikelihoodClass
from lyaemu.meanT import t0_likelihood
import lyaemu.distinct_colours_py3 as dc
from fake_spectra.plot_spectra import PlottingSpectra
from dla_data import ho21_cddf

def plot_dla_cddf():
    """Plot the strong absorber column density function"""
    emudir = "emu_full_hires"
    sims = ["ns0.859Ap1.29e-09herei3.92heref2.72alphaq1.87hub0.693omegamh20.141hireionz7.15bhfeedback0.0579",
            "ns0.909Ap1.98e-09herei3.75heref3.01alphaq2.43hub0.682omegamh20.14hireionz7.6bhfeedback0.0449",
            "ns0.914Ap1.32e-09herei3.85heref2.65alphaq1.57hub0.742omegamh20.141hireionz6.88bhfeedback0.04"]
    nums = [23, 25, 22]
#    ns0.859Ap1.29e-09herei3.92heref2.72alphaq1.87hub0.693omegamh20.141hireionz7.15bhfeedback0.0579/output/SPECTRA_012/rand_spectra_DLA.hdf5
#    ns0.859Ap1.29e-09herei3.92heref2.72alphaq1.87hub0.693omegamh20.141hireionz7.15bhfeedback0.0579/output/SPECTRA_018/rand_spectra_DLA.hdf5
#    ns0.859Ap1.29e-09herei3.92heref2.72alphaq1.87hub0.693omegamh20.141hireionz7.15bhfeedback0.0579/output/SPECTRA_023/rand_spectra_DLA.hdf5
#    ns0.859Ap1.29e-09herei3.92heref2.72alphaq1.87hub0.693omegamh20.141hireionz7.15bhfeedback0.0579/output/SPECTRA_024/rand_spectra_DLA.hdf5
#    ns0.909Ap1.98e-09herei3.75heref3.01alphaq2.43hub0.682omegamh20.14hireionz7.6bhfeedback0.0449/output/SPECTRA_012/rand_spectra_DLA.hdf5
#    ns0.909Ap1.98e-09herei3.75heref3.01alphaq2.43hub0.682omegamh20.14hireionz7.6bhfeedback0.0449/output/SPECTRA_019/rand_spectra_DLA.hdf5
#    ns0.909Ap1.98e-09herei3.75heref3.01alphaq2.43hub0.682omegamh20.14hireionz7.6bhfeedback0.0449/output/SPECTRA_025/rand_spectra_DLA.hdf5
#    ns0.914Ap1.32e-09herei3.85heref2.65alphaq1.57hub0.742omegamh20.141hireionz6.88bhfeedback0.04/output/SPECTRA_012/rand_spectra_DLA.hdf5
#    ns0.914Ap1.32e-09herei3.85heref2.65alphaq1.57hub0.742omegamh20.141hireionz6.88bhfeedback0.04/output/SPECTRA_017/rand_spectra_DLA.hdf5
#    ns0.914Ap1.32e-09herei3.85heref2.65alphaq1.57hub0.742omegamh20.141hireionz6.88bhfeedback0.04/output/SPECTRA_022/rand_spectra_DLA.hdf5
    for i in range(3):
        basedir = os.path.join(emudir, sims[i])
        basedir = os.path.join(basedir, "output")
        ps = PlottingSpectra(num=nums[i], base=basedir, savefile="rand_spectra_DLA.hdf5")
        ps.plot_cddf(minN=19, moment=True)
    ho21_cddf(redshift=2.2, moment=True)
    plt.savefig("../figures/cddf_hires.pdf")

def close(x, y):
    """Decide if two fp numbers are close"""
    return np.any(np.abs(x - y) < 0.01)

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
        if not close(zz, np.array([4.0, 3.0, 2.2])):
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
    plt.clf()

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
        ax.text(0.015, 1.06, "z="+str(zz))
        ax.grid(visible=True, axis='y')
        ax.set_ylim(0.95, 1.10)
        ax.set_yticks([0.95, 0.98, 1.0, 1.02, 1.06]) #, [str(1.0), str(1.02), str(1.04)])
        if (index-1) % 3 > 0:
            ax.set_yticklabels([])
        else:
            plt.ylabel(r"$P_F / P_F^{ref}$")
        if index == 1:
            ax.legend(loc="upper left", frameon=False)
        axes.append(ax)
        index += 1
        plt.xlabel("k (s/km)")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("../figures/resolution-convergence.pdf")
    plt.clf()

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
    plt.clf()

def save_fig(name, plotdir):
    """Format and save a figure"""
    plt.xlim(1e-3,2e-2)
    #plt.ylim(bottom=0.9, top=1.1)
    plt.xlabel(r"$k_F$")
    plt.ylabel(r"$\Delta P_F(k)$ ($%s$)" % name[1])
    plt.legend(ncol=1,fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir,"single_param_%s.pdf" % name[0]))
    plt.clf()

def single_parameter_plot(zzs=None, plotdir='../figures'):
    """Plot change in each parameter of an emulator from direct simulations."""
    #emulatordir = os.path.join(os.path.dirname(__file__), "emu_full_extend")
    emulatordir = os.path.join(os.path.dirname(__file__), "dtau-48-46")
    hremudir = os.path.join(os.path.dirname(__file__), "dtau-48-46/hires")
    like = LikelihoodClass(basedir=emulatordir, HRbasedir=hremudir, data_corr=False, tau_thresh=1e6, loo_errors=True, traindir=emulatordir+"/trained_mf")
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
        #upper[i] = plimits[i,1]
        upper[i] = 0.8*(plimits[i,1] - means[i]) + means[i]
        okf2, upperfv, _ = like.get_predicted(upper)
        assert np.all(np.abs(okf[0] / okf2[0] -1) < 1e-3)
        lower = np.array(means)
        #lower[i] = plimits[i,0]
        lower[i] = 0.8*(plimits[i,0] - means[i]) + means[i]
        okf2, lowerfv, _ = like.get_predicted(lower)
        assert np.all(np.abs(okf[-1] / okf2[-1] -1) < 1e-3)
        lblstr = r"$%s=%.2g$, $z=%.2g$"
        if name[0] == 'omegamh2':
            lblstr = r"$%s=%.3g$, $z=%.2g$"
        for (j,zz) in enumerate(zzs):
            zind = np.argmin(np.abs(like.zout - zz))
            plt.semilogx(okf[zind], upperfv[zind]/defaultfv[zind], label= lblstr % (name[1], upper[i], zz), color=dist_col[2*j % 12])
            plt.semilogx(okf[zind], lowerfv[zind]/defaultfv[zind], label= lblstr % (name[1], lower[i], zz), ls="--", color=dist_col[(2*j+1) %12])
        save_fig(name, plotdir)
    return like

def save_fig_t0(plotdir):
    """Format and save a figure"""
    #plt.xlim(1e-3,2e-2)
    #plt.ylim(bottom=0.9, top=1.1)
    plt.xlabel(r"$z$")
    plt.ylabel(r"$\Delta T0(z)$")
    plt.legend(ncol=1,fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir,"single_param_t0.pdf"))
    plt.clf()

def single_parameter_t0_plot(plotdir='../figures'):
    """Plot change in each parameter of an emulator from direct simulations."""
    #emulatordir = os.path.join(os.path.dirname(__file__), "emu_full_extend")
    emulatordir = os.path.join(os.path.dirname(__file__), "dtau-48-46")
    hremudir = os.path.join(os.path.dirname(__file__), "dtau-48-46/hires")
    like = t0_likelihood.T0LikelihoodClass(basedir=emulatordir, HRbasedir=hremudir, loo_errors=True)
    plimits = like.param_limits
    means = np.mean(plimits, axis=1)
    defaultfv, _ = like.get_predicted(means)
    pnames = like.emulator.print_pnames()
    assert len(pnames) == np.size(means)
    print(pnames)
    pnames = [('herei', r'z_\mathrm{He i}'), ('heref', r'z_\mathrm{He f}'), ('alphaq', r'\alpha_q'), ('hireionz', r'z_{Hi}')]
    pind = [2,3,4,7]
    dist_col = dc.get_distinct(12)
    for (ii, name) in enumerate(pnames):
        upper = np.array(means)
        #upper[i] = plimits[i,1]
        i = pind[ii]
        upper[i] = 0.8*(plimits[i,1] - means[i]) + means[i]
        upperfv, _ = like.get_predicted(upper)
        lower = np.array(means)
        #lower[i] = plimits[i,0]
        lower[i] = 0.8*(plimits[i,0] - means[i]) + means[i]
        lowerfv, _ = like.get_predicted(lower)
        lblstr = r"$%s=%.2g$"
        if name[0] == 'omegamh2':
            lblstr = r"$%s=%.3g$"
        plt.plot(like.zout[::-1], (upperfv[0]-defaultfv[0])[::-1], label= lblstr % (name[1], upper[i]), color=dist_col[i])
        plt.plot(like.zout[::-1], (lowerfv[0]-defaultfv[0])[::-1], label= lblstr % (name[1], lower[i]), ls="--", color=dist_col[i])
    save_fig_t0(plotdir)
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
#    make_res_convergence()
   #make_box_convergence("box_converge.hdf5")
#     single_parameter_plot()
#     single_parameter_t0_plot()
    plot_dla_cddf()
