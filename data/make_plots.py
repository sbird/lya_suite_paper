"""Figures for the simulation suite paper."""

import os.path
import shutil
import h5py
import numpy as np
import matplotlib.pyplot as plt
from lyaemu.likelihood import LikelihoodClass
from lyaemu.meanT import t0_likelihood
import lyaemu.flux_power as fps
from lyaemu.mean_flux import ConstMeanFlux
from lyaemu.coarse_grid import Emulator
import lyaemu.lyman_data as ldd
import lyaemu.distinct_colours_py3 as dc
from fake_spectra.plot_spectra import PlottingSpectra
from dla_data import ho21_cddf

def regen_15mpc_flux_box(simdir, mf=None):
    """Regenerate the 15 Mpc box flux power spectra ratios."""
    if mf is not None:
        mf = ConstMeanFlux(1)
    minz = 2.2
    maxz = 5.0
    #Set max k to a large value
    myspec = fps.MySpectra(max_z=maxz, min_z=minz, max_k=25)
    powers = myspec.get_snapshot_list(simdir)
    mef = None
    if mf is not None:
        mef = np.exp(-1*mf.get_t0(powers.get_zout()))[0]
    flux_vectors = powers.get_power_native_binning(mean_fluxes=mef, tau_thresh=1e6)
    kfkms = powers.get_kf_kms()
    return kfkms, flux_vectors, powers.get_zout()

def save_15mpc_flux_box(convfile="fluxpower_converge_2.hdf5"):
    """Save the 15 Mpc flux vectors in the box"""
    kfkms_384,flux_vectors_384,zout = regen_15mpc_flux_box("L15n384/output")
    kfkms_512,flux_vectors_512,zout = regen_15mpc_flux_box("L15n512/output")
    kfkms_384,flux_vectors_384_mf,zout = regen_15mpc_flux_box("L15n384/output", mf=True)
    kfkms_512,flux_vectors_512_mf,zout = regen_15mpc_flux_box("L15n512/output", mf=True)
    with h5py.File(convfile, 'w') as hh:
        hh.create_group("flux_vectors")
        hh["flux_vectors"]["L15n384"] = flux_vectors_384
        hh["flux_vectors"]["L15n512"] = flux_vectors_512
        hh["flux_vectors"]["L15n384mf"] = flux_vectors_384_mf
        hh["flux_vectors"]["L15n512mf"] = flux_vectors_512_mf
        hh.create_group("kfkms")
        hh["kfkms"]["L15n512"] = kfkms_512
        hh["kfkms"]["L15n384"] = kfkms_384
        hh["zout"] = zout

def regen_single_flux_power_emu(emudir, outdir="", mf=None):
    """Make and save a file of the flux power spectra with and without optical depth rescaling"""
    if mf is not None:
        mf = ConstMeanFlux(1)
    minz = 2.1
    maxz = 5.0
    emu = Emulator(emudir, tau_thresh=1e6, mf=mf)
    emu.load()
    emu.max_z = maxz
    emu.min_z = minz
    #Set max k to a large value
    emu.kf = np.concatenate([ldd.BOSSData().get_kf(),np.logspace(np.log10(0.02), np.log10(0.1), 20)])
    emu.set_maxk()
    print(emu.kf[-10:], emu.maxk)
    emu.myspec = fps.MySpectra(max_z=emu.max_z, min_z=emu.min_z, max_k=emu.maxk)
    #Backup any existing flux powers.
    savefile = "cc_emulator_flux_vectors_tau1000000.hdf5"
    try:
        shutil.move(os.path.join(emudir, savefile), os.path.join(emudir, savefile+".backup"))
    except FileNotFoundError:
        pass
    emu.get_flux_vectors(min_z=2.2, max_z=5.0)
    #Move new flux powers where I want them
    newsavefile = savefile
    if mf is None:
        newsavefile= "nomf_emulator_flux_vectors_tau1000000.hdf5"
    save = os.path.join("fpk_highk/"+outdir, newsavefile)
    shutil.move(os.path.join(emudir, savefile), save)
    try:
        shutil.move(os.path.join(emudir, savefile+".backup"), os.path.join(emudir, savefile))
    except FileNotFoundError:
        pass

def get_flux_power_resolution(hiresdir, lowresdir):
    """Make and save a file of the flux power spectra with and without optical depth rescaling"""
    #Without mean flux rescaling
    regen_single_flux_power_emu(hiresdir, outdir="hires")
    regen_single_flux_power_emu(hiresdir, outdir="hires", mf=True)
    regen_single_flux_power_emu(lowresdir, mf=True)
    regen_single_flux_power_emu(lowresdir)

def plot_dla_cddf():
    """Plot the strong absorber column density function"""
    emudir = "emu_full_hires"
    sims = ["ns0.859Ap1.29e-09herei3.92heref2.72alphaq1.87hub0.693omegamh20.141hireionz7.15bhfeedback0.0579",
            "ns0.909Ap1.98e-09herei3.75heref3.01alphaq2.43hub0.682omegamh20.14hireionz7.6bhfeedback0.0449",
            "ns0.914Ap1.32e-09herei3.85heref2.65alphaq1.57hub0.742omegamh20.141hireionz6.88bhfeedback0.04"]
    nums = [23, 25, 22]
    for i in range(3):
        basedir = os.path.join(emudir, sims[i])
        basedir = os.path.join(basedir, "output")
        ps = PlottingSpectra(num=nums[i], base=basedir, savefile="rand_spectra_DLA.hdf5")
        ps.plot_cddf(minN=19, moment=True, dlogN=0.1, maxN=22.4)
    ho21_cddf(redshift=2.2, moment=True)
    plt.ylabel("N f(N)")
    plt.savefig("../figures/cddf_hires.pdf")

def close(x, y):
    """Decide if two fp numbers are close"""
    return np.any(np.abs(x - y) < 0.01)

def make_box_convergence(convfile, convfile2):
    """Make a plot showing the convergence of the flux power spectrum with box size."""
    with h5py.File(convfile) as hh:
        #Low-res is current.
        flux_powers_lr = hh["flux_powers"]["L120n1024"][:]
        flux_powers_hr = hh["flux_powers"]["L60n512"][:]
        kfkms_vhr = hh["kfkms"][:]
        redshifts = hh["zout"][:]
        nk = np.shape(kfkms_vhr)[0]
        flux_powers_lr = flux_powers_lr.reshape((nk,-1))
        flux_powers_hr = flux_powers_hr.reshape((nk,-1))
    with h5py.File(convfile2) as hh:
        #Low-res is current.
        flux_powers_lr2 = hh["flux_powers"]["orig"][:]
        flux_powers_hr2 = hh["flux_powers"]["seed"][:]
        kfkms_vhr2 = hh["kfkms"][:]
        redshifts2 = hh["zout"][:]
        nk = np.shape(kfkms_vhr2)[0]
        flux_powers_lr2 = flux_powers_lr2.reshape((nk,-1))
        flux_powers_hr2 = flux_powers_hr2.reshape((nk,-1))
#     assert np.all(redshifts == redshifts2)
    fig = plt.figure()
    dist_col = dc.get_distinct(2)
    axes = []
    index = 1
    for ii, zz in enumerate(redshifts):
        if not close(zz, np.array([4.0, 3.6, 3.2, 3.0, 2.6,2.2])):
            continue
        sharex=None
        sharey=None
        if index > 3:
            sharex = axes[(index-1) % 3]
        #if (index-1) % 3 > 0:
            #sharey = axes[index -1 - ((index-1) % 3)]
        ax = fig.add_subplot(4,3, index, sharex=sharex, sharey=sharey)
        if sharex is None:
            ax.set_xlim(1e-3, 2e-2)
        ax.semilogx(kfkms_vhr[ii], flux_powers_lr[ii]/flux_powers_hr[ii], color=dist_col[0], ls="-", label="120/60")
        ax.semilogx(kfkms_vhr2[ii], flux_powers_lr2[ii]/flux_powers_hr2[ii], color=dist_col[1], ls="--", label="Seed")
        ax.text(1.5e-3, 0.95, "z=%.2g" % zz)
        ax.set_ylim(0.93, 1.07)
        ax.grid(visible=True, axis='y')
        if (index-1) % 3 > 0:
            ax.set_yticks([0.95, 0.98,1.0, 1.02, 1.05]) #, [str(0.97), str(1.0), str(1.03)])
            ax.set_yticklabels([])
        else:
            ax.set_yticks([0.95, 0.98,1.0, 1.02, 1.05]) #, [str(0.97), str(1.0), str(1.03)])
            plt.ylabel(r"$P_F / P_F^{ref}$")
        if index == 3:
            ax.legend(loc="upper right", ncol=2, fontsize=7)
#         if index > 8:
#             ax.set_xlim(2e-3, 2e-2)
#         if index > 8:
#             ax.set_xticklabels([])
#             ax.set_xticks([2e-3,5e-3,0.01,2e-2])
        if index < 4:
            plt.setp(ax.get_xticklabels(), visible=False)
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

def make_res_convergence2(convfile="fluxpower_converge_2.hdf5", mf_hires="fpk_highk/hires", mf_lowres="fpk_highk"):
    """Make a plot showing the convergence of the flux power spectrum with resolution."""
    mffile = "cc_emulator_flux_vectors_tau1000000.hdf5"
    mf_flux_hires = h5py.File(os.path.join(mf_hires, mffile))
    mf_flux = h5py.File(os.path.join(mf_lowres, mffile))
    #Find spectra with tau0 = 1.
    #ii = np.where(np.abs(mf_flux_hires["params"][:][:,0] -1) < 0.02)
    params_hr = mf_flux_hires["params"][:] #[ii,1:][0]
    #ii_lr = np.where(np.abs(mf_flux["params"][:][:,0] -1) < 0.02)
    params_lr = mf_flux["params"][:] #[ii_lr,1:][0]
    flux_powers_hr2 = mf_flux_hires["flux_vectors"][:] #[ii,:][0]
    flux_powers_lr2 = mf_flux["flux_vectors"][:] #[ii_lr,:][0]
    nhires = np.shape(params_hr)[0]
    nlores = np.shape(params_lr)[0]
    paraminds = [np.argmin(np.sum((params_hr[iii,:]- params_lr)**2,axis=1)) for iii in range(nhires)]
    kfkms_vhr2 = mf_flux_hires["kfkms"][:] #[ii,:][0]
    redshifts2 = mf_flux_hires["zout"][:]
    nz = np.shape(redshifts2)[0]
    redshifts_lr = mf_flux["zout"][:]
    nzlr = np.shape(mf_flux["zout"][:])[0]
    flux_powers_lr2 = flux_powers_lr2.reshape((nlores, nzlr,-1))
    flux_powers_hr2 = flux_powers_hr2.reshape((nhires, nz,-1))
    assert np.shape(kfkms_vhr2)[-1] == np.shape(flux_powers_hr2)[-1]
    assert np.shape(kfkms_vhr2)[-1] == np.shape(flux_powers_lr2)[-1]
    with h5py.File(convfile) as hh:
        kfkms_vhr = hh["kfkms"]["L15n512"][:]
        redshifts = hh["zout"][:]
        nzvhr = np.size(redshifts)
        flux_powers_hr = hh["flux_vectors"]["L15n384mf"][:]
        flux_powers_vhr = hh["flux_vectors"]["L15n512mf"][:]
        flux_powers_hr = flux_powers_hr.reshape((nzvhr, -1))
        flux_powers_vhr = flux_powers_vhr.reshape((nzvhr, -1))
    fig = plt.figure()
    axes = []
    index = 1
    dist_col = dc.get_distinct(2)
    for ii, zz in enumerate(redshifts2):
        if zz > 5.1 or zz < 2.1:
            continue
        if close(zz, [4.8, 4.4,4.0]):
            continue
        sharex=None
        sharey=None
        if index > 3:
            sharex = axes[(index-1) % 3]
        #if (index-1) % 3 > 0:
            #sharey = axes[index -1 - ((index-1) % 3)]
        ax = fig.add_subplot(4,3, index, sharex=sharex, sharey=sharey)
        ii_lr = np.where(np.abs(redshifts_lr - zz) < 0.01)[0][0]
        for jj in range(nhires):
            label = "%.2g kpc/h" % (120000./1536.)
            if jj > 0:
                label = ""
            ax.semilogx(kfkms_vhr2[jj, ii], flux_powers_lr2[paraminds[jj], ii_lr]/flux_powers_hr2[jj, ii], label=label, color=dist_col[0], ls="-")
        zz2 = np.where(np.abs(redshifts - zz) < 0.01)
        ax.semilogx(kfkms_vhr[zz2][0], flux_powers_hr[zz2][0]/flux_powers_vhr[zz2][0], label="%.2g kpc/h" % (15000./384.), ls="--", color=dist_col[1])
        ax.text(0.025, 1.06, "z=%.1f" % zz)
        ax.grid(visible=True, axis='y')
        ax.set_ylim(0.95, 1.10)
        ax.set_yticks([0.95, 0.98, 1.0, 1.02, 1.06]) #, [str(1.0), str(1.02), str(1.04)])
        if (index-1) % 3 > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            plt.ylabel(r"$P_F / P_F^{ref}$")
        if index == 1:
            ax.legend(loc="upper left", frameon=False, fontsize='small')
#         print("z=%g, index %d\n" % (zz, index))
        if index < 10:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlim(1.5e-3, 0.1)
            ax.set_xlabel("k (s/km)")
        axes.append(ax)
        index += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("../figures/resolution-convergence.pdf")
    plt.clf()

import scipy.interpolate

def rebin_power_to_kms(kfkms, kfmpc, flux_powers, zbins, omega_m, omega_l=None):
    """Rebins a power spectrum to constant km/s bins.
    Bins larger than the box are discarded. The return type is thus a list,
    with each redshift bin having potentially different lengths."""
    if omega_l is None:
        omega_l = 1 - omega_m
    nz = np.size(zbins)
    assert np.size(flux_powers) == nz * np.size(kfmpc)
    # conversion factor from mpc to kms units
    velfac = 1./(1+zbins) * 100.0*np.sqrt(omega_m*(1 + zbins)**3 + omega_l)
    # original simulation output
    okmsbins = np.outer(kfmpc, 1./velfac).T
    flux_powers = flux_powers.reshape(okmsbins.shape)
    # interpolate simulation output for averaging
    rebinned = [scipy.interpolate.UnivariateSpline(okmsbins[i], flux_powers[i], s=0, k=3) for i in range(nz)]
    new_flux_powers = np.array([rebinned[i](kfkms) for i in range(nz)])
    assert np.all(new_flux_powers > 0)
    return np.repeat([kfkms], nz, axis=0), new_flux_powers

def get_flux(mf_dir, rebinned = True):
    """Rebin the MF flux power file into the BOSS bins"""
    mffile = "cc_emulator_flux_vectors_tau1000000.hdf5"
    mf_flux = h5py.File(os.path.join(mf_dir, mffile))
    params = mf_flux["params"][:]
    #Omegam h2 is param 6, h is param 5
    omega_m = mf_flux["params"][:][:,6]/mf_flux["params"][:][:,5]**2
    flux_powers = mf_flux["flux_vectors"][:]
    nlores = np.shape(params)[0]
    kfkms = mf_flux["kfkms"][:] #[ii,:][0]
    kfmpc = mf_flux["kfmpc"][:] #[ii,:][0]
    redshifts = mf_flux["zout"][:]
    mf_flux.close()
    nz = np.shape(redshifts)[0]
    flux_powers = flux_powers.reshape((nlores, nz,-1))
    assert np.shape(kfkms)[-1] == np.shape(flux_powers)[-1]
    if rebinned:
        npar = np.size(params[:,0])
        boss = ldd.BOSSData()
        kfkms = boss.kf[:35]
        flux_powers = np.array([rebin_power_to_kms(kfkms, kfmpc, flux_powers[i], redshifts, omega_m = omega_m[i])[1] for i in range(npar)])
    return kfkms, flux_powers, params, redshifts

def make_res_convergence_3(mf_hires="fpk_highk/hires", mf_lowres="fpk_highk"):
    """Make a plot showing the convergence of the flux power spectrum with resolution, rebinned as the emulator sees it."""
    kfkms_vhr2, flux_powers_lr2, params_lr, redshifts_lr = get_flux(mf_lowres)
    kfkms_vhr2, flux_powers_hr2, params_hr, redshifts2 = get_flux(mf_hires)
    nhires = np.shape(params_hr)[0]
    paraminds = [np.argmin(np.sum((params_hr[iii,:]- params_lr)**2,axis=1)) for iii in range(nhires)]
    fig = plt.figure()
    axes = []
    index = 1
    dist_col = dc.get_distinct(2)
    for ii, zz in enumerate(redshifts2):
        if zz > 5.1 or zz < 2.1:
            continue
        if close(zz, [4.8, 4.4,4.0]):
            continue
        sharex=None
        sharey=None
        if index > 3:
            sharex = axes[(index-1) % 3]
        #if (index-1) % 3 > 0:
            #sharey = axes[index -1 - ((index-1) % 3)]
        ax = fig.add_subplot(4,3, index, sharex=sharex, sharey=sharey)
        ii_lr = np.where(np.abs(redshifts_lr - zz) < 0.01)[0][0]
        for jj in range(nhires):
            label = "%.2g kpc/h" % (120000./1536.)
            if jj > 0:
                label = ""
            ax.semilogx(kfkms_vhr2, flux_powers_lr2[paraminds[jj], ii_lr]/flux_powers_hr2[jj, ii], label=label, color=dist_col[0], ls="-")
        ax.text(0.025, 1.06, "z=%.1f" % zz)
        ax.grid(visible=True, axis='y')
        ax.set_ylim(1., 1.06)
        ax.set_yticks([1.0, 1.02, 1.06]) #, [str(1.0), str(1.02), str(1.04)])
        if (index-1) % 3 > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            plt.ylabel(r"$P_F / P_F^{ref}$")
        if index == 1:
            ax.legend(loc="upper left", frameon=False, fontsize='small')
#         print("z=%g, index %d\n" % (zz, index))
        if index < 10:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlim(1.5e-3, 0.02)
            ax.set_xlabel("k (s/km)")
        axes.append(ax)
        index += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("../figures/resolution-convergence_test.pdf")
    plt.clf()

def make_temperature_variation(tempfile, ex=5, gkfile="Gaikwad_2020b_T0_Evolution_All_Statistics.txt"):
    """Make a plot of the possible temperature variations over time."""
    obs = np.loadtxt(gkfile)
    plt.xlim(5.4, 2.2)
    hh = h5py.File(tempfile)
    redshift = hh["zout"][:]
    mint = np.min(hh["meanT"][:], axis=0)
    maxt = np.max(hh["meanT"][:], axis=0)
    plt.fill_between(redshift, mint/1e4, maxt/1e4, color="grey", alpha=0.3)
    plt.errorbar(obs[:,0], obs[:,9]/1e4, fmt='o', xerr=0.1, yerr=obs[:,10]/1e4)
    plt.plot(redshift, hh["meanT"][:][ex]/1e4, color="black", ls="-")
    print(hh["params"][:][ex])
    plt.xlabel("z")
    plt.ylabel(r"$T_0$ ($10^4$ K)")
    plt.savefig("../figures/mean-temperature.pdf")
    plt.clf()

def make_res_convergence_t0(tempfile, hirestempfile):
    """Make a plot of the change in the mean temperature with resolution."""
    plt.xlim(5.4, 2.2)
    meanThires = h5py.File(hirestempfile)
    meanT = h5py.File(tempfile)
    nhires = np.size(meanThires["params"][:][:,0])
    paraminds = [np.argmin(np.sum((meanThires["params"][:][ii,:]- meanT["params"][:])**2,axis=1)) for ii in range(nhires)]
    dist_col = dc.get_distinct(nhires)
    redshift = meanThires["zout"][:]
    lss = ["-", "--", ":"]
    #Plot each simulation's resolution correction.
    for i in range(nhires):
        ratio = meanT["meanT"][:][paraminds[i], :] / meanThires["meanT"][:][i,:]
        plt.plot(redshift, ratio, color=dist_col[i], label="ns=%.3g" % meanThires["params"][:][i,0], ls=lss[i])
    plt.legend()
    plt.xlabel("z")
    plt.ylabel(r"$T_0(N = 1536) / T_0 (N = 3072)$")
    plt.savefig("../figures/mean-temperature-resolution.pdf")
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

def make_loo_values():
    """Generate the LOO errors for the default saved flux power spectra."""
    emulatordir = os.path.join(os.path.dirname(__file__), "dtau-48-48")
    hremudir = os.path.join(os.path.dirname(__file__), "dtau-48-48/hires")
    like = LikelihoodClass(basedir=emulatordir, HRbasedir=hremudir, data_corr=False, tau_thresh=1e6, loo_errors=False, traindir=None, use_meant=False)
    like.calculate_loo_errors(savefile="loo_fps_3.hdf5")
#    likesf = LikelihoodClass(basedir=emulatordir, data_corr=False, tau_thresh=1e6, loo_errors=True, traindir=None, use_meant=False)
#    likesf.calculate_loo_errors(savefile="loo_fps_2.hdf5")

def single_parameter_plot(zzs=None, plotdir='../figures'):
    """Plot change in each parameter of an emulator from direct simulations."""
    #emulatordir = os.path.join(os.path.dirname(__file__), "emu_full_extend")
    emulatordir = os.path.join(os.path.dirname(__file__), "dtau-48-48")
    hremudir = os.path.join(os.path.dirname(__file__), "dtau-48-48/hires")
    like = LikelihoodClass(basedir=emulatordir, HRbasedir=hremudir, data_corr=False, tau_thresh=1e6, loo_errors=True, traindir=emulatordir+"/trained_mf")
    plimits = like.param_limits
    means = np.mean(plimits, axis=1)
    okf, defaultfv, _ = like.get_predicted(means)
    pnames = like.get_pnames()
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
        if name[0] == 'herei':
            zzs = np.array([3.6, 3.2])
        elif name[0] == 'heref' or name[0] == 'alphaq':
            zzs = np.array([3.2, 2.6])
        elif name[0] == 'hireionz':
            zzs = np.array([3.2, 4.4])
        elif name[0] == 'bhfeedback':
            zzs = np.array([3.2, 2.2])
        else:
            zzs = np.array([2.2, 3.2, 4.4])
        print(name[0], zzs)
        for (j,zz) in enumerate(zzs):
            zind = np.argmin(np.abs(like.zout - zz))
            plt.semilogx(okf[zind], upperfv[zind]/defaultfv[zind], label= lblstr % (name[1], upper[i], zz), color=dist_col[2*j % 12])
            plt.semilogx(okf[zind], lowerfv[zind]/defaultfv[zind], label= lblstr % (name[1], lower[i], zz), ls="--", color=dist_col[(2*j+1) %12])
        save_fig(name, plotdir)
    return like

def save_fig_t0(plotdir, extra=""):
    """Format and save a figure"""
    #plt.xlim(1e-3,2e-2)
    #plt.ylim(bottom=0.9, top=1.1)
    plt.xlabel(r"$z$")
    plt.ylabel(r"$\Delta T0(z)$")
    plt.legend(ncol=1,fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir,"single_param_t0"+extra+".pdf"))
    plt.clf()

def single_parameter_t0_plot(plotdir='../figures', one=False):
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
    if one:
        pnames = [('herei', r'z_\mathrm{He i}'), ('hireionz', r'z_{Hi}')]
        pind = [2,7]
    else:
        pnames = [('heref', r'z_\mathrm{He f}'), ('alphaq', r'\alpha_q')]
        pind = [3,4]
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
    if one:
        extra = "_hi_"
    else:
        extra = "_hef_"
    save_fig_t0(plotdir, extra=extra)
    return like

def make_spectra_convergence():
    """Plot showing the convergence with number of spectra"""
    simdir = "emu_full/ns0.816Ap1.55e-09herei3.55heref3.15alphaq1.48hub0.658omegamh20.143hireionz7.38bhfeedback0.0533/output"
    #largest: grid with 720^2 spectra through each axis
    sgrid32 = spectra.Spectra(20, simdir, None, None, savefile="lya_forest_spectra_grid_720.hdf5", res=10)
    (kfgrid32, flux_pow_grid32) = ssgrid32.get_flux_power_1D(mean_flux_desired=mf, tau_thresh=1e6)
    #Desired: grid with 480^2 spectra through each axis
    ssgrid3 = spectra.Spectra(20, simdir, None, None, savefile="lya_forest_spectra_grid_480.hdf5", res=10)
    (kfgrid3, flux_pow_grid3) = ssgrid3.get_flux_power_1D(mean_flux_desired=mf, tau_thresh=1e6)
    #Low density: grid with 179^2 spectra through *one* axis only!
    ss = spectra.Spectra(20, simdir, None, None, savefile="lya_forest_spectra.hdf5", res=10)
    (kf, flux_pow) = ss.get_flux_power_1D(mean_flux_desired=mf, tau_thresh=1e6)
    gridspline = scipy.interpolate.interp1d(kfgrid32, flux_pow_grid32)
    plt.semilogx(kf, flux_pow/gridspline(kf), ls=":", label=r"$179^2$")
    plt.semilogx(kfgrid3, flux_pow_grid3/gridspline(kfgrid3), ls="-", label=r"$3\times 480^2$")
    plt.ylim(0.9, 1.1)
    plt.xlim(xmax=0.1)
    plt.legend()
    plt.savefig("../figures/gridfpk-diff.pdf")
    plt.clf()
    plt.loglog(kfgrid32, flux_pow_grid32, ls="-",  label=r"$3\times 720^2$")
    plt.loglog(kfgrid3, flux_pow_grid3, ls="--",  label=r"$3\times 480^2$")
    plt.loglog(kf, flux_pow, ls=":",  label=r"$179^2$")
    plt.ylim(10, 40)
    plt.xlim(xmax=0.02)
    plt.legend()
    plt.savefig("../figures/gridfpk.pdf")


if __name__ == "__main__":
#     get_flux_power_resolution("emu_full_hires", "emu_full_extend")
#     make_temperature_variation("dtau-48-46/emulator_meanT.hdf5")
#     make_res_convergence_t0("dtau-48-46/emulator_meanT.hdf5", "dtau-48-46/hires/emulator_meanT.hdf5")
#     make_res_convergence2()
#     make_res_convergence_3()
   make_box_convergence("box_converge.hdf5", "seed_converge.hdf5")
#     single_parameter_plot()
#     single_parameter_t0_plot(one=False)
#     single_parameter_t0_plot(one=True)
    # plot_dla_cddf()
