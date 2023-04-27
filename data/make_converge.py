"""Create flux power spectra and save them to a file"""
import os
import h5py
import scipy.interpolate
import numpy as np
from fake_spectra import spectra
from fake_spectra import abstractsnapshot as absn
from lyaemu import mean_flux

def rebin_power_to_kms(kfkms, kfmpc, flux_powers, zbins, omega_m, omega_l = None):
    """Rebins a power spectrum to constant km/s bins.
    Bins larger than the box are discarded. The return type is thus a list,
    with each redshift bin having potentially different lengths."""
    if omega_l is None:
        omega_l = 1 - omega_m
    nz = np.size(zbins)
    nk = np.size(kfmpc)
    assert np.size(flux_powers) == nz * nk
    velfac = lambda zz: 1./(1+zz) * 100.0* np.sqrt(omega_m * (1 + zz)**3 + omega_l)
    flux_rebinned = np.zeros(nz*np.size(kfkms))
    rebinned=[scipy.interpolate.interpolate.interp1d(kfmpc,flux_powers[ii*nk:(ii+1)*nk]) for ii in range(nz)]
    okmsbins = [kfkms[np.where(kfkms > np.min(kfmpc)/velfac(zz))] for zz in zbins]
    flux_rebinned = [rebinned[ii](okmsbins[ii]*velfac(zz)) for ii, zz in enumerate(zbins)]
    return okmsbins, flux_rebinned

class FluxPower():
    """Class stores the flux power spectrum."""
    def __init__(self, maxk):
        self.spectrae = []
        self.snaps = []
        self.maxk = maxk
        self.kf = None

    def add_snapshot(self,snapshot, spec):
        """Add a power spectrum to the list."""
        self.snaps.append(snapshot)
        self.spectrae.append(spec)

    def len(self):
        """Get the number of snapshots in the list"""
        return len(self.spectrae)

    def get_power(self, kf, mean_fluxes):
        """Generate a flux power spectrum rebinned to be like the flux power from BOSS.
        This can be used as an artificial data vector."""
        mf = None
        flux_arr = np.empty(shape=(self.len(),np.size(kf)))
        for (i,ss) in enumerate(self.spectrae):
            if mean_fluxes is not None:
                mf = mean_fluxes[i]
            kf_sim, flux_power_sim = ss.get_flux_power_1D("H",1,1215, mean_flux_desired=mf)
            #Rebin flux power to have desired k bins
            rebinned=scipy.interpolate.interpolate.interp1d(kf_sim,flux_power_sim)
            ii = np.where(kf > kf_sim[0])
            ff = flux_power_sim[0]*np.ones_like(kf)
            ff[ii] = rebinned(kf[ii])
            flux_arr[i] = ff
        flux_arr = np.ravel(flux_arr)
        assert np.shape(flux_arr) == (self.len()*np.size(kf),)
        self.drop_table()
        return flux_arr

    def get_power_native_binning(self, mean_fluxes):
        """ Generate the flux power, with known optical depth, from a list of snapshots.
            maxk should be in comoving Mpc/h.
            kf is stored in comoving Mpc/h units.
            The P_F returned is in km/s units.
        """
        mf = None
        flux_arr = np.array([])
        for (i,ss) in enumerate(self.spectrae):
            if mean_fluxes is not None:
                mf = mean_fluxes[i]
            kf_sim, flux_power_sim = ss.get_flux_power_1D("H",1,1215, mean_flux_desired=mf)
            #Store k_F in comoving Mpc/h units, so that it is independent of redshift.
            vscale = ss.velfac * 3.085678e24/ss.units.UnitLength_in_cm
            kf_sim *= vscale
            ii = np.where(kf_sim <= self.maxk)
            flux_arr = np.append(flux_arr,flux_power_sim[ii])
            if self.kf is None:
                self.kf = kf_sim[ii]
            else:
                assert np.all(np.abs(kf_sim[ii]/self.kf - 1) < 1e-5)
        flux_arr = np.ravel(flux_arr)
        assert np.shape(flux_arr) == (self.len()*np.size(self.kf),)
        self.drop_table()
        return flux_arr

    def get_kf_kms(self, kf=None):
        """Get a vector of kf in km/s units for all redshifts."""
        if kf is None:
            kf = self.kf
        kfkms = np.array([ kf / (ss.velfac * 3.085678e24/ss.units.UnitLength_in_cm) for ss in self.spectrae])
        return kfkms

    def get_zout(self):
        """Get output redshifts"""
        return np.array([ss.red for ss in self.spectrae])

    def drop_table(self):
        """Reset the H1 tau array in all spectra, so it needs to be loaded from disc again."""
        for ss in self.spectrae:
            ss.tau[('H',1,1215)] = np.array([0])

class MySpectra():
    """This class stores the randomly positioned sightlines once,
       so that they are the same for each emulator point.
       max_k is in comoving h/Mpc."""
    def __init__(self, numlos = 32000, max_z= 4.2, min_z=2.1, max_k = 5., savefile=None):
        self.NumLos = numlos
        #For SDSS or BOSS the spectral resolution is
        #60 km/s at 5000 A and 80 km/s at 4300 A.
        #In principle we could generate smoothed spectra
        #and then correct the window function.
        #However, this non-linear smoothing will change the mean flux
        #and hence does not commute with mean flux rescaling.
        #I have checked that for k < 0.1 the corrected version
        #is identical to the unsmoothed version (without mean flux rescaling).
        self.spec_res = 0.
        #For BOSS the pixel resolution is actually 69 km/s.
        #We use a much smaller pixel resolution so that the window functions
        #are small for mean flux rescaling, and also so that HCDs are confined.
        self.pix_res = 10.
        self.NumLos = numlos
        #Want output every 0.2 from z=max to z=2.2, matching SDSS.
        self.zout = np.arange(max_z,min_z,-0.2)
        self.max_k = max_k
        if savefile is None:
            self.savefile = "lya_forest_spectra.hdf5"
        else:
            self.savefile = savefile

    def _get_cofm(self, num, base):
        """Get an array of sightlines."""
        try:
            #Use saved sightlines if we have them.
            return (self.cofm, self.axis)
        except AttributeError:
            #Otherwise get sightlines at random positions
            #Re-seed for repeatability
            np.random.seed(23)
            box = _get_header_attr_from_snap("BoxSize", num, base)
            #All through y axis
            axis = np.ones(self.NumLos)
            cofm = box*np.random.random_sample((self.NumLos,3))
            return cofm, axis

    def _check_redshift(self, red):
        """Check the redshift of a snapshot set is what we want."""
        if np.min(np.abs(red - self.zout)) > 0.01:
            return 0
        return 1

    def _get_spectra_snap(self, snap, base):
        """Get a snapshot with generated HI spectra"""
        #If savefile exists, reload. Otherwise do not.
        def mkspec(snap, base, cofm, axis, rf):
            """Helper function"""
            return spectra.Spectra(snap, base, cofm, axis, res=self.pix_res, savefile=self.savefile,spec_res = self.spec_res, reload_file=rf,sf_neutral=False,quiet=True, load_snapshot=rf)
        #First try to get data from the savefile, and if we can't, try the snapshot.
        try:
            ss = mkspec(snap, base, None, None, rf=False)
            if not self._check_redshift(ss.red):
                return None
        except OSError as io:
            return None
        #Check we have the same spectra
        try:
            assert np.all(ss.cofm == self.cofm)
        except AttributeError:
            #If this is the first load, we just want to use the snapshot values.
            (self.cofm, self.axis) = (ss.cofm, ss.axis)
        return ss

    def get_snapshot_list(self, base, snappref="SPECTRA_"):
        """Get the flux power spectrum in the format used by McDonald 2004
        for a snapshot set."""
        #print('Looking for spectra in', base)
        base = os.path.expanduser(base)
        powerspectra = FluxPower(maxk=self.max_k)
        for snap in range(30):
            snapdir = os.path.join(base,snappref+str(snap).rjust(3,'0'))
            #We ran out of snapshots
            if not os.path.exists(snapdir):
                snapdir = os.path.join(base,"PART_"+str(snap).rjust(3,'0'))
                if not os.path.exists(snapdir):
                    snapdir = os.path.join(base, "snap_"+str(snap).rjust(3,'0'))
                    if not os.path.exists(snapdir):
                        continue
            print(snapdir)
            #We have all we need
            if powerspectra.len() == np.size(self.zout):
                break
            try:
                ss = self._get_spectra_snap(snap, base)
#                 print('Found spectra in', ss)
                if ss is not None:
                    powerspectra.add_snapshot(snap,ss)
            except IOError:
                print("Didn't find any spectra because of IOError")
                raise
        #Make sure we have enough outputs
        if powerspectra.len() != np.size(self.zout):
            raise ValueError("Found only",powerspectra.len(),"of",np.size(self.zout),"from snaps:",powerspectra.snaps)
        return powerspectra

def _get_header_attr_from_snap(attr, num, base):
    """Get a header attribute from a snapshot, if it exists."""
    f = absn.AbstractSnapshotFactory(num, base)
    value = f.get_header_attr(attr)
    del f
    return value

def converge(big, small, max_z=4.2, min_z=2.1, mf=True, savefile=None):
    """Save different box sizes"""
    myspec_big = MySpectra(max_z=max_z, min_z=min_z, savefile=savefile)
    power_big = myspec_big.get_snapshot_list(big)
    myspec_small = MySpectra(max_z=max_z, min_z=min_z, savefile=savefile)
    power_small = myspec_small.get_snapshot_list(small)
    zout_big = power_big.get_zout()
    zout_small = power_small.get_zout()
    assert np.max(np.abs(zout_big - zout_small)) < 0.02
    if mf:
        constmf = mean_flux.ConstMeanFlux()
        mef = constmf.get_mean_flux(zout_big)
        mef = mef[0]
    else:
        mef = None
    fpk_small = power_small.get_power_native_binning(mean_fluxes=mef)
    fpk_big = power_big.get_power_native_binning(mean_fluxes=mef)
    maxk = np.min([np.max(power_small.kf), np.max(power_big.kf)])
    mink = np.max([np.min(power_small.kf), np.min(power_big.kf)])
    ind = np.where((power_small.kf < maxk)*(power_small.kf > mink))
    nks_old = np.size(power_small.kf)
    kf = power_small.kf[ind]
    nkb = np.size(power_big.kf)
    nks = np.size(kf)
    nz = np.size(zout_big)
    fpk_big_rebin = np.zeros(nks * nz)
    fpk_small_rebin = np.zeros(nks * nz)
    for ii in np.arange(nz):
        rebinned_big=scipy.interpolate.interpolate.interp1d(power_big.kf,fpk_big[ii*nkb:(ii+1)*nkb])
        fpk_big_rebin[ii*nks:(ii+1)*nks] = rebinned_big(kf)
        fpk_small_rebin[ii*nks:(ii+1)*nks] = fpk_small[ii*nks_old:(ii+1)*nks_old][ind]
    return zout_big, power_small, kf, fpk_big_rebin, fpk_small_rebin

def box_converge(big, small, mf=True, savefile="box_converge.hdf5"):
    """Convergence with box size"""
    zout_big, power_small, kf, fpk_big_rebin, fpk_small_rebin = converge(big, small, mf=mf)
    with h5py.File(savefile, 'w') as ff:
        ff["zout"] = zout_big
        ff["kfkms"] = power_small.get_kf_kms(kf)
        ff["kfmpc"] = kf
        ff.create_group("flux_powers")
        ff["flux_powers"]["L120n1024"] = fpk_big_rebin
        ff["flux_powers"]["L60n512"] = fpk_small_rebin

def res_converge(big, small, mf=True, savefile="res_converge.hdf5"):
    """Convergence with resolution"""
    zout_big, power_small, kf, fpk_big_rebin, fpk_small_rebin = converge(big, small, mf=mf, max_z=5.4, min_z=1.95, savefile="lya_forest_spectra_grid_480.hdf5")
    with h5py.File(savefile, 'w') as ff:
        ff["zout"] = zout_big
        ff["kfkms"] = power_small.get_kf_kms(kf)
        ff["kfmpc"] = kf
        ff.create_group("flux_powers")
        ff["flux_powers"]["L120n3072"] = fpk_big_rebin
        ff["flux_powers"]["L120n1536"] = fpk_small_rebin

def seed_converge(datadir, mf=True, savefile="seed_converge.hdf5"):
    """Convergence with box size"""
    orig = datadir+"emu_full/ns0.881Ap2.25e-09herei3.75heref2.75alphaq1.32hub0.708omegamh20.144hireionz6.62bhfeedback0.06/output"
    seed = datadir+"emu_full/ns0.881Ap2.25e-09herei3.75heref2.75alphaq1.32hub0.708omegamh20.144hireionz6.62bhfeedback0.06-seed/output"
    zout_big, power_small, kf, fpk_big_rebin, fpk_small_rebin = converge(orig, seed, mf=mf, savefile="lya_forest_spectra_grid_480.hdf5")
    with h5py.File(savefile, 'w') as ff:
        ff["zout"] = zout_big
        ff["kfkms"] = power_small.get_kf_kms(kf)
        ff["kfmpc"] = kf
        ff.create_group("flux_powers")
        ff["flux_powers"]["orig"] = fpk_big_rebin
        ff["flux_powers"]["seed"] = fpk_small_rebin

if __name__ == "__main__":
#     box_converge("~/data/Lya_forest/benchmark2/L120n1024converge/output","~/data/Lya_forest/benchmark2/L60n512converge/output")
#     box_converge("~/data/Lya_forest/benchmark2/L120n1024converge/output","~/data/Lya_forest/benchmark2/L60n512converge/output", mf=False, savefile="box_converge_nomf.hdf5")
    datadir= "/bigdata/birdlab/shared/Lya_emu_spectra/"
#     ressim = "ns0.859Ap1.29e-09herei3.92heref2.72alphaq1.87hub0.693omegamh20.141hireionz7.15bhfeedback0.0579/output"
#     res_converge(datadir+"emu_full_hires/"+ressim,datadir+"emu_full/"+ressim)
#     res_converge(datadir+"emu_full_hires/"+ressim,datadir+"emu_full/"+ressim, mf=False, savefile="res_converge_nomf.hdf5")
    seed_converge(datadir=datadir)
