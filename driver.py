import sys, os
import numpy as np

from .ddbdata import DDBData
from .gkkdata import GKKData
from .enkdata import EnkData
from .utils import QgridFile, make_kkmap, find_kidx, mpi_watch, MPI, comm, is_master, simple_timer
from .constants import hartree2ev
from .cymodules.zprworker import calc_FM, calc_DWria

class ZPRDriver(QgridFile):

    @mpi_watch
    @simple_timer('IO done, total wall time = {t}\n')
    def __init__(self, path_root, *args, nbandk=None, nbandkq=None, use_gwpt=False, eqp_option=None, readwfq=False, **kwargs):
        if is_master():
            print()
            print('==============================================================================')
            print('Program ZPRPACK')
            print('Author: Xiaoxun Gong (xiaoxun.gong@berkeley.edu)')
            print('==============================================================================')
            print()
        
        if comm is not None:
            if comm.rank == 0:
                print('Using MPI')
                print(f'Number of processors: {comm.size}')
                print()
        else:
            print('Module mpi4py not found, execucting serially\n')

        super(ZPRDriver, self).__init__(path_root, *args, **kwargs)

        self.ddbdata = DDBData(path_root, *args, **kwargs)
        self.enkdata = EnkData(path_root, nband=nbandkq, eqp_option=eqp_option, readwfq=readwfq, prefix=kwargs.get('prefix', 'phonon-q'))
        self.gkkdata = GKKData(path_root, *args, use_gwpt=use_gwpt, nbandkq=nbandkq, nbandk=nbandk, **kwargs)

        gkkmat = self.gkkdata.gkk_matrix
        self.kqmap = make_kkmap(self.enkdata.kpt, self.gkkdata.kpt + self.qpt_loc) # gkkdata.kpt is the single k point
        self.kidx = find_kidx(self.enkdata.kpt, self.gkkdata.kpt); assert self.kidx >= 0
        self.gkkmode2 = np.abs(np.einsum('qmadkn,qvad->qmvkn', gkkmat, self.ddbdata.Uqvad))**2

        # get gkkmat_G for DW term
        gamma = np.array([[0., 0., 0.]])
        gamidx = find_kidx(self.qpt, gamma, allow_multiple=False); assert gamidx >= 0
        if comm is None:
            gkkmat_G = gkkmat[gamidx, :, :, :, 0, :]
        else:
            gamproc = np.searchsorted(self.qrange_all, gamidx, side='right') - 1
            if gamproc == comm.rank:
                gkkmat_G = gkkmat[gamidx-self.qrange_loc.start, :, :, :, 0, :]
            else:
                gkkmat_G = np.empty_like(gkkmat[0, :, :, :, 0, :])
            comm.Bcast([gkkmat_G, MPI.COMPLEX16], root=gamproc)
        self.gkkmat_G = gkkmat_G

        if comm is not None: comm.Barrier()

    @mpi_watch
    @simple_timer('\nJob done, total wall time = {t}\n')
    def calc_ren(self, temps, nbndkqmax=None, eta=0.01, adiabatic=False,
                 spectral=False, w_spectral=0., s_spectral=0.01):
        if is_master(): print('Calculating el-ph renormalization')

        Uqvad = self.ddbdata.Uqvad
        w_qv = self.ddbdata.w_qv
        wtq = self.wtq_loc
        e_skn = self.enkdata.enk
        f_skn = self.enkdata.onk / 2.0
        if nbndkqmax is None: nbndkqmax = self.gkkdata.nbandkq
        eta /= hartree2ev
        if not spectral: s_spectral = 0.0

        if is_master():
            print(f'\nUsing {nbndkqmax} bands at k+q in summation')
            print(f'Using imaginary smearing {eta*hartree2ev} eV')

        nB_qvT = np.zeros((self.nq_loc, 3*self.gkkdata.natom, len(temps))) # (nq_loc, nmode, ntemp)
        for iq_loc in range(self.nq_loc):
            nB_qvT[iq_loc,:,:] = self.ddbdata.ddbfile_loc[iq_loc].get_bose(temps) # get_bose automatically sets n_B=0 for omega<0

        if adiabatic:
            w_qv = np.zeros_like(w_qv)
            
        # calculate FM
        dEFM = calc_FM(self.gkkmode2, e_skn, f_skn, nB_qvT, w_qv, wtq, self.kqmap, nbndkqmax, self.kidx, eta, w_spectral, s_spectral)

        # calculate DW
        ekn = e_skn[:, self.kidx, :]
        dEDW = calc_DWria(self.gkkmat_G, Uqvad, ekn, nB_qvT, wtq, w_qv, nbndkqmax, eta, w_spectral, s_spectral)

        # average over degenerate subspace
        deltae = 1e-5 # Ha, energy differences less than this value will be treated as degenerate
        subspace_slice = np.nonzero(np.r_[1, np.diff(ekn[0, :self.gkkdata.nbandk])>deltae, 1])[0]
        for isub in range(len(subspace_slice)-1):
            this_slice = slice(subspace_slice[isub], subspace_slice[isub+1])
            dEFM[this_slice, :] = np.mean(dEFM[this_slice, :], axis=0)
            dEDW[this_slice, :] = np.mean(dEDW[this_slice, :], axis=0)
        
        # collect and sum up
        if comm is not None:
            def collect(mat):
                if is_master():
                    mat_recv = np.empty((self.gkkdata.nbandk, len(temps)), dtype='f8')
                else:
                    mat_recv = None
                comm.Reduce([mat, MPI.REAL8], [mat_recv, MPI.REAL8], op=MPI.SUM, root=0)
                return mat_recv
            dEFM = collect(dEFM)
            dEDW = collect(dEDW)
            
        if not is_master(): return

        dEnk = dEFM + dEDW
        self.dEnk = dEnk
        self.dEFM = dEFM
        self.dEDW = dEDW
        self.temps = temps
        
    @mpi_watch
    def write(self, savedir, fname=''):
        if not is_master(): return
        
        ekn = self.enkdata.enk[:, self.kidx, :]

        fout = open(f'{savedir}/zprdata{fname}.txt', 'w')
        fout.write('Electron-phonon renormalization result\n')
        fout.write('Author: Xiaoxun Gong (xiaoxun.gong@berkeley.edu)\n\n')
        fout.write(f'Energy unit: eV\n\n')
        kpt = self.gkkdata.kpt[0]
        fout.write(f'k = {kpt[0]:14.8f} {kpt[1]:14.8f} {kpt[2]:14.8f}\n\n')
        for iT in range(len(self.temps)):
            T = self.temps[iT]
            fout.write(f'T = {T:6.1f} K:\n\n')
            fout.write('    n          Enk         dEnk     dEnk(FM)     dEnk(DW)\n')
            for ibnd in range(self.gkkdata.nbandk):
                fout.write(f'{ibnd+1:5d} {ekn[0, ibnd]*hartree2ev:12.5f} ' + \
                           f'{self.dEnk[ibnd, iT]*hartree2ev:12.5f} ' + \
                           f'{self.dEFM[ibnd, iT]*hartree2ev:12.5f} ' + \
                           f'{self.dEDW[ibnd, iT]*hartree2ev:12.5f}\n')
            fout.write('\n')
        fout.close()

        # epceout = open(f'{savedir}/epce{fname}.txt', 'w')
        # for ik in range(gkk.kpt.shape[0]):
        #     kpt = gkk.kpt[ik]
        #     for ik in range(gkk.kpt.shape[0]):
        #     epceout.write(f'     k = {kpt[0]:14.8f} {kpt[1]:14.8f} {kpt[2]:14.8f}\n')
        #     epceout.write('   Enk = ' + ' '.join(f'{ekn[ik, ibnd]:12.5f}' for ibnd in range(nbndout)) + '\n\n')
        #     for iq in range(gkk.nq):
        #         epceout.write(f'     q = {gkk.qlist[iq][0]:14.8f} {gkk.qlist[iq][1]:14.8f} {gkk.qlist[iq][2]:14.8f}\n')
        #         epceout.write('  EPCE = ' + ' '.join(f'{epce[iq, ibnd, ik]:12.5f}' for ibnd in range(nbndout)) + '\n\n')
        #     epceout.write('--------------------\n')
        # epceout.close()
    
    @mpi_watch
    @simple_timer('Spectral analysis done, total wall time = {t}')
    def zpr_spectral_analysis(self, savefname, ws_spectral, s_spectral, ibnd, eta=0.01):
        ws_spectral = np.array(ws_spectral) / hartree2ev / 1000 # mev -> hartree
        s_spectral /= 1000 * hartree2ev # mev -> hartree

        dEnk_wn = np.empty((len(ws_spectral), self.gkkdata.nbandk))
        dEFM_wn = np.empty((len(ws_spectral), self.gkkdata.nbandk))
        dEDW_wn = np.empty((len(ws_spectral), self.gkkdata.nbandk))

        sys.stdout = open(os.devnull, 'w') # disable output message
        for iw, w in enumerate(ws_spectral):
            self.calc_ren([0.], eta=eta, spectral=True, w_spectral=w, s_spectral=s_spectral)
            dEnk_wn[iw, :] = self.dEnk[:, 0]
            dEFM_wn[iw, :] = self.dEFM[:, 0]
            dEDW_wn[iw, :] = self.dEDW[:, 0]
        sys.stdout.close()
        sys.stdout = sys.__stdout__

        if not is_master(): return
        
        fout = open(savefname, 'w')
        fout.write('Zero point renormalization spectral analysis\n')
        fout.write(f'Band index = {ibnd}\n')
        fout.write(f'Unit of dEnk(f): 1 (no unit)\n\n')
        fout.write(' ifreq      freq(meV)             dEnk(f)             dEFM(f)             dEDW(f)\n')
        
        ws_spectral *= 1000 * hartree2ev
        ibnd -= 1
        for iw, w in enumerate(ws_spectral):
            fout.write(f'{iw+1:5d} {w:15.10f} {dEnk_wn[iw, ibnd]:19.10e} {dEFM_wn[iw, ibnd]:19.10e} {dEDW_wn[iw, ibnd]:19.10e}\n')
        fout.close()

def calc_FM_py(gkkmode2, e_skn, f_skn, nB_qvT, w_qv, wtq, kqmap, kidx, eta):
    r'''
    \sum_{q\nu}\sum_m w_q |g_{mn\nu}(k,q)|^2 
    (\frac{1-f_{mk+q}+n_{q\nu}}{\epsilon_{nk}-\epsilon_{mk+q}-\hbar\omega_{q\nu}+i\eta} + 
    \frac{f_{mk+q}+n_{q\nu}}{\epsilon_{nk}-\epsilon_{mk+q}+\hbar\omega_{q\nu}-i\eta})

    gkkmode2: (q, bandk+q, v, k, bandk)
    e_skn: (spin, k, nk)
    f_skn: (spin, k, nk)
    nB_qvT: (q, v, T)
    w_qv: (q, v)
    wtq: (q)
    nval: number of valence bands
    eta: small imaginary number
    '''

    nq = gkkmode2.shape[0]
    nv = w_qv.shape[1]
    nbndk = gkkmode2.shape[4]
    nbndkq = gkkmode2.shape[1]
    nbnde = e_skn.shape[2]
    nT = nB_qvT.shape[2]

    assert nq == gkkmode2.shape[0] == nB_qvT.shape[0] == w_qv.shape[0] == len(wtq)
    assert nv == gkkmode2.shape[2] == nB_qvT.shape[1] == w_qv.shape[1]
    assert e_skn.shape[0] == 1
    assert gkkmode2.shape[3] == 1
    assert nbndk <= nbndkq <= nbnde
    assert f_skn.shape == e_skn.shape

    fm = np.zeros((nbndk, nT), dtype='c16')

    for iT in range(nT):
        for ibn in range(nbndk):
            for iq in range(nq):
                for iv in range(nv):
                    for ibm in range(nbndkq):
                        fm[ibn, iT] += wtq[iq] * gkkmode2[iq, ibm, iv, 0, ibn] * (
                            (1. - f_skn[0,kqmap[iq],ibm] + nB_qvT[iq,iv,iT]) / \
                            (e_skn[0,kidx,ibn] - e_skn[0,kqmap[iq],ibm] - w_qv[iq,iv] + 1j*eta) + \
                            (f_skn[0,kqmap[iq],ibm] + nB_qvT[iq,iv,iT]) / \
                            (e_skn[0,kidx,ibn] - e_skn[0,kqmap[iq],ibm] + w_qv[iq,iv] -1j*eta))

    return np.ascontiguousarray(fm.real)

def calc_DWria_py(gkkmat_G, Uqvad, ekn, nB_qvT, wtq, eta):
    r'''
    -\sum_{q\nu}\sum_m'\sum_{\kappa\alpha}\sum_{\kappa'\alpha'} 
    w_q \frac{g_{mn\kappa\alpha}(k,\Gamma)g^*_{mn\kappa'\alpha'}(k,\Gamma)}{\epsilon_{nk}-\epsilon_{mk}} 
    (u^*_{\kappa\alpha'}(q\nu)u_{\kappa\alpha}(q\nu) + u^*_{\kappa'\alpha'}(q\nu)u_{\kappa'\alpha}(q\nu))
    (n_{q\nu}+\frac{1}{2})

    gkkmat_G: (mk+q, nat, ndir, bandk) only at q=Gamma and single k point
    Uqvad: (q, v, nat, ndir)
    ekn: (nspin, nk) only at single k point
    nB_qvT: (q, v, T)
    wtq: (q)
    '''

    nq = Uqvad.shape[0]
    nat = Uqvad.shape[2]
    nT = nB_qvT.shape[2]
    nbndk = gkkmat_G.shape[3]
    nbndkq = gkkmat_G.shape[0]
    nbnde = ekn.shape[1]

    assert nq == Uqvad.shape[0] == nB_qvT.shape[0] == len(wtq)
    assert nat == gkkmat_G.shape[1] == Uqvad.shape[2]
    assert 3*nat == Uqvad.shape[1] == nB_qvT.shape[1]
    assert ekn.shape[0] == 1
    assert nbndk <= nbndkq <= nbnde

    dwria = np.zeros((nbndk, nT), dtype='c16')

    for iT in range(nT):
        for iat in range(nat):
            for jat in range(nat):
                for idir in range(3):
                    for jdir in range(3):
                        tmp = 0.
                        for iq in range(nq):
                            for iv in range(3*nat):
                                tmp += wtq[iq] * (nB_qvT[iq,iv,iT] + .5) * \
                                (Uqvad[iq, iv, iat, jdir].conj() * Uqvad[iq, iv, iat, idir] + 
                                 Uqvad[iq, iv, jat, jdir].conj() * Uqvad[iq, iv, jat, idir])
                        for ibn in range(nbndk):
                            for ibm in range(nbndkq):
                                dwria[ibn, iT] += - tmp * \
                                gkkmat_G[ibm, iat, idir, ibn] * gkkmat_G[ibm, jat, jdir, ibn].conj() / \
                                (ekn[0,ibn] - ekn[0,ibm] + 1j*eta)
    
    return np.ascontiguousarray(dwria.real)
