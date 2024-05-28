     
import numpy as np

cimport cython
from libc.complex cimport conj
from libc.math cimport exp, sqrt, M_PI

ctypedef Py_ssize_t pyidx

cdef double sqrt2pi = sqrt(2 * M_PI)
@cython.cdivision(True)
cdef double stdnorm(double x, double mu, double sigma) noexcept:
    return exp( -((x-mu)/sigma)**2 / 2.) / sqrt2pi / sigma

@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.nonecheck(False) 
@cython.cdivision(True)
def calc_FM(double[:, :, :, :, ::1] gkkmode2, 
            double[:, :, ::1] e_skn, 
            double[:, :, ::1] f_skn, 
            double[:, :, ::1] nB_qvT, 
            double[:, ::1] w_qv, 
            double[::1] wtq, 
            long[::1] kqmap, 
            pyidx nbndkqmax, long kidx, double eta, double w_spectral, double s_spectral):
    r'''
    \sum_{q\nu}\sum_m w_q |g_{mn\nu}(k,q)|^2 
    (\frac{1-f_{mk+q}+n_{q\nu}}{\epsilon_{nk}-\epsilon_{mk+q}-\hbar\omega_{q\nu}+i\eta} + 
    \frac{f_{mk+q}+n_{q\nu}}{\epsilon_{nk}-\epsilon_{mk+q}+\hbar\omega_{q\nu}-i\eta})

    gkkmode2: (q, bandk+q, v, k, bandk) square of gkk matrix in mode basis (hartree)
    e_skn: (spin, k, nk) band energy (hartree)
    f_skn: (spin, k, nk) band occupation / 2
    nB_qvT: (q, v, T) bose-einstein occupation
    w_qv: (q, v) phonon frequency (hartree)
    wtq: (q) weight of q points
    kqmap: (q) maps k+q to k in enk
    nbndkqmax: number of bands at k+q in summation 
    kidx: index of k point for ZPR calculation in e_skn
    eta: small number for imaginary part in denominator
    w_spectral: frequency (hartree) for spectral analysis
    s_spectral: delta function width (hartree) for spectral analysis
    '''

    cdef pyidx nq = gkkmode2.shape[0]
    cdef pyidx nv = w_qv.shape[1]
    cdef pyidx nbndk = gkkmode2.shape[4]
    cdef pyidx nbndkq = gkkmode2.shape[1]
    cdef pyidx nbnde = e_skn.shape[2]
    cdef pyidx nT = nB_qvT.shape[2]
    cdef pyidx iT, ibn, iq, iv, ibm, kqidx
    cdef double complex tmp

    assert nq == gkkmode2.shape[0] == nB_qvT.shape[0] == w_qv.shape[0] == len(wtq)
    assert nv == gkkmode2.shape[2] == nB_qvT.shape[1] == w_qv.shape[1]
    assert e_skn.shape[0] == 1
    assert gkkmode2.shape[3] == 1
    assert nbndk <= nbndkq <= nbnde
    assert f_skn.shape[0] == e_skn.shape[0]
    assert f_skn.shape[1] == e_skn.shape[1]
    assert f_skn.shape[2] == e_skn.shape[2]

    if nbndkqmax < nbndkq:
        nbndkq = nbndkqmax

    fm = np.zeros((nbndk, nT), dtype='c16')
    cdef double complex[:, ::1] fm_v = fm   

    for iT in range(nT):
        for ibn in range(nbndk):
            for iq in range(nq):
                kqidx = kqmap[iq]
                for iv in range(nv):
                    tmp = 0.
                    for ibm in range(nbndkq):
                        tmp = tmp + wtq[iq] * gkkmode2[iq, ibm, iv, 0, ibn] * (
                            (1. - f_skn[0,kqidx,ibm] + nB_qvT[iq,iv,iT]) / \
                            (e_skn[0,kidx,ibn] - e_skn[0,kqidx,ibm] - w_qv[iq,iv] + 1j*eta) + \
                            (f_skn[0,kqidx,ibm] + nB_qvT[iq,iv,iT]) / \
                            (e_skn[0,kidx,ibn] - e_skn[0,kqidx,ibm] + w_qv[iq,iv] - 1j*eta))
                    if s_spectral > 0:
                        tmp = tmp * stdnorm(w_qv[iq, iv], w_spectral, s_spectral)
                    fm_v[ibn, iT] = fm_v[ibn, iT] + tmp

    return np.ascontiguousarray(fm.real)

@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.nonecheck(False) 
@cython.cdivision(True)
def calc_DWria(double complex[:, :, :, ::1] gkkmat_G, 
               double complex[:, :, :, ::1] Uqvad, 
               double[:, ::1] ekn, 
               double[:, :, ::1] nB_qvT, 
               double[::1] wtq, 
               double[:, ::1] w_qv, 
               pyidx nbndkqmax, double eta, double w_spectral, double s_spectral):
    r'''
    -\sum_{q\nu}\sum_m'\sum_{\kappa\alpha}\sum_{\kappa'\alpha'} 
    w_q \frac{g_{mn\kappa\alpha}(k,\Gamma)g^*_{mn\kappa'\alpha'}(k,\Gamma)}{\epsilon_{nk}-\epsilon_{mk}} 
    (u^*_{\kappa\alpha'}(q\nu)u_{\kappa\alpha}(q\nu) + u^*_{\kappa'\alpha'}(q\nu)u_{\kappa'\alpha}(q\nu))
    (n_{q\nu}+\frac{1}{2})

    gkkmat_G: (mk+q, nat, ndir, bandk) gkk matrix (hartree) in reduced coordinates, only at q=Gamma and single k point
    Uqvad: (q, v, nat, ndir) reduced phonon mode displacement in reduced coordinates
    ekn: (nspin, nk) band energy (hartree) only at single k point
    nB_qvT: (q, v, T) bose-einstein occupation number
    wtq: (q) weight of q point
    w_qv: (q, v) phonon frequency (hartree)
    nbndkqmax: number of bands at k+q in summation 
    eta: small number for imaginary part in denominator
    w_spectral: frequency (hartree) for spectral analysis, only effective when s_spectral > 0
    s_spectral: delta function width (hartree) for spectral analysis, only effective when s_spectral > 0
    '''

    cdef pyidx nq = Uqvad.shape[0]
    cdef pyidx nat = Uqvad.shape[2]
    cdef pyidx nT = nB_qvT.shape[2]
    cdef pyidx nbndk = gkkmat_G.shape[3]
    cdef pyidx nbndkq = gkkmat_G.shape[0]
    cdef pyidx nbnde = ekn.shape[1]
    cdef pyidx iT, iat, jat, idir, jdir
    cdef double complex tmp1, tmp2

    assert nq == Uqvad.shape[0] == nB_qvT.shape[0] == len(wtq)
    assert nat == gkkmat_G.shape[1] == Uqvad.shape[2]
    assert 3*nat == Uqvad.shape[1] == nB_qvT.shape[1]
    assert ekn.shape[0] == 1
    assert nbndk <= nbndkq <= nbnde

    dwria = np.zeros((nbndk, nT), dtype='c16')
    cdef double complex[:, ::1] dwria_v = dwria

    if nbndkqmax < nbndkq:
        nbndkq = nbndkqmax

    for iT in range(nT):
        for iat in range(nat):
            for jat in range(nat):
                for idir in range(3):
                    for jdir in range(3):
                        tmp1 = 0.
                        for iq in range(nq):
                            for iv in range(3*nat):
                                tmp2 = wtq[iq] * (nB_qvT[iq,iv,iT] + .5) * \
                                (conj(Uqvad[iq, iv, iat, jdir]) * Uqvad[iq, iv, iat, idir] + 
                                 conj(Uqvad[iq, iv, jat, jdir]) * Uqvad[iq, iv, jat, idir])
                                if s_spectral > 0:
                                    tmp2 = tmp2 * stdnorm(w_qv[iq, iv], w_spectral, s_spectral)
                                tmp1 = tmp1 + tmp2
                        for ibn in range(nbndk):
                            for ibm in range(nbndkq):
                                dwria_v[ibn, iT] = dwria_v[ibn, iT] - tmp1 * \
                                gkkmat_G[ibm, iat, idir, ibn] * conj(gkkmat_G[ibm, jat, jdir, ibn]) / \
                                (ekn[0,ibn] - ekn[0,ibm] + 1j*eta)
    
    return np.ascontiguousarray(dwria.real)