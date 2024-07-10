import numpy as np
from .from_gab.ddbfile import DdbFile
from .utils import QgridFile, same_kpt, is_master, make_kkmap
from .constants import hartree2ev, bohr2ang

class DDBData(QgridFile):
    def __init__(self, *args, **kwargs):
        super(DDBData, self).__init__(*args, **kwargs)

        if is_master(): print('Reading dynamical matrices\n')
        self.ddbfile_loc = []
        for iq in self.qrange_loc:
            ddbpath = f'{self.folder_name(iq)}/Dvscf/out_data/odat_DDB.nc'
            ddbfile = DdbFile(ddbpath)
            assert same_kpt(self.qpt[iq], ddbfile.qred)
            self.ddbfile_loc.append(DdbFile(ddbpath))
        
        for iq_loc in range(self.nq_loc):
            self.ddbfile_loc[iq_loc].get_reduced_displ() # asr?
            
        self.Uqvad = np.stack([self.ddbfile_loc[iq_loc].polvec.transpose(0, 2, 1) 
                               for iq_loc in range(self.nq_loc)]) # (nq_loc, nmode, natom, ndir)
        self.Uqvad = np.ascontiguousarray(self.Uqvad)
        self.w_qv = np.stack([self.ddbfile_loc[iq_loc].omega.real for iq_loc in range(self.nq_loc)]) # notice negative ones!
        

# ===============================
# Below are functions for phonon band structure calculation using Fourier interpolation
# ===============================

def get_kpts_from_path(hsks, nkpts_path):
    hsks = np.array(hsks)
    nkpts_path = np.array(nkpts_path)

    assert len(hsks) == len(nkpts_path)
    assert nkpts_path[-1] == 1
    nk = np.sum(nkpts_path)
    kpts_full = np.empty((nk, 3), dtype='f8')
    pos = 0
    ikhs = -1
    for ikhs in range(len(hsks)-1):
        wk = nkpts_path[ikhs]
        arange = np.arange(0., 1., 1/wk)
        kstart = hsks[ikhs]
        kend = hsks[ikhs+1]
        kpts_full[pos:pos+wk, :] = kstart[None, :] + (kend - kstart)[None, :] * arange[:, None]
        pos += wk
    kpts_full[pos, :] = hsks[ikhs+1]
    pos += 1
    assert pos == nk

    hskpos = list(np.concatenate(([0], np.cumsum(nkpts_path)[:-1])))

    return nk, kpts_full, hskpos

def index_traverse(indices):
    '''
    Example:
    >>> index_traverse([np.arange(0, 2, 1), np.arange(0, 3, 1)])
    array([[0, 0],
           [0, 1],
           [0, 2],
           [1, 0],
           [1, 1],
           [1, 2]])
    '''
    return np.stack(np.meshgrid(*indices, indexing='ij'), axis=-1).reshape(-1, len(indices))

class Structure:
    '''
    Attributes:
    ---------
    rprim(3, 3): R_ij is the jth cartesian component of ith lattice vector, in bohr
    gprim(3, 3): G_ij is the jth cartesian component of ith primitive reciprocal lattice vector (NOT SCALED BY 2pi)
    atomic_species: (nspc), unique atomic numbers
    nspc: int, number of species
    natom: int, number of atoms
    
    atomic_positions_red(natom, 3): fractional coordinates
    atomic_positions_cart(natom, 3): in bohr
    cell_volume: in bohr^3
    
    _trans_to_uc(natom, 3): how to translate the atoms to [0, 1) unit cell
    '''
    
    def __init__(self, rprim, atomic_numbers, atomic_positions, efermi=0.0, atomic_positions_is_cart=False):
        '''
        Parameters
        ---------
        rprim(3, 3): R_ij is the jth cartesian component of ith lattice vector, in bohr
        atomic_numbers: (natom) atomic numbers
        atomic_positions_red(natom, 3)
        efermi: in hartree
        '''
        
        rprim = np.array(rprim)
        self.rprim = rprim
        self.gprim = np.linalg.inv(rprim.T)
        
        atomic_numbers = np.array(atomic_numbers, dtype=int)
        self.atomic_numbers = atomic_numbers
        self.atomic_species = np.sort(np.unique(atomic_numbers))
        self.natom = len(atomic_numbers)
        self.nspc = len(self.atomic_species)
        
        assert len(atomic_positions) == self.natom
        if not atomic_positions_is_cart:
            self.atomic_positions_red = np.array(atomic_positions)
            self.atomic_positions_cart = self.atomic_positions_red @ self.rprim
        else:
            self.atomic_positions_cart = np.array(atomic_positions)
            self.atomic_positions_red = self.atomic_positions_cart @ self.gprim.T
        
        trans_to_uc, self.atomic_positions_red_uc = np.divmod(self.atomic_positions_red, 1)
        self._trans_to_uc = trans_to_uc.astype(int)
        self.atomic_positions_cart_uc = self.atomic_positions_red_uc @ self.rprim
        
        self.cell_volume = np.abs(np.linalg.det(rprim))
        
        self.efermi = efermi  
    
    def trans_uc_to_original(self, translation, iatom1, iatom2):
        return translation + self._trans_to_uc[iatom1, :] - self._trans_to_uc[iatom2, :]

def to_wigner_seitz(rprim, pos, pos_is_cart):
    '''
    Convert all pos to Wigner-Seitz unit cell

    Returns:
    ---------
    pos_ws(npos, 3): pos in WS unit cell. Will be in cartesian distance is original pos is in cartesian distance.
    trans_ws(npos, 3): primitive translations needed to convert pos to WS unit cell
    '''

    stru = Structure(rprim, [1 for _ in range(len(pos))], pos, atomic_positions_is_cart=pos_is_cart)
    corners_red = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0],
                            [1, 1, 0],
                            [1, 0, 1],
                            [0, 1, 1],
                            [1, 1, 1]])
    corners_cart = corners_red @ stru.rprim
    dists = np.linalg.norm(corners_cart[None, :, :] - stru.atomic_positions_cart_uc[:, None, :], axis=2) # (pos, corners)
    argmin_dists = np.argmin(dists, axis=1)
    trans_ws = -stru._trans_to_uc - corners_red[argmin_dists, :]
    pos_ws = stru.atomic_positions_red + trans_ws
    if pos_is_cart:
        pos_ws = pos_ws @ stru.rprim
    
    return pos_ws, trans_ws

def write(kpts, hskpos, hsksymbol, eigs, eigfile='eig.dat'):
    '''
    eigs(nk, nbnd)
    '''

    nbnd = eigs.shape[1]
    nk = eigs.shape[0]

    f = open(eigfile, 'w')
    f.write('Band energies in eV\n')
    f.write('      nk    nbnd\n')
    f.write(f'{nk:8d}{nbnd:8d}\n')
    for ikpt in range(nk):
        kpt = kpts[ikpt]
        f.write(f'{kpt[0]:13.9f}{kpt[1]:13.9f}{kpt[2]:13.9f}{nbnd:8d}')
        if ikpt in hskpos:
            f.write('  ' + hsksymbol[hskpos.index(ikpt)] + '\n')
        else:
            f.write('\n')
        for ibnd in range(nbnd):
            f.write(f'{1:8d}{ibnd+1:8d}{eigs[ikpt, ibnd]*hartree2ev:15.9f}\n')
    f.close()

def calc_phbands(ddbdata, qgrid, hsqs, nqpts_path, qptsymbol, loto=None, asr=True, savedir='./'):
    '''
    Phonon band structure calculation using Fourier interpolation

    loto: use another DDB.nc file to replace dynamical matrix at gamma
    '''

    msg = 'DDBData must be on a full gamma-centered q-grid without symmetry'
    assert len(qgrid) == 3
    assert ddbdata.nq == np.prod(qgrid), msg
    # assert np.allclose(ddbdata.wtq, 1/ddbdata.nq, atol=1e-6), msg
    assert ddbdata.nq == ddbdata.nq_loc, 'Does not support parallelization yet'

    nq_fi, qpts_fi, hsqpos = get_kpts_from_path(hsqs, nqpts_path)
    assert len(qptsymbol) == len(hsqs)

    natom = ddbdata.ddbfile_loc[0].natom
    rprim = ddbdata.ddbfile_loc[0].rprim
    xred = ddbdata.ddbfile_loc[0].xred
    qgrid = np.array(qgrid)

    # create qmesh that looks like (nq1*nq2*nq3, 3) where nq3 is the fastest index
    R_co = index_traverse([np.arange(0, qgrid[0], 1),
                           np.arange(0, qgrid[1], 1),
                           np.arange(0, qgrid[2], 1)]) # R_xyz is defined in [0, nq_xyz)
    qmesh_co = R_co / qgrid[None, :]

    qqmap = make_kkmap(ddbdata.qpt, qmesh_co)

    dynmat_qco = np.empty((ddbdata.nq, 3*natom, 3*natom), dtype='c16')

    for iq in range(ddbdata.nq):
        ddbfile = ddbdata.ddbfile_loc[qqmap[iq]]
        dynmat = ddbfile.get_mass_scaled_dynmat_cart()
        if ddbfile.is_gamma:
            if loto is not None: # loto splitting
                ddbfile = DdbFile(loto)
                dynmat = ddbfile.get_mass_scaled_dynmat_cart()
            # apply asr
            if asr:
                eigval, eigvect = np.linalg.eigh(dynmat)
                eigval[0:3] = 0.0
                dynmat = np.sum(eigvect[:, None, :] * eigval[None, None, :] * eigvect.conj()[None, :, :], axis=2)
        dynmat_qco[iq, :, :] = dynmat
    
    # Inverse of Giustino 2017 Eq. 14, and Gonze & Lee 1997 Eq. 10
    # fconst_Rco is the mass-scaled force constant on R_co
    fconst_Rco = np.fft.fftn(dynmat_qco.reshape(*qgrid, 3*natom, 3*natom), s=qgrid, axes=(0, 1, 2), norm='forward')
    fconst_Rco = fconst_Rco.reshape(-1, 3*natom, 3*natom)

    # Adding a BvK lattice vector doesn't change R_co
    # Here we need to find the R_co having the smallest length within that freedom 
    Rcosmallest_ij = [] 
    for iatm in range(natom):
        tmplist = []
        for jatm in range(iatm+1):
            Rrel = (R_co + xred[jatm] - xred[iatm]) / qgrid[None, :]
            _, trans_ws = to_wigner_seitz(rprim, Rrel, False)
            tmplist.append(R_co + trans_ws * qgrid[None, :])
        Rcosmallest_ij.append(tmplist)
    # print(Rcosmallest_ij)

    # Do inverse FT and solve eigenvalue
    eigval_q = np.empty((nq_fi, 3*natom), dtype='f8')
    for iq_fi in range(nq_fi):
        qpt = qpts_fi[iq_fi]
        dynmat = np.empty((3*natom, 3*natom), dtype='c16')
        for iatm in range(natom):
            for jatm in range(iatm+1):
                slice_i = slice(iatm*3, iatm*3+3)
                slice_j = slice(jatm*3, jatm*3+3)
                R = Rcosmallest_ij[iatm][jatm]
                dynmat[slice_i, slice_j] = np.sum(fconst_Rco[:, slice_i, slice_j] * \
                                                  np.exp(2*np.pi*1j * np.dot(R, qpt))[:, None, None],
                                                  axis=0)
                if jatm < iatm:
                    dynmat[slice_j, slice_i] = dynmat[slice_i, slice_j].T.conj()
        
        eigval, eigvect = np.linalg.eigh(dynmat)
        # TODO: scale eigenvectors, ddbfile.py line 266
        eigval_q[iq_fi, :] = np.sqrt(np.abs(eigval)) * np.sign(eigval)

    write(qpts_fi, hsqpos, qptsymbol, eigval_q, f'{savedir}/eig.dat')
    np.savetxt(f'{savedir}/lat.dat', rprim.T * bohr2ang)
