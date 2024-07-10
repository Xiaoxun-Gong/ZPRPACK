import numpy as np
import netCDF4 as nc

from .utils import make_kkmap, is_master, QgridFile
from .constants import hartree2ev

class EnkData:
    def __init__(self, path_root, nband=None, eqp_option=None, readwfq=False, prefix='phonon-q'):
        self.path_root = path_root


        gsr_file = nc.Dataset(path_root + '/Wfk/out_data/odat_GSR.nc', 'r')
        self.nk = len(gsr_file.dimensions['number_of_kpoints'])
        self.nband = len(gsr_file.dimensions['max_number_of_states'])
        if (nband is not None) and (nband < self.nband): self.nband = nband
        self.kpt = np.array(gsr_file.variables['reduced_coordinates_of_kpoints'])
        self.nspin = 1
        # self.wtk = np.array(gsr_file.variables['kpoint_weights']) # wtk is not used
        self.enk_dft = np.array(gsr_file.variables['eigenvalues'][:, :, 0:self.nband])
        self.onk = np.array(gsr_file.variables['occupations'][:, :, 0:self.nband])
        gsr_file.close()
        
        if readwfq:
            assert eqp_option is None
            # note: no need exclude duplicate k points
            qgrid = QgridFile(path_root, prefix=prefix)
            self.nk += qgrid.nq_loc
            kpt_add = np.empty((qgrid.nq_loc, 3), dtype='f8')
            enk_add = np.empty((self.nspin, qgrid.nq_loc, self.nband), dtype='f8')
            onk_add = np.empty((self.nspin, qgrid.nq_loc, self.nband), dtype='f8')
            for iq in qgrid.qrange_loc:
                gsr_file = nc.Dataset(f'{qgrid.folder_name(iq)}/Wfq/out_data/odat_GSR.nc', 'r')
                k = np.array(gsr_file.variables['reduced_coordinates_of_kpoints'])
                assert k.shape == (1, 3)
                kpt_add[iq-qgrid.qrange_loc.start, :] = k[0, :]
                tmp = np.array(gsr_file.variables['eigenvalues'][:, 0, 0:self.nband])
                enk_add[:, iq-qgrid.qrange_loc.start, :] = tmp
                tmp = np.array(gsr_file.variables['occupations'][:, 0, 0:self.nband])
                onk_add[:, iq-qgrid.qrange_loc.start, :] = tmp
                gsr_file.close()
            self.kpt = np.concatenate((self.kpt, kpt_add), axis=0)
            self.enk_dft = np.concatenate((self.enk_dft, enk_add), axis=1)
            self.onk = np.concatenate((self.onk, onk_add), axis=1)
        
        assert np.all((self.onk==2.0) + (self.onk==0.0))

        self.enk = self.enk_dft.copy()

        if eqp_option is not None:
            kpt_sigma, eqprange, eqp = read_sigmahp(f'{path_root}/sigma_hp.log', band_option=eqp_option)
            # !!!
            # kpt_sigma = np.concatenate((kpt_sigma, -kpt_sigma), axis=0)
            # eqp = np.concatenate((eqp, eqp), axis=0)
            # !!!
            assert len(kpt_sigma) == len(self.kpt), 'K grid mismatch between DFT and GW'
            kkmap = make_kkmap(kpt_sigma, self.kpt, symopt=0) # make kkmap w/o TR symmetry
            self.eqp = eqp[None, kkmap, :] / hartree2ev
            self.eqprange = eqprange
            nbandmin, nbandmax = eqprange
            nbandmax = min(nbandmax, self.nband)
            self.enk[:, :, nbandmin-1:nbandmax] = self.eqp[:, :, :nbandmax-nbandmin+1]


def read_sigmahp(filename, band_option='eqp1'):
    # doesn't support spin
    # Notice that nbndmin and nbndmax are 1-based indices!
    if is_master(): print(f'Using band option {band_option} to read sigma_hp\n')
    # len1 = 124 # position to the end of Eqp0
    
    if band_option == 'eqp1':
        eindex = 9
    elif band_option == 'eqp0':
        eindex = 8
    elif band_option=='dft' or band_option=='dft-check':
        eindex = 1
    else:
        raise NotImplementedError(f'{band_option}')
  
    # scan the file to find nk, nbndmin and nbndmax
    sigfile = open(filename)
    nk = 0
    line = sigfile.readline()
    while line:
        line_sp = line.split()
        if len(line_sp)==3 and line_sp[0]=='band_index':
            nbandmin = int(line_sp[1])
            nbandmax = int(line_sp[2])
        elif len(line_sp)==11 and line_sp[0]=='n':
            len1 = line.find('Eqp0') + 4 # position to the end of Eqp0
        elif len(line_sp)>0 and line_sp[0]=='k' and line_sp[1]=='=':
            nk += 1
        line = sigfile.readline()
    sigfile.close()
    
    nband = nbandmax - nbandmin + 1
    kpt = np.zeros((nk, 3))
    band_energy = np.zeros((nk, nband))

    sigfile = open(filename)
    
    line = sigfile.readline()
    ik = 0
    while line:
        line_sp = line.split()
            
        if len(line_sp)>0 and line_sp[0]=='k' and line_sp[1]=='=':
            kpt[ik, 0] = float(line_sp[2]); kpt[ik, 1] = float(line_sp[3]); kpt[ik, 2] = float(line_sp[4])
            
            for _ in range(3): line = sigfile.readline()
            
            line_sp = line.split()
            for ibnd in range(nbandmin-1, nbandmax):
                ibnd1 = int(line_sp[0]) - 1
                assert ibnd1 == ibnd
                eqp1invalid = False
                if len(line_sp) < 11:
                    # print(line_sp)
                    line = line[:len1]
                    line_sp = line.split()
                    assert len(line_sp) == 9
                    eqp1invalid = True
                elif not 0 < float(line_sp[10]) <= 1:
                    # print(line)
                    eqp1invalid = True
                eindex1 = 8 if eindex == 9 and eqp1invalid else eindex
                energy = float(line_sp[eindex1])
                band_energy[ik, ibnd-nbandmin+1] = energy

                line = sigfile.readline()
                line_sp = line.split()
            # print(ik, ibnd)
            # print(line_sp)
            ik += 1
            
        line = sigfile.readline()
  
    sigfile.close()

    return kpt, (nbandmin, nbandmax), band_energy