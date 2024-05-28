import numpy as np
import netCDF4 as nc

from .constants import hartree2ev
from .utils import QgridFile, same_kpt, tqdm_mpi_tofile, is_master

class GKKData(QgridFile):
    def __init__(self, *args, use_gwpt=False, nbandk=None, nbandkq=None, **kwargs):
        super(GKKData, self).__init__(*args, **kwargs)

        gkkfile_G = nc.Dataset(f'{self.folder_name(0)}/Epc/out_data/odat_GKK.nc', 'r')
        self.nbandkq = len(gkkfile_G.dimensions['max_number_of_states'])
        if (nbandkq is not None) and (nbandkq < self.nbandkq): self.nbandkq = nbandkq
        self.nbandk = self.nbandkq
        if (nbandk is not None) and (nbandk < self.nbandk): self.nbandk = nbandk
        self.natom = len(gkkfile_G.dimensions['number_of_atoms'])
        self.nkpt = len(gkkfile_G.dimensions['number_of_kpoints'])
        assert self.nkpt == 1
        self.kpt = np.array(gkkfile_G.variables['reduced_coordinates_of_kpoints'])
        self.nsppol = len(gkkfile_G.dimensions['number_of_spins'])
        assert self.nsppol == 1
        gkkfile_G.close()

        # gkk matrix: (nq, nband_kq, natom, ndir, nkpt, nband*nsppol)
        self.gkk_dfpt = np.zeros((self.nq_loc, self.nbandkq, self.natom, 3, self.nkpt, self.nbandk), dtype='c16') # self.nband*self.nsppol
        self.read_gkk_dfpt()
        
        self.gkk_matrix = self.gkk_dfpt.copy()
        if use_gwpt:
            self.read_gkk_gwcor()

    def read_gkk_dfpt(self):
        ''' Read gkk matrix elements from Abinit DFPT output into gkkmat_q '''

        if is_master(): print(f'Reading DFPT el-ph matrices of {self.nq_loc} q points')
        for iq in tqdm_mpi_tofile(self.qrange_loc):
            # print('reading', iq)
            gkkmat_q = self.gkk_dfpt[iq-self.qrange_loc.start, ...]

            gkk_file = nc.Dataset(f'{self.folder_name(iq)}/Epc/out_data/odat_GKK.nc', 'r')

            q = np.array(gkk_file.variables['current_q_point'])
            assert same_kpt(q, self.qpt[iq])
            k = np.array(gkk_file.variables['reduced_coordinates_of_kpoints'])
            assert np.all(same_kpt(k, self.kpt))
            
            gkkmat_q.real = gkk_file.variables['second_derivative_eigenenergies_actif'][:self.nbandkq,:,:,:,:self.nbandk*2:2]
            gkkmat_q.imag = gkk_file.variables['second_derivative_eigenenergies_actif'][:self.nbandkq,:,:,:,1:self.nbandk*2:2]
              
            # gkkmat_q *= hartree2eV 
            # we keep everything in hartree amu

            gkk_file.close()
      
    def read_gkk_gwcor(self):
        ''' Read gkk correction from BerkleyGW GWPT output
            By default, it will read sigma(E) evaluated at both bra and ket bare energies, and take average.
            According to current results, the difference is small, but it is still good to do averaging physically. '''
        # This is the VERSION that BGW-GWPT has already gotten rid of the braket settings
        # Therefore there is no need to braket = m and n loop, just need to read once (GWPT calculation also do once)

        # self.gkk_gwcor[:, :, :, :, :, :] = 0.0 + 0.0j
        if is_master(): print(f'Reading GWPT el-ph matrices of {self.nq_loc} q points')
        for iq in tqdm_mpi_tofile(range(self.nq_loc)):

          # print('Reading sigma in phonon-q-' + str(qlabel))
          for iatom in range(self.natom):
            for idir in range(3):
              idx = iatom * 3 + idir + 1

              sigma_path = f'{self.folder_name(iq + self.qrange_loc.start)}/sigma_{idx}/sigma.out'
              # sigma_path = self.path_root + '/phonon-q-' + str(qlabel) + '/sigma_' + str(idx) + '/sigma.out'
              #print(sigma_path)
              with open(sigma_path) as sigma_file:
                ik = -1
                lines = []
                for line in sigma_file:
                  lines.append(line)
                for iline in range(len(lines)):
                  ln = lines[iline].split()
                  #print(ln)
                  if len(ln) == 10:
                    if ln[0]=='Number' and ln[1]=='of' and ln[2]=='bands' and ln[3]=='to' and ln[4]=='compute' \
                        and ln[5]=='diagonal' and ln[6]=='self-energy' and ln[7]=='matrix' and ln[8]=='elements:':
                      ndiag = int(ln[9])
                      #print('ndiag', self.sigma.ndiag)
                    elif ln[0]=='Number' and ln[1]=='of' and ln[2]=='off-diagonal' and ln[3]=='bands' and ln[4]=='to' \
                        and ln[5]=='compute' and ln[6]=='self-energy' and ln[7]=='matrix' and ln[8]=='elements:':
                      noffdiag = int(ln[9])
                      #print('noffdiag', self.sigma.noffdiag)

                  # Because of the for loop, ndiag and noffdiag will be initialized before reading matrix elements
                  # so we can stay in the same for loop
                  if len(ln) == 4:
                    if ln[0]=='Electron-phonon' and ln[1]=='band' and ln[2]=='index' and ln[3]=='off-diagonal:':
                      ik += 1
                      for ioff in range(noffdiag):
                        # GW_cor = (dSig(l) + dSig(lp))/2.0 - dVxc
                        # note GWPT sigma.out formating has been changed, adding lp, dSig1(lp), dSig3 columns
                        # gwpt2epw (this python script) modified accordingly
                        oneln = lines[iline+3+ioff*2+1].split()
                        ibn = int(oneln[0])
                        ibm = int(oneln[1])
                        gwcor_one_real = (float(oneln[10]) + float(oneln[12]))/2.0 - float(oneln[5])

                        oneln = lines[iline+3+ioff*2+2].split()
                        if (ibn != int(oneln[0])) or (ibm != int(oneln[1])):
                            raise ValueError('Reading GWPT sigma.out real and imag index mismatch!')
                        try:
                          gwcor_one_imag = (float(oneln[10]) + float(oneln[12]))/2.0 - float(oneln[5])
                        except:
                          print(oneln)
                          exit()

                        gwcor_one = gwcor_one_real + 1.0j * gwcor_one_imag
                        self.gkk_matrix[iq, ibn-1, iatom, idir, ik, ibm-1] += gwcor_one / hartree2ev

