import numpy as np
from .from_gab.ddbfile import DdbFile
from .utils import QgridFile, same_kpt, is_master

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
        