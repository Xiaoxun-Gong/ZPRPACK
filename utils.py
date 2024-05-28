import os, sys
import time, datetime
import traceback
from math import ceil
from tqdm import tqdm
from tqdm.utils import disp_len, _unicode
import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ModuleNotFoundError:
    MPI = None
    comm = None

def is_master(comm=comm):
    return comm is None or comm.rank == 0

def distrib_grps(length, ngrps, displ_last_elem=False):
    '''distribute a vector with `length` into `ngrps` groups'''
    avg, rem = divmod(length, ngrps)
    count = [avg + 1 if p < rem else avg for p in range(ngrps)]
    if displ_last_elem:
        displ = np.cumsum([0] + count)
    else:
        displ = np.cumsum([0] + count[:-1])
    count = np.array(count)
    return count, displ

def distrib_vec(length, displ_last_elem=False, comm=comm):
    '''length of displ will be comm.size if displ_last_elem=False, or comm.size+1 if displ_last_elem=True'''
    if comm is not None:
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1
    count, displ = distrib_grps(length, size, displ_last_elem=displ_last_elem)
    return rank, count, displ

def mpi_watch(f):
    """Decorator. Terminate all mpi process if an exception is raised."""
    def g(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            sys.stdout.flush()
            sys.stderr.flush()
            time.sleep(1) # wait for other nodes if they have sth to print
            if not is_master():
                time.sleep(5) # wait for root to get here first
                print(f'Error from proc {comm.rank}:')
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            if comm is None:
                exit(1)
            else:
                comm.Abort(1)
    return g


class QgridFile:
    def __init__(self, path_root, prefix='phonon-q'):
        '''qrange_loc: range object'''

        self.path_root = path_root
        self.prefix = prefix
        self.qpt = np.loadtxt(f'{path_root}/q.list')
        if len(self.qpt.shape) == 1:
            self.qpt = self.qpt[None, :]
        assert len(self.qpt.shape) == 2
        self.nq = len(self.qpt)
        if os.path.isfile(f'{path_root}/q.weights'):
            self.wtq = np.loadtxt(f'{path_root}/q.weights')
        else:
            self.wtq = np.ones(self.nq)
        self.wtq /= np.sum(self.wtq)

        if os.path.isfile(f'{path_root}/q.labels'):
            with open(f'{path_root}/q.labels') as f:
                self.qlabels = f.read().split()
        else:
            self.qlabels = list(map(str,range(1, self.nq+1)))

        rank, count, displ = distrib_vec(self.nq, displ_last_elem=True)
        qrange_loc = range(displ[rank], displ[rank+1])
        # qrange_loc = range(0, 3) # debug

        self.qrange_loc = qrange_loc
        self.qrange_all = displ
        self.nq_loc = qrange_loc.stop - qrange_loc.start
        self.qpt_loc = self.qpt[qrange_loc, :]
        self.wtq_loc = self.wtq[qrange_loc]
    
    def folder_name(self, iq):
        return f'{self.path_root}/{self.prefix}-{self.qlabels[iq]}/'


def same_kpt(kpt1, kpt2):
    '''
    Parameters:
    ---------
      kpt1: [(extra_dimensions), 3]
      kpt2: [(extra_dimensions), 3]
    
    Returns:
    ---------
      same_kpt [(extra_dimensions)] containing boolean values
    '''
    eps = 1.0e-5
    kdiff = kpt1 - kpt2
    kdiff1BZ = firstBZ(kdiff)
    samekpt = np.all(np.abs(kdiff1BZ) < eps, axis=-1)
    return samekpt


def same_kpt_sym(kpt1, kpt2, symopt=0):
    '''
    Parameters:
    ---------
      kpt1: [(extra_dimensions), 3]
      kpt2: [(extra_dimensions), 3]
      symopt: 0 - no symmetry; 1 - only time-reversal symmetry
    
    Returns:
    ---------
      same_kpt [(extra_dimensions)] containing boolean values
    '''
    assert symopt in [0,  1]
    if symopt == 0:
        samekpt = same_kpt(kpt1, kpt2)
    elif symopt == 1:
        samekpt = same_kpt(kpt1, kpt2)
        samekpt += same_kpt(-kpt1, kpt2)
    return samekpt


def firstBZ(kpt):
    '''
    find kpt in 1BZ (-0.5, 0.5]
    '''
    return -np.divmod(-kpt+0.5, 1)[1] + 0.5


def find_kidx(kpt, kqpt, symopt=0, allow_multiple=True):
    '''
    Find the index of kqpt (one point) in the kpt (all points).
    If multiple match exists, only return the index of the first match.
    If there's no match, then return -1
    '''

    kidx = np.where(same_kpt_sym(kpt, kqpt, symopt=symopt))[0]
    if not ((kidx.ndim==1) and (kidx.shape[0]>=1)):
        msg = f'\nCannot find kpt {kqpt} among k points:\n'
        msg += str(kpt)
        raise ValueError(msg)
    if (not allow_multiple) and (kidx.shape[0]>1):
        msg = f'Multiple kpt {kqpt} found among k points:\n'
        msg += str(kpt)
        raise ValueError(msg)
    return kidx[0].item()


def make_kkmap(kpt1, kpt2, symopt=0):
    '''
    maps every point in kpt1 to kpt2
    
    Returns:
    ---------
    kkmap[nk2]: kkmap[i] is the index of the k-point in kpt1 corresponding to the ith k-point in kpt2
    '''
    # nk = kpt1.shape[0]
    # assert kpt2.shape[0] == nk
    nk2 = kpt2.shape[0]
    
    kkmap = np.zeros(nk2, dtype=int)
    for ik, kqpt in enumerate(kpt2):
        kkmap[ik] = find_kidx(kpt1, kqpt, symopt=symopt)
      
    return kkmap


class tqdm_mpi_tofile(tqdm):
    def __init__(self, iterable=None, **kwargs):
        if 'total' in kwargs:
            total = kwargs['total']
        else:
            total = len(iterable)
        miniters = ceil(total / 10)
        kwargs['miniters'] = miniters
        kwargs['file'] = sys.stdout
        kwargs['delay'] = 1e-5
        kwargs['leave'] = True # total%miniters!=0
        kwargs['bar_format'] = '{l_bar}{bar:40}{r_bar}{bar:-10b}'
        disable = kwargs.pop('disable', False)
        kwargs['disable'] = not is_master() or disable
        # prevent tqdm from changing miniters by itself,
        # see /home1/09019/xiaoxun/.local/lib/python3.9/site-packages/tqdm/_monitor.py", line 82
        kwargs['mininterval'] = 1
        kwargs['maxinterval'] = 1e10 
        super().__init__(iterable=iterable, **kwargs)
        # self.dynamic_miniters = False
        # self.miniters = miniters
        self.start_time = time.time()
    
    # @property
    # def miniters(self):
    #     return self._miniters
    
    # @miniters.setter
    # def miniters(self, v):
    #     self._miniters = v
    #     traceback.print_stack()
    
    
    @staticmethod
    def status_printer(file):
        """
        Manage the printing and in-place updating of a line of characters.
        Note that if the string is longer than a line, then in-place
        updating may not work (it will print a new line at each refresh).
        """
        fp = file
        fp_flush = getattr(fp, 'flush', lambda: None)  # pragma: no cover
        if fp in (sys.stderr, sys.stdout):
            getattr(sys.stderr, 'flush', lambda: None)()
            getattr(sys.stdout, 'flush', lambda: None)()

        def fp_write(s):
            fp.write(_unicode(s))
            fp_flush()

        last_len = [0]

        def print_status(s):
            len_s = disp_len(s)
            fp_write(s + (' ' * max(last_len[0] - len_s, 0)) + '\n') # always write new bar in new line
            last_len[0] = len_s

        return print_status

    def close(self):
        """Cleanup and (if leave=False) close the progressbar."""
        if self.disable:
            return

        # Prevent multiple closures
        self.disable = True

        # decrement instance pos and remove from internal set
        pos = abs(self.pos)
        self._decr_instances(self)

        if self.last_print_t < self.start_t + self.delay:
            # haven't ever displayed; nothing to clear
            return

        # GUI mode
        if getattr(self, 'sp', None) is None:
            return

        # annoyingly, _supports_unicode isn't good enough
        def fp_write(s):
            self.fp.write(_unicode(s))

        try:
            fp_write('')
        except ValueError as e:
            if 'closed' in str(e):
                return
            raise  # pragma: no cover

        leave = pos == 0 if self.leave is None else self.leave

        with self._lock:
            if leave:
                # stats for overall rate (no weighted average)
                self._ema_dt = lambda: None
                self.display(pos=0)
                # fp_write('\n')
            # else:
                # clear previous display
                # if self.display(msg='', pos=pos) and not pos:
                    # fp_write('\r')
        
        elapsed = time.time() - self.start_time
        fp_write(f'Done, elapsed time: {elapsed:4.1f}s.\n\n') # write timing message

    # def set_postfix(self, ordered_dict=None, refresh=False, **kwargs):
    #     return super().set_postfix(self, ordered_dict=ordered_dict, refresh=refresh, **kwargs)
    
    def update(self, n=1):
        if not self.disable:
            if self.n + n >= self.total:
                self.n += n
                self.last_print_n = self.n
                self.last_print_t = time.time()
                return
        return super().update(n=n)
    
    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""

        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable

        # If the bar is disabled, then just walk the iterable
        # (note: keep this check outside the loop for performance)
        if self.disable:
            for obj in iterable:
                yield obj
            return

        try:
            for obj in iterable:
                yield obj
                self.update()
        finally:
            self.close()

def simple_timer(formatstr):
    def timer_decorator(f):
        def g(*args, **kwargs):
            start_time = time.time()
            ret = f(*args, **kwargs)
            if is_master():
                total_time = datetime.timedelta(seconds=int(time.time()-start_time))
                # print(f'total wall time: {str(total_time)}\n')
                print(formatstr.format(t=str(total_time)))
            return ret
        return g
    return timer_decorator
