import numpy as np
import os, sys, time
from numba import njit
from scipy import optimize

class _HiddenPrints:
    """Simple class to enable running code without printing using python with statements.

    This is a hack to supress printing from imported packages over which we do not have full controll. (xfab.tools)
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def contained_by_intervals(s, intervals):
    for bracket in intervals:
        if s >= bracket[0] and s <= bracket[1]:
            return True
    return False

def CIFopen( ciffile ):
    from CifFile import ReadCif
    cf = ReadCif(ciffile)
    return cf[ list(cf.keys())[0] ]

def print_progress(progress_fraction, message):
    """Print a progress bar in the executing shell terminal.

    Args:
        progress_fraction (:obj:`float`): progress between 0 and 1.
        message (:obj:`str`): Optional message prepend the loading bar with. (max 55 characters)

    """
    assert len(message) <= 55., "The provided message to print is too long, max 55 characters allowed."
    progress_in_precent = np.round(100*progress_fraction,1)
    progress_bar_length = int( progress_fraction*40 )
    sys.stdout.write( "\r{0}{1} | {2}>{3} |".format(message, " "*(55-len(message)),"="*progress_bar_length, " "*(40-progress_bar_length))+" "+str(progress_in_precent)+"%" )
    if progress_fraction!=1.0: 
        sys.stdout.flush()
    else:
        sys.stdout.write( "\n" )

@njit
def clip_line_with_convex_polyhedron( line_points, line_direction, plane_points, plane_normals ):
    """Compute clip-lengths of a set of parallel lines clipped by a convex polyhedron defined by a series of 2d planes.

        For algorihtm description see:
            Mike Cyrus and Jay Beck. “Generalized two- and three-dimensional clipping”. (1978)

        Args:
            line_points (:obj:`numpy array`): base points of rays (exterior to polyhedron), ```shape=(n,3)```
            line_direction  (:obj:`numpy array`): normalised ray direction (all rays have the same direction),  ```shape=(3,)```
            plane_points (:obj:`numpy array`): point in each polyhedron face plane, ```shape=(m,3)```
            plane_normals (:obj:`numpy array`): outwards element face normals, shape: ```shape=(m,3)```

        Returns:
            clip_lengths (:obj:`numpy array`) : intersection lengths.  ```shape=(n,)```

    """
    #TODO: Add test
    clip_lengths = np.zeros((line_points.shape[0],))
    t2 = np.dot( plane_normals, line_direction ) 
    te_mask = t2<0
    tl_mask = t2>0
    for i,line_point in enumerate( line_points ): 

        # find paramteric line-plane intersect based on orthogonal equations
        t1 = np.sum( np.multiply( plane_points-line_point, plane_normals ), axis=1 ) # 
        ti = t1/t2
        
        # Sort intersections as potential entry and exit points
        te = np.max( ti[te_mask] )
        tl = np.min( ti[tl_mask] )

        if tl>te:
            clip_lengths[i] = tl-te

    return clip_lengths
