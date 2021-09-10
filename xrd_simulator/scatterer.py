import numpy as np 
import matplotlib.pyplot as plt

class Scatterer(object):

    """Defines a scattering single crystal as a convex polyhedra.

    Args:
        convex_hull (:obj:`scipy.spatial.ConvexHull`): Object describing the convex hull of the scatterer.
        kprime (:obj:`string`): Scattering vector, i.e the wavevector pointing in the direction of diffraction.
        s (:obj:`float`): Parametric value in range [0,1] where 0 corresponds to a beam with wavevector k1
            while s=1 to a beam with wavevector k2. The s value of the scatterer implies what detector
            position is applicable during diffraction.

    Attributes:
        convex_hull (:obj:`scipy.spatial.ConvexHull`): Object describing the convex hull of the scatterer.
        kprime (:obj:`string`): Scattering vector, i.e the wavevector pointing in the direction of diffraction.
        s (:obj:`float`): Parametric value in range [0,1] where 0 corresponds to a beam with wavevector k1
            while s=1 to a beam with wavevector k2. The s value of the scatterer implies what detector
            position is applicable during diffraction.

    """ 

    def __init__(self, convex_hull, kprime, s ):
        self.convex_hull = convex_hull
        self.kprime = kprime
        self.s = s

    def get_centroid(self):
        """Get centroid of the scattering region.

        Returns:
            centroid (:obj:`numpy array`) ```shape=(3,)```

        """
        return np.mean( convex_hull.points, axis=0 )

    def get_volume(self):
        """Get volume of the scattering region.

        Returns:
            volume (:obj:`float`)

        """
        return convex_hull.volume