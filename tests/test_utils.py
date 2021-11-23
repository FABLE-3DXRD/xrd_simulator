import unittest
import numpy as np
from xrd_simulator import utils

class TestUtils(unittest.TestCase):

    def setUp(self):
        np.random.seed(10) # changes all randomisation in the test

    def test_clip_line_with_convex_polyhedron(self):
        line_points    = np.ascontiguousarray( [[-1.,0.2,0.2],[-1.,0.4,0.6]] )
        line_direction = np.ascontiguousarray( [1.0, 0.0, 0.0] )
        line_direction = line_direction/np.linalg.norm(line_direction)
        plane_points   = np.ascontiguousarray( [[0.,0.5,0.5], [1,0.5,0.5], [0.5,0.5,0.], [0.5,0.5,1.], [0.5,0,0.5], [0.5,1.,0.5]] )
        plane_normals  = np.ascontiguousarray( [[-1.,0.,0.],[1.,0.,0.], [0.,0.,-1.],[0.,0.,1.], [0.,-1.,0.],[0.,1.,0.]] )
        clip_lengths   = utils.clip_line_with_convex_polyhedron( line_points, line_direction, plane_points, plane_normals )
        for l in clip_lengths: 
            self.assertAlmostEqual(l, 1.0, msg="Projection through unity cube should give unity clip length")

        line_direction = np.ascontiguousarray( [1.0, 0.2, 0.1] )
        line_direction = line_direction/np.linalg.norm(line_direction)
        clip_lengths   = utils.clip_line_with_convex_polyhedron( line_points, line_direction, plane_points, plane_normals )
        for l in clip_lengths: 
            self.assertGreater(l, 1.0, msg="Titlted projection through unity cube should give greater than unity clip length")

if __name__ == '__main__':
    unittest.main()