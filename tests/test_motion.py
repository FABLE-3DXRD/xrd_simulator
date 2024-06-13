
import unittest
import os
import numpy as np
from xrd_simulator.motion import RigidBodyMotion


class TestMotion(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_init(self):
        rotation_axis = np.random.rand(3,)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.random.rand(1,) * np.pi
        translation = np.random.rand(3,)
        motion = RigidBodyMotion(rotation_axis, rotation_angle, translation)
        self.assertTrue(motion is not None)

    def test_rotate(self):
        rotation_axis = np.array([1., 0, 0])
        rotation_angle = np.pi / 2.
        translation = np.random.rand(3,)
        motion = RigidBodyMotion(rotation_axis, rotation_angle, translation)

        x = motion.rotate(np.array([1., 0, 0]), time=1)
        z = motion.rotate(np.array([0, 1, 0]), time=1)
        y = motion.rotate(np.array([0, 0, -1]), time=1)

        self.assertAlmostEqual(np.linalg.norm(
            x - np.array([1., 0, 0])), 0, msg='Error in rotator')
        self.assertAlmostEqual(np.linalg.norm(
            y - np.array([0, 1, 0])), 0, msg='Error in rotator')
        self.assertAlmostEqual(np.linalg.norm(
            z - np.array([0, 0, 1])), 0, msg='Error in rotator')

    def test_translate(self):
        rotation_axis = np.random.rand(3,)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.random.rand(1,)
        translation = np.random.rand(3,) - 0.5
        motion = RigidBodyMotion(rotation_axis, rotation_angle, translation)

        v0 = np.random.rand(3,) - 0.5
        v = motion.translate(v0, time=0.4323)

        self.assertAlmostEqual(
            np.linalg.norm(
                v -
                translation *
                0.4323 -
                v0),
            0,
            msg='Error in translation')

    def test_call(self):
        rotation_axis = np.array([1., 0, 0])
        rotation_angle = np.pi / 2.
        translation = np.array([1., 0, 0])
        motion = RigidBodyMotion(rotation_axis, rotation_angle, translation)

        y = np.array([0, 1, 0])
        v = motion(y, time=1)

        self.assertAlmostEqual(np.linalg.norm(
            v - np.array([1, 0, 1])), 0, msg='Error in rigid body transformation')

    def test_save_and_load(self):
        rotation_axis = np.random.rand(3,)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.random.rand(1,) * np.pi
        translation = np.random.rand(3,)
        motion = RigidBodyMotion(rotation_axis, rotation_angle, translation)
        path = os.path.join(
            os.path.join(
                os.path.dirname(__file__),
                'data'),
            'my_motion')
        motion.save(path)
        motion = RigidBodyMotion.load(path+'.motion')
        self.assertTrue(
            np.allclose(
                motion.rotation_axis,
                rotation_axis),
            msg='Data corrupted on save and load')
        os.remove(path+'.motion')
    
    def test_inverse(self):
        rotation_axis = np.random.rand(3,)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.random.rand() * np.pi
        translation = np.random.rand(3,)
        origin = np.array([1.23,-1234,3.1])

        motion = RigidBodyMotion(rotation_axis, rotation_angle, translation, origin)
        inverse_motion = motion.inverse()

        self.assertTrue( inverse_motion!=motion )
        for i in range(3):
            self.assertAlmostEqual( inverse_motion.rotation_axis[i], -motion.rotation_axis[i] )
            self.assertAlmostEqual( inverse_motion.translation[i], -motion.translation[i] )
        self.assertAlmostEqual( inverse_motion.rotation_angle, motion.rotation_angle )

        points_0 = np.random.rand(22, 3)
        points = motion.rotate(points_0, time=0.243687)
        points = inverse_motion.rotate(points, time=0.243687)

        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                self.assertAlmostEqual( points_0[i,j], points[i,j] )

    def test_origin(self):
        # TODO: implement
        rotation_axis = np.array([1., 0, 0])
        rotation_angle = np.pi / 2.
        translation = np.array([1., 0, 0])
        origin = np.array([0, 1, 0])
        motion = RigidBodyMotion(rotation_axis, rotation_angle, translation, origin)

        y = np.array([0, 1, 0])
        v = motion(y, time=1)

        self.assertAlmostEqual(np.linalg.norm(
            v - np.array([1, 1, 0])), 0, msg='Error in rigid body transformation')

        rotation_axis = np.array([1., 0, 0])
        rotation_angle = np.pi / 2.
        translation = np.random.rand(3,)
        origin = np.array([0, 0, 1])
        motion = RigidBodyMotion(rotation_axis, rotation_angle, translation, origin)
        z = motion.rotate(np.array([0, 1, 0]), time=1) # rotation should NOT respect the origin!
        self.assertAlmostEqual(np.linalg.norm(
            z - np.array([0, 0, 1])), 0, msg='Error in rotator')

        rotation_axis = np.random.rand(3,)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.random.rand(1,)
        translation = np.random.rand(3,) - 0.5
        motion = RigidBodyMotion(rotation_axis, rotation_angle, translation, origin = np.array([1.2, -1213, 1]))

        v0 = np.random.rand(3,) - 0.5
        v = motion.translate(v0, time=0.4323)

        self.assertAlmostEqual(
            np.linalg.norm(
                v -
                translation *
                0.4323 -
                v0),
            0,
            msg='Error in translation')

if __name__ == '__main__':
    unittest.main()
