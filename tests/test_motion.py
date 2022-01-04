
import unittest
import numpy as np
from xrd_simulator.motion import RigidBodyMotion


class TestPhase(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
