import numpy as np
from xrd_simulator.motion import RigidBodyMotion

# The motion is described by a rotation axis and a translation vector
motion = RigidBodyMotion(rotation_axis=np.array([0, 1/np.sqrt(2), -1/np.sqrt(2)]),
                         rotation_angle=np.radians(2.0),
                         translation=np.array([123, -153.3, 3.42]))

# The motion can be applied to transform a set of points
points = np.random.rand(3, 10)
transformed_points = motion( points, time=0.421 )

# The motion may be saved to disc for later usage.
motion.save('my_motion')
motion_loaded_from_disc = motion.load('my_motion.motion')