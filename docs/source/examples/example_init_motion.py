import numpy as np
import os
from xrd_simulator.motion import RigidBodyMotion

# The motion is described by a rotation axis and a translation vector
motion = RigidBodyMotion(rotation_axis=np.array([0, 1/np.sqrt(2), -1/np.sqrt(2)]),
                         rotation_angle=np.radians(2.0),
                         translation=np.array([123, -153.3, 3.42]))

# The motion can be applied to transform a set of points
# Points should have shape (N, 3) where N is the number of points
points = np.random.rand(10, 3)
transformed_points = motion( points, time=0.421 )

# The motion may be saved to disc for later usage.
artifacts_dir = os.path.join(os.path.dirname(__file__), 'test_artifacts')
os.makedirs(artifacts_dir, exist_ok=True)
motion.save(os.path.join(artifacts_dir, 'my_motion'))
motion_loaded_from_disc = RigidBodyMotion.load(os.path.join(artifacts_dir, 'my_motion.motion'))