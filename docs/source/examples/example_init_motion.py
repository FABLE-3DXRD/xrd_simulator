import numpy as np
from xrd_simulator.motion import RigidBodyMotion

motion = RigidBodyMotion(rotation_axis=np.array([0, 1/np.sqrt(2), -1/np.sqrt(2)]),
                         rotation_angle=np.radians(2.0),
                         translation=np.array([123, -153.3, 3.42]))