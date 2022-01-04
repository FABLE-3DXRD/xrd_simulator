import numpy as np
from xrd_simulator.xfab import tools
import matplotlib.pyplot as plt
from scipy.signal import convolve
from xrd_simulator import laue
from xrd_simulator.motion import _RodriguezRotator

"""Simple simulation of 50 random quartz grains in powder diffraction style only using laue.py
and no spatial functions, i.e not considering grain shapes and the like. This is a check to
see that we have our basic crystal equations under control.
"""

np.random.seed(5)
U = np.eye(3, 3)
strain_tensor = np.zeros((6,))
unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
B = tools.epsilon_to_b(strain_tensor, unit_cell)
wavelength = 0.285227
D = 142938.28756189224  # microns
detector = np.zeros((1024, 1024))
pixsize = 75  # microns

x = np.array([1., 0., 0.])
omega = np.linspace(0., np.pi / 2., 3)
ks = np.array([np.array([[np.cos(om), -np.sin(om), 0],
              [np.sin(om), np.cos(om), 0], [0, 0, 1]]).dot(x) for om in omega])
ks = 2 * np.pi * ks / wavelength

hklrange = 3
for ii in range(50):  # sample of 10 crystals
    print('Crystal no ', ii, 'of total ', 50)
    phi1, PHI, phi2 = np.random.rand(3,) * 2 * np.pi
    U = tools.euler_to_u(phi1, PHI, phi2)
    for hmiller in range(-hklrange, hklrange + 1):
        for kmiller in range(-hklrange, hklrange + 1):
            for lmiller in range(-hklrange, hklrange + 1):
                G_hkl = np.array([hmiller, kmiller, lmiller])
                for i in range(len(ks) - 1):

                    G = laue.get_G(U, B, G_hkl)
                    theta = laue.get_bragg_angle(G, wavelength)

                    rotation_axis = np.array([0, 0, 1])
                    rotator = _RodriguezRotator(rotation_axis)
                    rotation_angle = omega[i + 1] - omega[i]
                    c_0, c_1, c_2 = laue.get_tangens_half_angle_equation(
                        ks[i], theta, G, rotation_axis)
                    s1, s2 = laue.find_solutions_to_tangens_half_angle_equation(
                        c_0, c_1, c_2, rotation_angle)

                    for j, s in enumerate([s1, s2]):
                        if s is not None:

                            wavevector = rotator(ks[i], s * rotation_angle)
                            kprime = G + wavevector

                            ang = rotation_angle * s
                            sin = np.sin(-(omega[i] + ang))
                            cos = np.cos(-(omega[i] + ang))
                            R = np.array(
                                [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
                            kprime = R.dot(kprime)
                            khat = kprime / np.linalg.norm(kprime)
                            sd = D / khat[0]

                            yd = khat[1] * sd
                            zd = khat[2] * sd

                            col = (-(yd / pixsize) +
                                   detector.shape[1] // 2).astype(int)
                            row = (-(zd / pixsize) +
                                   detector.shape[0] // 2).astype(int)

                            if col > 0 and col < detector.shape[1] and row > 0 and row < detector.shape[0]:
                                detector[col, row] += 1


kernel = np.ones((4, 4))
detector = convolve(detector, kernel, mode='full', method='auto')
plt.imshow(detector, cmap='gray')
plt.title("Hits: " + str(np.sum(detector) / np.sum(kernel)))
plt.show()
