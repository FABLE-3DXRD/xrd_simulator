import unittest
import numpy as np
from xfab import tools
from xrd_simulator.polycrystal import Polycrystal
import matplotlib.pyplot as plt
from scipy.signal import convolve
from xfab import tools

np.random.seed(5)
U = np.eye(3,3)
strain_tensor = np.zeros((6,))
unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
B = tools.epsilon_to_b( strain_tensor, unit_cell )
wavelength = 0.285227
D = 142938.28756189224 #microns
detector = np.zeros((2048,2048))
pixsize = 75 #microns

x = np.array([1.,0.,0.])
omega = np.linspace(0., np.pi, 9)
ks = np.array( [ np.array([[np.cos(om),-np.sin(om),0],[np.sin(om),np.cos(om),0],[0,0,1]]).dot(x) for om in omega])
ks = 2*np.pi*ks/wavelength
thetas  =[]
pc = Polycrystal(None, None, None)

for _ in range(10): # sample of 10 crystals

    phi1, PHI, phi2 = np.random.rand(3,)*2*np.pi
    U = tools.euler_to_u(phi1, PHI, phi2)
    for h in range(-5,5):
        for k in range(-5,5):
            for l in range(-5,5):
                G_hkl = np.array( [h,k,l] )
                for i in range(len(ks)-1):
                    kprime1, kprime2 = pc._get_kprimes( ks[i], ks[i+1], U, B, G_hkl, wavelength )
                    for j,kprime in enumerate([kprime1, kprime2]):
                        if kprime is not None:

                            alpha   = pc._get_alpha(ks[i], ks[i+1], wavelength)
                            rhat    = pc._get_rhat(ks[i], ks[i+1])
                            G  = pc._get_G(U, B, G_hkl)
                            theta   = pc._get_bragg_angle(G, wavelength)
                            c_0, c_1, c_2 = pc._get_tangens_half_angle_equation(ks[i], theta, G, rhat ) 
                            s1, s2 = pc._find_solutions_to_tangens_half_angle_equation( c_0, c_1, c_2, alpha )
                            ss = [s1,s2][j]
                            ang = alpha*ss

                            s = np.sin( -(omega[i]+ang) )
                            c = np.cos( -(omega[i]+ang) )
                            R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
                            kprime = R.dot(kprime)
                            
                            #(s * kprime/np.linalg.norm(kprime)).dot(x) = D
                            khat = kprime/np.linalg.norm(kprime)
                            s = D / khat[0]

                            yd = khat[1]*s
                            zd = khat[2]*s

                            col = ( -(yd/pixsize) + detector.shape[1]//2 ).astype(int)
                            row = ( -(zd/pixsize) + detector.shape[0]//2 ).astype(int)

                            if col>0 and col<detector.shape[1] and row>0 and row<detector.shape[0]:
                                detector[col, row] += 1

                            thetas.append(theta)

plt.hist(np.degrees(thetas),180)
plt.show()

kernel = np.ones((5,5))
detector = convolve(detector, kernel, mode='full', method='auto')
plt.imshow(detector, cmap='gray')
plt.title("Hits: "+str(np.sum(detector)/np.sum(kernel) ))
plt.show()
                        
