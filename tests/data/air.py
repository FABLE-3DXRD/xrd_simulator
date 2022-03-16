import numpy as np

"""
Absorption coefficent for dry air as provided by physics.nist.gov.

Reference: https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/air.html
collected: 16 March 2022

"""

air = np.array( [
       #   MeV        mu/rho    mu_en/rho
       [1.00000e-3,  3.606e+3,  3.599e+3],
       [1.50000e-3,  1.191e+3,  1.188e+3],
       [2.00000e-3,  5.279e+2,  5.262e+2],
       [3.00000e-3,  1.625e+2,  1.614e+2],
       [3.20290e-3,  1.340e+2,  1.330e+2],
       [3.20290e-3,  1.485e+2,  1.460e+2],
       [4.00000e-3,  7.788e+1,  7.636e+1],
       [5.00000e-3,  4.027e+1,  3.931e+1],
       [6.00000e-3,  2.341e+1,  2.270e+1],
       [8.00000e-3,  9.921e+0,  9.446e+0],
       [1.00000e-2,  5.120e+0,  4.742e+0],
       [1.50000e-2,  1.614e+0,  1.334e+0],
       [2.00000e-2,  7.779e-1,  5.389e-1],
       [3.00000e-2,  3.538e-1,  1.537e-1],
       [4.00000e-2,  2.485e-1,  6.833e-2],
       [5.00000e-2,  2.080e-1,  4.098e-2],
       [6.00000e-2,  1.875e-1,  3.041e-2],
       [8.00000e-2,  1.662e-1,  2.407e-2],
       [1.00000e-1,  1.541e-1,  2.325e-2],
       [1.50000e-1,  1.356e-1,  2.496e-2],
       [2.00000e-1,  1.233e-1,  2.672e-2],
       [3.00000e-1,  1.067e-1,  2.872e-2],
       [4.00000e-1,  9.549e-2,  2.949e-2],
       [5.00000e-1,  8.712e-2,  2.966e-2],
       [6.00000e-1,  8.055e-2,  2.953e-2],
       [8.00000e-1,  7.074e-2,  2.882e-2],
       [1.00000e+0,  6.358e-2,  2.789e-2],
       [1.25000e+0,  5.687e-2,  2.666e-2],
       [1.50000e+0,  5.175e-2,  2.547e-2],
       [2.00000e+0,  4.447e-2,  2.345e-2],
       [3.00000e+0,  3.581e-2,  2.057e-2],
       [4.00000e+0,  3.079e-2,  1.870e-2],
       [5.00000e+0,  2.751e-2,  1.740e-2],
       [6.00000e+0,  2.522e-2,  1.647e-2],
       [8.00000e+0,  2.225e-2,  1.525e-2],
       [1.00000e+1,  2.045e-2,  1.450e-2],
       [1.50000e+1,  1.810e-2,  1.353e-2],
       [2.00000e+1,  1.705e-2,  1.311e-2]
       ])

density = 1225. / 1e6 # g / cm^3

# mu is in units : cm^2 / g = (1/cm) / (g/cm^3) 
# i.e lengths in Beers law should be in cm and density in g/cm^3

if __name__=='__main__':
   import matplotlib.pyplot as plt
   # E = 43.469 keV
   v = 2.485e-1 # 40 kev
   l = 14.293828756189224 # must be in cm!
   print( np.exp( -v*density*l ) )
   plt.figure()
   plt.plot(air[:,0], air[:,1], 'k', label=r'$\mu$ / $\rho$' )
   plt.plot(air[:,0], air[:,2], '--k', label=r'$\mu_{en}$ / $\rho$' )
   plt.grid(True)
   plt.title('Air, dry (near sea level)')
   plt.xlabel('MeV')
   plt.ylabel(r'$\mu$ / $\rho$   (cm$^2$/g)')
   plt.yscale('log')
   plt.xscale('log')
   plt.legend()
   plt.show()
