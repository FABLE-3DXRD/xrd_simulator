from scipy.special import iv
# Von Mises-Fisher orientation distribution function.
# https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
def ODF(x, q):
    kappa = 0.064 * x.dot(x) + 0.001
    mu    = np.array([ 0.,  0.95689793,  0.28706938, -0.0440173 ])
    p     = 4.0
    I     = iv( p/2 - 1, kappa)
    Cp    = kappa**(p/2 - 1) / ( (2*np.pi)**(p/2) * I )
    return Cp * ( np.exp( kappa * np.dot( q, mu ) ) + np.exp( -(kappa * np.dot( q, mu )) ) )