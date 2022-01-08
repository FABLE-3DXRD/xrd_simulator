"""
xfab.tools module is a collection of functions
for doing calculation useful in crystallography
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as n
from math import degrees
from xrd_simulator.xfab import checks


def find_omega_general(g_w, twoth, w_x, w_y):
    """
    For gw find the omega rotation (in radians) around an axis
    tilted by w_x radians around x (chi) and w_y radians around y (wedge)
    Furthermore find eta (in radians)
    Soren Schmidt, implemented by Jette Oddershede
    """

    assert abs(n.dot(g_w, g_w) - n.sin(twoth / 2)**2) < 1e-9, \
        'g-vector must have length sin(theta)'
    w_mat_x = n.array([[1, 0, 0],
                       [0, n.cos(w_x), -n.sin(w_x)],
                       [0, n.sin(w_x), n.cos(w_x)]])
    w_mat_y = n.array([[n.cos(w_y), 0, n.sin(w_y)],
                       [0, 1, 0],
                       [-n.sin(w_y), 0, n.cos(w_y)]])
    r_mat = n.dot(w_mat_x, w_mat_y)

    a = g_w[0] * r_mat[0][0] + g_w[1] * r_mat[0][1]
    b = g_w[0] * r_mat[1][0] - g_w[1] * r_mat[0][0]
    c = - n.dot(g_w, g_w) - g_w[2] * r_mat[0][2]
    d = a * a + b * b - c * c

    omega = []
    eta = []
    if d < 0:
        pass
    else:
        sq_d = n.sqrt(d)

        for i in range(2):
            cosomega = (a * c + b * sq_d * (-1)**i) / (a * a + b * b)
            sinomega = (b * c + a * sq_d * (-1)**(i + 1)) / (a * a + b * b)
            omega.append(n.arctan2(sinomega, cosomega))
            if omega[i] > n.pi:
                omega[i] = omega[i] - 2 * n.pi
            omega_mat = form_omega_mat_general(omega[i], w_x, w_y)
            g_t = n.dot(omega_mat, g_w)
            sineta = -2 * g_t[1] / n.sin(twoth)
            coseta = 2 * g_t[2] / n.sin(twoth)
            eta.append(n.arctan2(sineta, coseta))

    return n.array(omega), n.array(eta)


def find_omega_quart(g_w, twoth, w_x, w_y):
    """
    For gw find the omega rotation (in radians) around an axis
    tilted by w_x radians around x (chi) and w_y radians around y (wedge)
    Furthermore find eta (in radians)
    Soren Schmidt, implemented by Jette Oddershede
    """

    assert abs(n.dot(g_w, g_w) - n.sin(twoth / 2)**2) < 1e-9, \
        'g-vector must have length sin(theta)'
    w_mat_x = n.array([[1, 0, 0],
                       [0, n.cos(w_x), -n.sin(w_x)],
                       [0, n.sin(w_x), n.cos(w_x)]])
    w_mat_y = n.array([[n.cos(w_y), 0, n.sin(w_y)],
                       [0, 1, 0],
                       [-n.sin(w_y), 0, n.cos(w_y)]])
    normal = n.dot(w_mat_x, n.dot(w_mat_y, n.array([0, 0, 1])))

    a = g_w[0] * (1 - normal[0]**2) - \
        g_w[1] * normal[0] * normal[1] - \
        g_w[2] * normal[0] * normal[2]
    b = g_w[2] * normal[1] - g_w[1] * normal[2]
    c = - n.dot(g_w, g_w) - g_w[0] * normal[0]**2 - \
        g_w[1] * normal[0] * normal[1] - \
        g_w[2] * normal[0] * normal[2]
    d = a * a + b * b - c * c

    omega = []
    eta = []
    if d < 0:
        pass
    else:
        sq_d = n.sqrt(d)

        for i in range(2):
            cosomega = (a * c + b * sq_d * (-1)**i) / (a * a + b * b)
            sinomega = (b * c + a * sq_d * (-1)**(i + 1)) / (a * a + b * b)
            omega.append(n.arctan2(sinomega, cosomega))
            if omega[i] > n.pi:
                omega[i] = omega[i] - 2 * n.pi
            omega_mat = quart_to_omega(omega[i] * 180. / n.pi, w_x, w_y)
            g_t = n.dot(omega_mat, g_w)
            sineta = -2 * g_t[1] / n.sin(twoth)
            coseta = 2 * g_t[2] / n.sin(twoth)
            eta.append(n.arctan2(sineta, coseta))

    return n.array(omega), n.array(eta)


def find_omega_wedge(g_w, twoth, wedge):
    """
    calculate the omega and eta angles for a g-vector g_w given twotheta of
    g_w is provided. The calculation takes a possible wedge angle intp account.
    This code is a translation of c-code from GrainSpotter by Soren Schmidt
    """

    # Normalize G-vector
    g_w = g_w / n.sqrt(n.dot(g_w, g_w))

    costth = n.cos(twoth)
    sintth = n.sin(twoth)
    cosfactor = costth - 1
    length = n.sqrt(-2 * cosfactor)
    sinwedge = n.sin(wedge)
    coswedge = n.cos(wedge)
    # Determine cos(eta)
    coseta = (g_w[2] * length + sinwedge * cosfactor) / coswedge / sintth

    omega = []
    eta = []

    if (abs(coseta) > 1.):
        return omega, eta
    # else calc the two eta values
    eta = n.array([n.arccos(coseta), -n.arccos(coseta)])
    # replaced this by the above to make find_omega_wedge
    # and find_omega_quart give same eta
    # eta = n.array([n.arccos(coseta), 2*n.pi-n.arccos(coseta)])
    # print eta*180.0/n.pi

    # Now find the Omega value(s)
    # A slight change in the code from GrainSpotter: the lenght
    # Here the original a and b is divided by length since
    # these can be only scale somega and comega equally
    a = (coswedge * cosfactor + sinwedge * sintth * coseta)
    for i in range(2):
        b = -sintth * n.sin(eta[i])
        somega = (b * g_w[0] - a * g_w[1]) / (a * a + b * b)
        comega = (g_w[0] - b * somega) / a

        omega.append(n.arctan2(somega, comega))
        if omega[i] > n.pi:
            omega[i] = omega[i] - 2 * n.pi
    return n.array(omega), eta


def find_omega(g_w, twoth):
    """
    Calculate the omega angles for a g-vector gw given twotheta using Soeren
    Schmidts algorithm.
    Solves an equation of type a*cos(w)+b*sin(w) = c by the fixpoint method.

    """
    g_g = n.sqrt(n.dot(g_w, g_w))
    costth = n.cos(twoth)

    a = g_w[0] / g_g
    b = -g_w[1] / g_g
    c = (costth - 1) / n.sqrt(2 * (1 - costth))

    d = a**2 + b**2
    sq_d = d - c**2

    omega = []
    if sq_d > 0:
        sq_d = n.sqrt(sq_d)
        comega = (a * c + b * sq_d) / d
        somega = (b * c - a * sq_d) / d
        omega.append(n.arccos(comega))
#        if omega[0] > n.pi:
#            omega[0] = omega[0] - 2*n.pi
        if somega < 0:
            omega[0] = -omega[0]
        comega = comega - 2 * b * sq_d / d
        somega = somega + 2 * a * sq_d / d
        omega.append(n.arccos(comega))
#        if omega[1] > n.pi:
#            omega[1] = omega[1] - 2*n.pi
        if somega < 0:
            omega[1] = -omega[1]
    return n.array(omega)


def cell_invert(unit_cell):
    """
    cell_invert calculates the reciprocal unit cell parameters
    from the direct space unit cell

    INPUT: unit_cell = [a, b, c, alpha, beta, gamma]
    OUTPUT unit_cell_star = [astar, bstar, cstar, alpastar, betastar, gammastar]

    """
    a = unit_cell[0]
    b = unit_cell[1]
    c = unit_cell[2]
    calp = n.cos(unit_cell[3] * n.pi / 180.)
    cbet = n.cos(unit_cell[4] * n.pi / 180.)
    cgam = n.cos(unit_cell[5] * n.pi / 180.)
    salp = n.sin(unit_cell[3] * n.pi / 180.)
    sbet = n.sin(unit_cell[4] * n.pi / 180.)
    sgam = n.sin(unit_cell[5] * n.pi / 180.)
    V = cell_volume(unit_cell)

    astar = b * c * salp / V
    bstar = a * c * sbet / V
    cstar = a * b * sgam / V
    # salpstar = V/(a*b*c*sbet*sgam)
    # sbetstar = V/(a*b*c*salp*sgam)
    # sgamstar = V/(a*b*c*salp*sbet)
    calpstar = (cbet * cgam - calp) / (sbet * sgam)
    cbetstar = (calp * cgam - cbet) / (salp * sgam)
    cgamstar = (calp * cbet - cgam) / (salp * sbet)

    alpstar = n.arccos(calpstar) * 180. / n.pi
    betstar = n.arccos(cbetstar) * 180. / n.pi
    gamstar = n.arccos(cgamstar) * 180. / n.pi

    return [astar, bstar, cstar, alpstar, betstar, gamstar]


def form_omega_mat(omega):
    """
    Calc Omega rotation matrix having an omega angle of "omega"

    INPUT: omega (in radians)

    OUTPUT: Omega rotation matrix
    """
    Om = n.array([[n.cos(omega), -n.sin(omega), 0],
                  [n.sin(omega), n.cos(omega), 0],
                  [0, 0, 1]])
    return Om


def form_omega_mat_general(omega, chi, wedge):
    """
    Calc Omega rotation matrix having an omega angle of "omega"

    INPUT: omega,chi and wedge (in radians)

    OUTPUT: Omega rotation matrix
    """
    phi_x = n.array([[1, 0, 0],
                     [0, n.cos(chi), -n.sin(chi)],
                     [0, n.sin(chi), n.cos(chi)]])
    phi_y = n.array([[n.cos(wedge), 0, n.sin(wedge)],
                     [0, 1, 0],
                     [-n.sin(wedge), 0, n.cos(wedge)]])
    Om = form_omega_mat(omega)
    Om = n.dot(phi_x, n.dot(phi_y, Om))
    return Om


def cell_volume(unit_cell):
    """
    cell_volume calculates the volume of the unit cell in AA^3
    from the direct space unit cell parameters

    INPUT: unit_cell = [a, b, c, alpha, beta, gamma]
    OUTPUT: volume


    """
    a = unit_cell[0]
    b = unit_cell[1]
    c = unit_cell[2]
    calp = n.cos(unit_cell[3] * n.pi / 180.)
    cbet = n.cos(unit_cell[4] * n.pi / 180.)
    cgam = n.cos(unit_cell[5] * n.pi / 180.)

    angular = n.sqrt(
        1 -
        calp *
        calp -
        cbet *
        cbet -
        cgam *
        cgam +
        2 *
        calp *
        cbet *
        cgam)
    # Volume of unit cell
    V = a * b * c * angular
    return V


def form_b_mat(unit_cell):
    """
    calculate B matrix of (Gcart = B Ghkl) following eq. 3.4 in
    H.F. Poulsen.
    Three-dimensional X-ray diffraction microscopy.
    Mapping polycrystals and their dynamics.
    Springer Tracts in Modern Physics, v. 205), (Springer, Berlin, 2004).

    INPUT:  unit_cell - unit_cell = [a, b, c, alpha, beta, gamma]
    OUTPUT: B - a 3x3 matrix

    Henning Osholm Sorensen, Risoe-DTU, June 11, 2007.
    """

    a = unit_cell[0]
    b = unit_cell[1]
    c = unit_cell[2]
    calp = n.cos(unit_cell[3] * n.pi / 180.)
    cbet = n.cos(unit_cell[4] * n.pi / 180.)
    cgam = n.cos(unit_cell[5] * n.pi / 180.)
    salp = n.sin(unit_cell[3] * n.pi / 180.)
    sbet = n.sin(unit_cell[4] * n.pi / 180.)
    sgam = n.sin(unit_cell[5] * n.pi / 180.)

    # Volume of unit cell
    V = cell_volume(unit_cell)

    #  Calculate reciprocal lattice parameters:
    # NOTICE PHYSICIST DEFINITION of recip axes with 2*pi
    astar = 2 * n.pi * b * c * salp / V
    bstar = 2 * n.pi * a * c * sbet / V
    cstar = 2 * n.pi * a * b * sgam / V
    # salpstar = V/(a*b*c*sbet*sgam)
    sbetstar = V / (a * b * c * salp * sgam)
    sgamstar = V / (a * b * c * salp * sbet)
    # calpstar = (cbet*cgam-calp)/(sbet*sgam)
    cbetstar = (calp * cgam - cbet) / (salp * sgam)
    cgamstar = (calp * cbet - cgam) / (salp * sbet)

    # Form B matrix following eq. 3.4 in H.F Poulsen
    B = n.array([[astar, bstar * cgamstar, cstar * cbetstar],
                 [0, bstar * sgamstar, -cstar * sbetstar * calp],
                 [0, 0, cstar * sbetstar * salp]])
    return B


def form_a_mat(unit_cell):
    """
    calculate the A matrix given in eq. 3.23 of H.F. Poulsen.
    Three-dimensional X-ray diffraction microscopy.
    Mapping polycrystals and their dynamics.
    (Springer Tracts in Modern Physics, v. 205), (Springer, Berlin, 2004).

    INPUT: unit_cell - unit_cell = [a, b, c, alpha, beta, gamma]

    OUTPUT A - a 3x3 matrix

    Jette Oddershede, March 7, 2008.
    """

    a = unit_cell[0]
    b = unit_cell[1]
    c = unit_cell[2]
    calp = n.cos(unit_cell[3] * n.pi / 180.)
    cbet = n.cos(unit_cell[4] * n.pi / 180.)
    cgam = n.cos(unit_cell[5] * n.pi / 180.)
    # salp = n.sin(unit_cell[3]*n.pi/180.)
    sbet = n.sin(unit_cell[4] * n.pi / 180.)
    sgam = n.sin(unit_cell[5] * n.pi / 180.)

    # Volume of unit cell
    V = cell_volume(unit_cell)

    #  Calculate reciprocal lattice parameters
    salpstar = V / (a * b * c * sbet * sgam)
    calpstar = (cbet * cgam - calp) / (sbet * sgam)

    # Form A matrix following eq. 3.23 in H.F Poulsen
    A = n.array([[a, b * cgam, c * cbet],
                 [0, b * sgam, -c * sbet * calpstar],
                 [0, 0, c * sbet * salpstar]])
    return A


def form_a_mat_inv(unit_cell):
    """
    calculate the inverse of the A matrix given in eq. 3.23 of H.F. Poulsen.
    Three-dimensional X-ray diffraction microscopy.
    Mapping polycrystals and their dynamics.
    (Springer Tracts in Modern Physics, v. 205), (Springer, Berlin, 2004).

    INPUT: unit_cell - unit_cell = [a, b, c, alpha, beta, gamma]

    OUTPUT A^-1 - a 3x3 matrix

    Jette Oddershede, March 7, 2008.
    """

    A = form_a_mat(unit_cell)
    Ainv = n.linalg.inv(A)
    return Ainv


def ubi_to_cell(ubi):
    """
    calculate lattice constants from the UBI-matrix as
    defined in H.F.Poulsen 2004 eqn.3.23

    ubi_to_cell(ubi)

    ubi [3x3] matrix of (U*B)^-1
    in this case B = B /2pi
    returns unit_cell = [a, b, c, alpha, beta, gamma]

    """
    return n.array(a_to_cell(n.transpose(ubi)))


def ubi_to_u(ubi):
    """
    calculate lattice constants from the UBI-matrix
    defined(U*B)^-1 and is B from form_b_mat devided by 2pi

    ubi_to_u(ubi)

    ubi [3x3] matrix

    returns U matrix

    """
    unit_cell = ubi_to_cell(ubi)
    B = form_b_mat(unit_cell)
    U = n.transpose(n.dot(B, ubi)) / (2 * n.pi)
    if checks.is_activated():
        checks._check_rotation_matrix(U)

    return U


def ubi_to_u_and_eps(ubi, unit_cell):
    """
    calculate lattice lattice rotation and strain from the UBI-matrix

    (U,eps) = ubi_to_u_and_eps(ubi,unit_cell)

    ubi [3x3] matrix, (UB)^-1, where B=B/2*pi
    unit_cell = [a,b,c,alpha,beta,gamma]

    returns U matrix and strain tensor components
    eps = [e11, e12, e13, e22, e23, e33]

    """
    unit_cell_ubi = ubi_to_cell(ubi)
    B_ubi = form_b_mat(unit_cell_ubi)
    U = n.transpose(n.dot(B_ubi, ubi)) / (2 * n.pi)

    if checks.is_activated():
        checks._check_rotation_matrix(U)

    eps = b_to_epsilon(B_ubi, unit_cell)

    return (U, eps)


def a_to_cell(A):
    """
    calculate lattice constants from the A-matix as
    defined in H.F.Poulsen 2004 eqn.3.23

    a_to_cell(A)

    A [3x3] upper triangular matrix
    returns unit_cell = [a, b, c, alpha, beta, gamma]

    Jette Oddershede, March 10, 2008.
    """

    g = n.dot(n.transpose(A), A)
    a = n.sqrt(g[0, 0])
    b = n.sqrt(g[1, 1])
    c = n.sqrt(g[2, 2])
    alpha = degrees(n.arccos(g[1, 2] / b / c))
    beta = degrees(n.arccos(g[0, 2] / a / c))
    gamma = degrees(n.arccos(g[0, 1] / a / b))
    unit_cell = [a, b, c, alpha, beta, gamma]
    return unit_cell


def b_to_cell(B):
    """
    calculate lattice constants from the B-matix as
    defined in H.F.Poulsen 2004 eqn.3.4

    B [3x3] upper triangular matrix
    returns unit_cell = [a, b, c, alpha, beta, gamma]

    Jette Oddershede, April 21, 2008.
    """

    B = B / (2 * n.pi)
    g = n.dot(n.transpose(B), B)
    astar = n.sqrt(g[0, 0])
    bstar = n.sqrt(g[1, 1])
    cstar = n.sqrt(g[2, 2])
    alphastar = degrees(n.arccos(g[1, 2] / bstar / cstar))
    betastar = degrees(n.arccos(g[0, 2] / astar / cstar))
    gammastar = degrees(n.arccos(g[0, 1] / astar / bstar))

    unit_cell = cell_invert([astar, bstar, cstar,
                             alphastar, betastar, gammastar])
    return unit_cell


def epsilon_to_b_old(epsilon, unit_cell):
    """
    calculate B matrix of (Gcart = B Ghkl) from epsilon and
    unstrained cell as in H.F. Poulsen (2004) page 33.

    INPUT: epsilon - strain tensor [e11, e12, e13, e22, e23, e33]
    unit_cell - unit cell = [a, b, c, alpha, beta, gamma]

    OUTPUT: B - [3x3] for strained lattice constants

    Jette Oddershede, March 10, 2008.
    """

    A0inv = form_a_mat_inv(unit_cell)
    A = n.zeros((3, 3))
    A[0, 0] = (epsilon[0] + 1) / A0inv[0, 0]
    A[1, 1] = (epsilon[3] + 1) / A0inv[1, 1]
    A[2, 2] = (epsilon[5] + 1) / A0inv[2, 2]
    A[0, 1] = (2 * epsilon[1] - A[0, 0] * A0inv[0, 1]) / A0inv[1, 1]
    A[1, 2] = (2 * epsilon[4] - A[1, 1] * A0inv[1, 2]) / A0inv[2, 2]
    A[0, 2] = (2 * epsilon[2] - A[0, 0] * A0inv[0, 2] -
               A[0, 1] * A0inv[1, 2]) / A0inv[2, 2]
    strainedcell = a_to_cell(A)
    B = form_b_mat(strainedcell)
    return B


def b_to_epsilon_old(B, unit_cell):
    """
    calculate epsilon from the the unstrained cell and
    the B matrix of (Gcart = B Ghkl) as in H.F. Poulsen (2004) page 33.

    INPUT: B - upper triangular 3x3 matrix of strained lattice constants
    unit_cell -  unit cell = [a, b, c, alpha, beta, gamma]
    of unstrained lattice

    OUTPUT: epsilon = [e11, e12, e13, e22, e23, e33]

    Jette Oddershede, April 21, 2008.
    """

    A0inv = form_a_mat_inv(unit_cell)
    A = form_a_mat(b_to_cell(B))
    T = n.dot(A, A0inv)
    I = n.eye(3)
    eps = 0.5 * (T + n.transpose(T)) - I
    epsilon = [eps[0, 0], eps[0, 1], eps[0, 2],
               eps[1, 1], eps[1, 2], eps[2, 2]]
    return epsilon


def b_to_epsilon(B, unit_cell):
    """
    calculate epsilon from the the unstrained cell and
    the B matrix of (Gcart = B Ghkl).
    The algorithm is similar to H.F. Poulsen (2004) page 33,
    except A is defined such that A'B = I
    T' = (A*A0inv)' = A0inv'*A' = B0*Binv
    This definition of A ensures that U relating Gcart to Gsam
    also relates epsilon_cart to epsilon_sam

    INPUT: B - upper triangular 3x3 matrix of strained lattice constants
    unit_cell -  unit cell = [a, b, c, alpha, beta, gamma]
    of unstrained lattice

    OUTPUT: epsilon = [e11, e12, e13, e22, e23, e33]

    Jette Oddershede, jeto@fysik.dtu.dk, January 2012.
    """

    B0 = form_b_mat(unit_cell)
    T = n.dot(B0, n.linalg.inv(B))
    I = n.eye(3)
    eps = 0.5 * (T + n.transpose(T)) - I
    epsilon = [eps[0, 0], eps[0, 1], eps[0, 2],
               eps[1, 1], eps[1, 2], eps[2, 2]]
    return epsilon


def epsilon_to_b(epsilon, unit_cell):
    """
    calculate B matrix of (Gcart = B Ghkl) from epsilon and
    unstrained cell as in H.F. Poulsen (2004) page 33 with the
    exception that A is defined such that A'B = I
    2*(epsilon+I) = B0*Binv + (B0*Binv)'

    INPUT: epsilon - strain tensor [e11, e12, e13, e22, e23, e33]
    unit_cell - unit cell = [a, b, c, alpha, beta, gamma]
    of unstrained lattice

    OUTPUT: B - [3x3] for strained lattice constants

    Jette Oddershede, jeto@fysik.dtu.dk, January 2012.
    """

    B0 = form_b_mat(unit_cell)
    Binv = n.zeros((3, 3))
    Binv[0, 0] = (epsilon[0] + 1) / B0[0, 0]
    Binv[1, 1] = (epsilon[3] + 1) / B0[1, 1]
    Binv[2, 2] = (epsilon[5] + 1) / B0[2, 2]
    Binv[0, 1] = (2 * epsilon[1] - B0[0, 1] * Binv[1, 1]) / B0[0, 0]
    Binv[1, 2] = (2 * epsilon[4] - B0[1, 2] * Binv[2, 2]) / B0[1, 1]
    Binv[0, 2] = (2 * epsilon[2] - B0[0, 1] * Binv[1, 2] -
                  B0[0, 2] * Binv[2, 2]) / B0[0, 0]
    return n.linalg.inv(Binv)


def euler_to_u(phi1, PHI, phi2):
    """
    U matrix from Euler angles phi1, PHI, phi2.
    The formalism follows the Idet_corner_11-3DXRD specs

    U = euler_to_u(phi1, PHI, phi2)

    INPUT: phi1, PHI, and phi2 angles in radians
    OUTPUT: U = array([[U11, U12, U13],[ U21, U22, U23],[U31, U32, U33]])

    Changed input angles to be in radians instead of degrees
    Henning Osholm Sorensen, Riso National Laboratory, June 23, 2006.

    Translated from MATLAB to python by Jette Oddershede, March 26 2008
    Origingal MATLAB code from: Henning Poulsen, Risoe 15/6 2002.

    """
    if checks.is_activated():
        checks._check_euler_angles(phi1, PHI, phi2)

    U = n.zeros((3, 3))
    U[0, 0] = n.cos(phi1) * n.cos(phi2) - n.sin(phi1) * \
        n.sin(phi2) * n.cos(PHI)
    U[1, 0] = n.sin(phi1) * n.cos(phi2) + n.cos(phi1) * \
        n.sin(phi2) * n.cos(PHI)
    U[2, 0] = n.sin(phi2) * n.sin(PHI)
    U[0, 1] = -n.cos(phi1) * n.sin(phi2) - n.sin(phi1) * \
        n.cos(phi2) * n.cos(PHI)
    U[1, 1] = -n.sin(phi1) * n.sin(phi2) + n.cos(phi1) * \
        n.cos(phi2) * n.cos(PHI)
    U[2, 1] = n.cos(phi2) * n.sin(PHI)
    U[0, 2] = n.sin(phi1) * n.sin(PHI)
    U[1, 2] = -n.cos(phi1) * n.sin(PHI)
    U[2, 2] = n.cos(PHI)
    return U


def _arctan2(y, x):
    """Modified arctan function used locally in u_to_euler().
    """
    tol = 1e-8
    if n.abs(x) < tol:
        x = 0
    if n.abs(y) < tol:
        y = 0

    if x > 0:
        return n.arctan(y / x)
    elif x < 0 and y >= 0:
        return n.arctan(y / x) + n.pi
    elif x < 0 and y < 0:
        return n.arctan(y / x) - n.pi
    elif x == 0 and y > 0:
        return n.pi / 2
    elif x == 0 and y < 0:
        return -n.pi / 2
    elif x == 0 and y == 0:
        raise ValueError(
            'Local function _arctan2() does not accept arguments (0,0)')


def u_to_euler(U):
    """Convert unitary 3x3 rotation matrix into Euler angles in Bunge notation.
    The returned Euler angles are all in the range [0, 2*pi]. If Gimbal lock occurs
    (PHI=0) infinite number of solutions exists. The solution returned is phi2=0.

    Implementation is based on the notes by
        Depriester, Dorian. (2018). Computing Euler angles with Bunge convention from rotation matrix.
        https://www.researchgate.net/publication/324088567_Computing_Euler_angles_with_Bunge_convention_from_rotation_matrix
    notationwise U = g.T in in these notes.

        INPUT:
            U : unitary 3x3 rotation matrix as a numpy array (shape=(3,3))
        RETURNS
            angles : Euler angles in radians. numpy array as, [phi1, PHI, phi2].


        Last Modified: Axel Henningsson, January 2021
    """
    if checks.is_activated():
        checks._check_rotation_matrix(U)

    tol = 1e-8
    PHI = n.arccos(U[2, 2])
    if n.abs(PHI) < tol:
        phi1 = _arctan2(-U[0, 1], U[0, 0])
        phi2 = 0
    elif n.abs(PHI - n.pi) < tol:
        phi1 = _arctan2(U[0, 1], U[0, 0])
        phi2 = 0
    else:
        phi1 = _arctan2(U[0, 2], -U[1, 2])
        phi2 = _arctan2(U[2, 0], U[2, 1])

    if phi1 < 0:
        phi1 = phi1 + 2 * n.pi
    if phi2 < 0:
        phi2 = phi2 + 2 * n.pi

    return n.array([phi1, PHI, phi2])


def u_to_rod(U):
    """
    Get Rodrigues vector from U matrix (Busing Levy)
    INPUT: U 3x3 matrix
    OUTPUT: Rodrigues vector

    Function taken from GrainsSpotter by Soeren Schmidt
    """
    if checks.is_activated():
        checks._check_rotation_matrix(U)

    ttt = 1 + U[0, 0] + U[1, 1] + U[2, 2]
    if abs(ttt) < 1e-16:
        raise ValueError('Wrong trace of U')
    a = 1 / ttt
    r1 = (U[1, 2] - U[2, 1]) * a
    r2 = (U[2, 0] - U[0, 2]) * a
    r3 = (U[0, 1] - U[1, 0]) * a
    return n.array([r1, r2, r3])


def u_to_ubi(u_mat, unit_cell):
    """
    Get UBI matrix from U matrix and unit cell
    INPUT: U orientaion matrix and unit cell
    OUTPUT: UBI 3x3 matrix

    """
    if checks.is_activated():
        checks._check_rotation_matrix(u_mat)

    b_mat = form_b_mat(unit_cell)

    return n.linalg.inv(n.dot(u_mat, b_mat)) * (2 * n.pi)


def ubi_to_rod(ubi):
    """
    Get Rodrigues vector from UBI matrix
    INPUT: UBI 3x3 matrix
    OUTPUT: Rodrigues vector

    """

    return u_to_rod(ubi_to_u(ubi))


def ubi_to_u_b(ubi):
    """
    Get Rodrigues vector from UBI matrix
    INPUT: UBI 3x3 matrix
    OUTPUT: U orientaion matrix and B metric matrix

    """
    return ub_to_u_b(n.linalg.inv(ubi) * (2 * n.pi))


def rod_to_u(r):
    """
    rod_to_u calculates the U orientation matrix given an oriention
    represented in Rodrigues space. r = [r1, r2, r3]
    """
    g = n.zeros((3, 3))
    r2 = n.dot(r, r)

    for i in range(3):
        for j in range(3):
            if i == j:
                fac = 1
            else:
                fac = 0
            term = 0
            for k in range(3):
                if [i, j, k] == [0, 1, 2] or \
                    [i, j, k] == [1, 2, 0] or \
                        [i, j, k] == [2, 0, 1]:
                    sign = 1
                elif [i, j, k] == [2, 1, 0] or \
                    [i, j, k] == [0, 2, 1] or \
                        [i, j, k] == [1, 0, 2]:
                    sign = -1
                else:
                    sign = 0
                term = term + 2 * sign * r[k]
            g[i, j] = 1 / (1 + r2) * ((1 - r2) * fac + 2 * r[i] * r[j] - term)
    return n.transpose(g)


def ub_to_u_b(UB):
    """
    qr decomposition to get U unitary and B upper triangular with positive
    diagonal from UB
    """

    (U, B) = n.linalg.qr(UB)
    if B[0, 0] < 0:
        B[0, 0] = -B[0, 0]
        B[0, 1] = -B[0, 1]
        B[0, 2] = -B[0, 2]
        U[0, 0] = -U[0, 0]
        U[1, 0] = -U[1, 0]
        U[2, 0] = -U[2, 0]
    if B[1, 1] < 0:
        B[1, 1] = -B[1, 1]
        B[1, 2] = -B[1, 2]
        U[0, 1] = -U[0, 1]
        U[1, 1] = -U[1, 1]
        U[2, 1] = -U[2, 1]
    if B[2, 2] < 0:
        B[2, 2] = -B[2, 2]
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]

    if checks.is_activated():
        checks._check_rotation_matrix(U)

    return (U, B)


def reduce_cell(unit_cell, uvw=3):
    """
    reduce unit cell

    INPUT: unit_cell - unit_cell = [a, b, c, alpha, beta, gamma]
    OUTPUT unit_cell reduced -  array([a, b, c, alpha, beta, gamma])
    """

    res = n.zeros((0, 4))
    red_a_mat = n.zeros((3, 3))

    a_mat = form_a_mat(unit_cell)

    for i in n.arange(-uvw, uvw):
        for j in n.arange(-uvw, uvw):
            for k in n.arange(-uvw, uvw):
                tmp = n.dot(a_mat, n.array([i, j, k]))
                res = n.concatenate((res, [[i, j, k, n.linalg.norm(tmp)]]))

    res = res[n.argsort(res[:, 3]), :]

    red_a_mat[0] = n.dot(a_mat, res[1, :3])

    for i in range(2, len(res)):
        tmp = n.dot(a_mat, res[i, :3])
        kryds = n.cross(tmp, red_a_mat[0])
        if n.sum(n.abs(kryds)) > 0.00001:
            red_a_mat[1] = tmp
            break

    for j in range(i, len(res)):
        tmp = n.dot(a_mat, res[j, :3])
        dist = n.dot(kryds, tmp) / n.linalg.norm(kryds)
        if dist > 0.00001:
            red_a_mat[2] = tmp
            break

    return a_to_cell(red_a_mat)


def detect_tilt(tilt_x, tilt_y, tilt_z):
    """
    Calculate the tilt matrix
    tiltR(tilt_x,tilt_y,tilt_z)

    input tilt_x, tilt_y, tilt_z

    Henning Osholm Sorensen 2006
    """
    Rx = n.array([[1, 0, 0],
                  [0, n.cos(tilt_x), -n.sin(tilt_x)],
                  [0, n.sin(tilt_x), n.cos(tilt_x)]])
    Ry = n.array([[n.cos(tilt_y), 0, n.sin(tilt_y)],
                  [0, 1, 0],
                  [-n.sin(tilt_y), 0, n.cos(tilt_y)]])
    Rz = n.array([[n.cos(tilt_z), -n.sin(tilt_z), 0],
                  [n.sin(tilt_z), n.cos(tilt_z), 0],
                  [0, 0, 1]])
    R = n.dot(Rx, n.dot(Ry, Rz))
    return R


def quart_to_omega(w, w_x, w_y):
    """
     Calculate the Omega rotation matrix given w (the motorised rotation in
     degrees, usually around the z-axis).
     wx and wy (the rotations around x and y bringing the z-axis to the true
     rotation axis, in radians).
     Quarternions are used for the calculations to avoid singularities in
     subsequent refinements.
    """
    whalf = w * n.pi / 360.
    w_mat_x = n.array([[1, 0, 0],
                       [0, n.cos(w_x), -n.sin(w_x)],
                       [0, n.sin(w_x), n.cos(w_x)]])
    w_mat_y = n.array([[n.cos(w_y), 0, n.sin(w_y)],
                       [0, 1, 0],
                       [-n.sin(w_y), 0, n.cos(w_y)]])
    qua = n.dot(w_mat_x, n.dot(w_mat_y, n.array([0, 0, n.sin(whalf)])))
    q = [n.cos(whalf), qua[0], qua[1], qua[2]]
    omega_mat = n.array([[1 - 2 * q[2]**2 - 2 * q[3]**2,
                          2 * q[1] * q[2] - 2 * q[3] * q[0],
                          2 * q[1] * q[3] + 2 * q[2] * q[0]],
                         [2 * q[1] * q[2] + 2 * q[3] * q[0],
                          1 - 2 * q[1]**2 - 2 * q[3]**2,
                          2 * q[2] * q[3] - 2 * q[1] * q[0]],
                         [2 * q[1] * q[3] - 2 * q[2] * q[0],
                          2 * q[2] * q[3] + 2 * q[1] * q[0],
                          1 - 2 * q[1]**2 - 2 * q[2]**2]])
    return omega_mat


def sintl(unit_cell, hkl):
    """
    sintl calculate sin(theta)/lambda of the reflection "hkl" given
    the unit cell "unit_cell"

    sintl(unit_cell,hkl)

    INPUT:  unit_cell = [a, b, c, alpha, beta, gamma]
            hkl = [h, k, l]
    OUTPUT: sin(theta)/lambda

    Henning Osholm Sorensen, Risoe National Laboratory, June 23, 2006.
    """
    a = float(unit_cell[0])
    b = float(unit_cell[1])
    c = float(unit_cell[2])
    calp = n.cos(unit_cell[3] * n.pi / 180.)
    cbet = n.cos(unit_cell[4] * n.pi / 180.)
    cgam = n.cos(unit_cell[5] * n.pi / 180.)

    (h, k, l) = hkl

    part1 = (h * h / a**2) * (1 - calp**2) + (k * k / b**2) *\
            (1 - cbet**2) + (l * l / c**2) * (1 - cgam**2) +\
        2 * h * k * (calp * cbet - cgam) / (a * b) + 2 * h * l * (calp * cgam - cbet) / (a * c) +\
        2 * k * l * (cbet * cgam - calp) / (b * c)

    part2 = 1 - (calp**2 + cbet**2 + cgam**2) + 2 * calp * cbet * cgam

    stl = n.sqrt(part1) / (2 * n.sqrt(part2))

    return stl


def tth(unit_cell, hkl, wavelength):
    """

    tth calculate two theta of the reflection given
    the unit cell and wavelenght

    INPUT:  unit_cell = [a, b, c, alpha, beta, gamma] (in Angstroem and degrees)
            hkl = [h, k, l]
            wavelenth (in Angstroem)

    OUTPUT: twotheta (in radians)

    Henning Osholm Sorensen, Risoe-DTU, July 16, 2008.
    """

    stl = sintl(unit_cell, hkl)  # calls sintl function in tools
    twotheta = 2 * n.arcsin(wavelength * stl)

    return twotheta


def tth2(gve, wavelength):
    """

    calculates two theta for a scattering vector given the wavelenght

    INPUT:  gve: scattering vector
                 (defined in reciprocal space (2*pi/lambda))
            wavelenth (in Angstroem)

    OUTPUT: twotheta (in radians)

    Henning Osholm Sorensen, Risoe DTU, July 17, 2008.
    """
    length = n.sqrt(n.dot(gve, gve))
    twotheta = 2.0 * n.arcsin(length * wavelength / (4 * n.pi))

    return twotheta


def genhkl_all(
        unit_cell,
        sintlmin,
        sintlmax,
        sgname=None,
        sgno=None,
        cell_choice='standard',
        output_stl=False,
        verbose=True):
    """

    Generate the full set of reflections given a unit cell and space group up to maximum sin(theta)/lambda (sintlmax)

        The function is using the function genhkl_base for the actual generation

    INPUT:  unit cell     : [a , b, c, alpha, beta, gamma]
            sintlmin      : minimum sin(theta)/lambda for generated reflections
            sintlmax      : maximum sin(theta)/lambda for generated reflections
            sgno/sgname   : provide either the space group number or its name
                            e.g. sgno=225 or equivalently
                                 sgname='Fm-3m'
            output_stl    : Should sin(theta)/lambda be output (True/False)
                            default=False

    OUTPUT: list of reflections  (n by 3) or (n by 4)
            if sin(theta)/lambda is chosen to be output

    The algorithm follows the method described in:
    Le Page and Gabe (1979) J. Appl. Cryst., 12, 464-466

    Henning Osholm Sorensen, University of Copenhagen, July 22, 2010.
    """
    from xrd_simulator.xfab import sg
    if sgname is not None:
        spg = sg.sg(sgname=sgname, cell_choice=cell_choice)
    elif sgno is not None:
        spg = sg.sg(sgno=sgno, cell_choice=cell_choice)
    else:
        raise ValueError('No space group information given')

    H = genhkl_base(unit_cell,
                    spg.syscond,
                    sintlmin, sintlmax,
                    crystal_system=spg.crystal_system,
                    Laue_class=spg.Laue,
                    cell_choice=spg.cell_choice,
                    output_stl=True)

    Hall = n.zeros((0, 4))
    # Making sure that the inversion element also for non-centrosymmetric
    # space groups
    Rots = n.concatenate((spg.rot[:spg.nuniq], -spg.rot[:spg.nuniq]))
    (dummy, rows) = n.unique((Rots * n.random.rand(3, 3)
                              ).sum(axis=2).sum(axis=1), return_index=True)
    Rots = Rots[n.sort(rows)]

    for refl in H[:]:
        hkls = []
        stl = refl[3]
        for R in Rots:
            hkls.append(n.dot(refl[:3], R))
        a = n.array(hkls)
        (dummy, rows) = n.unique((a * n.random.rand(3)).sum(axis=1),
                                 return_index=True)
        Hsub = n.concatenate((a[rows],
                             n.array([[stl] * len(rows)]).transpose()),
                             axis=1)
        Hall = n.concatenate((Hall, Hsub))

    if not output_stl:
        return Hall[:, :3]
    else:
        return Hall


def genhkl_unique(
        unit_cell,
        sintlmin,
        sintlmax,
        sgname=None,
        sgno=None,
        cell_choice='standard',
        output_stl=False):
    """

    Generate the only unique set of reflections given a unit cell and space group up to maximum sin(theta)/lambda (sintlmax)

        The function is using the function genhkl_base for the actual generation

    INPUT:  unit cell     : [a , b, c, alpha, beta, gamma]
            sintlmin      : minimum sin(theta)/lambda for generated reflections
            sintlmax      : maximum sin(theta)/lambda for generated reflections
            sgno/sgname   : provide either the space group number or its name
                            e.g. sgno=225 or equivalently
                                 sgname='Fm-3m'
            output_stl    : Should sin(theta)/lambda be output (True/False)
                            default=False

    OUTPUT: list of reflections  (n by 3) or
                                 (n by 4) if sin(theta)/lambda is chosen to be output

    The algorithm follows the method described in:
    Le Page and Gabe (1979) J. Appl. Cryst., 12, 464-466

    Henning Osholm Sorensen, University of Copenhagen, July 22, 2010.
    """

    from xrd_simulator.xfab import sg
    if sgname is not None:
        spg = sg.sg(sgname=sgname, cell_choice=cell_choice)
    elif sgno is not None:
        spg = sg.sg(sgno=sgno, cell_choice=cell_choice)
    else:
        raise ValueError('No space group information given')

    H = genhkl_base(unit_cell,
                    spg.syscond,
                    sintlmin, sintlmax,
                    crystal_system=spg.crystal_system,
                    Laue_class=spg.Laue,
                    cell_choice=spg.cell_choice,
                    output_stl=True)

    if not output_stl:
        return H[:, :3]
    else:
        return H


def genhkl_base(
        unit_cell,
        sysconditions,
        sintlmin,
        sintlmax,
        crystal_system='triclinic',
        Laue_class='-1',
        cell_choice='standard',
        output_stl=None,
        verbose=False):
    """

    Generate the unique set of reflections for the cell up to maximum sin(theta)/lambda (sintlmax)

    The algorithm follows the method described in:
    Le Page and Gabe (1979) J. Appl. Cryst., 12, 464-466

    INPUT:  unit cell     : [a , b, c, alpha, beta, gamma]
            sysconditions : conditions for systematic absent reflections
                            a 26 element list e.g. [0,0,2,0,0,0,0,0,.....,3]
                            see help(sysabs) function for details.
            sintlmin      : minimum sin(theta)/lambda for generated reflections
            sintlmax      : maximum sin(theta)/lambda for generated reflections
            crystal_system: Crystal system (string), e.g. 'hexagonal'
            Laue class    : Laue class of the lattice (-1, 2/m, mmm, .... etc)
            cell_choice   : If more than cell choice can be made
                            e.g. R-3 can be either rhombohedral or hexagonal
            output_stl    : Should sin(theta)/lambda be output (True/False) default=False

    OUTPUT: list of reflections  (n by 3) or
                                 (n by 4) if sin(theta)/lambda is chosen to be output

    Henning Osholm Sorensen, University of Copenhagen, July 22, 2010.
    """
    segm = None

    # Triclinic : Laue group -1
    if Laue_class == '-1':
        if verbose:
            print('Laue class : -1', unit_cell)
        segm = n.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        [[-1, 0, 1], [-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        [[-1, 1, 0], [-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                        [[0, 1, -1], [1, 0, 0], [0, 1, 0], [0, 0, -1]]])

    # Monoclinic : Laue group 2/M
    # unique a
    # segm = n.array([[[ 0, 0,  0], [ 0, 1, 0], [ 1, 0, 0], [ 0, 0,  1]],
    #                [[ 0,-1,  1], [ 0,-1, 0], [ 1, 0, 0], [ 0, 0,  1]]])

    # Monoclinic : Laue group 2/M
    # unique b
    if Laue_class == '2/m':
        if verbose:
            print('Laue class : 2/m', unit_cell)
        segm = n.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        [[-1, 0, 1], [-1, 0, 0], [0, 1, 0], [0, 0, 1]]])

    # unique c
    # segm = n.array([[[ 0, 0,  0], [ 1, 0, 0], [ 0, 0, 1], [ 0, 1,  0]],
    #                [[-1, 1,  0], [-1, 0, 0], [ 0, 0, 1], [ 0, 1,  0]]])

    # Orthorhombic : Laue group MMM
    if Laue_class == 'mmm':
        segm = n.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]])

    # Tetragonal
    # Laue group : 4/MMM
    if Laue_class == '4/mmm':
        segm = n.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]]])

    # Laue group : 4/M
    if Laue_class == '4/m':
        segm = n.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]],
                        [[1, 2, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]]])

    # Hexagonal
    # Laue group : 6/MMM
    if Laue_class == '6/mmm':
        segm = n.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]]])

    # Laue group : 6/M
    if Laue_class == '6/m':
        segm = n.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]],
                        [[1, 2, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]]])

    # Laue group : -3M1
    if Laue_class == '-3m1':
        if verbose:
            print('Laue class : -3m1 (hex)', unit_cell)
        if unit_cell[4] == unit_cell[5]:
            if verbose:
                print('#############################################################')
                print('# Are you using a rhombohedral cell in a hexagonal setting? #')
                print('#############################################################')
        segm = n.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]],
                        [[0, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]]])

    # Laue group : -31M
    if Laue_class == '-31m':
        if verbose:
            print('Laue class : -31m (hex)', unit_cell)
        if unit_cell[4] == unit_cell[5]:
            if verbose:
                print('#############################################################')
                print('# Are you using a rhombohedral cell in a hexagonal setting? #')
                print('#############################################################')
        segm = n.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]],
                        [[1, 1, -1], [1, 0, 0], [1, 1, 0], [0, 0, -1]]])

    # Laue group : -3
    if Laue_class == '-3' and cell_choice != 'rhombohedral':
        if verbose:
            print('Laue class : -3 (hex)', unit_cell)
        if unit_cell[4] == unit_cell[5]:
            if verbose:
                print('#############################################################')
                print('# Are you using a rhombohedral cell in a hexagonal setting? #')
                print('#############################################################')
        segm = n.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]],
                        [[1, 2, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]],
                        [[0, 1, 1], [0, 1, 0], [-1, 1, 0], [0, 0, 1]]])

    # RHOMBOHEDRAL
    # Laue group : -3M
    if Laue_class == '-3m' and cell_choice == 'rhombohedral':
        if verbose:
            print('Laue class : -3m (Rhom)', unit_cell)
        if unit_cell[4] != unit_cell[5]:
            if verbose:
                print('#############################################################')
                print('# Are you using a hexagonal cell in a rhombohedral setting? #')
                print('#############################################################')
        segm = n.array([[[0, 0, 0], [1, 0, 0], [1, 0, -1], [1, 1, 1]],
                        [[1, 1, 0], [1, 0, -1], [0, 0, -1], [1, 1, 1]]])

    # Laue group : -3
    if Laue_class == '-3' and cell_choice == 'rhombohedral':
        if verbose:
            print('Laue class : -3 (Rhom)', unit_cell)
        if unit_cell[4] != unit_cell[5]:
            if verbose:
                print('#############################################################')
                print('# Are you using a hexagonal cell in a rhombohedral setting? #')
                print('#############################################################')
        segm = n.array([[[0, 0, 0], [1, 0, 0], [1, 0, -1], [1, 1, 1]],
                        [[1, 1, 0], [1, 0, -1], [0, 0, -1], [1, 1, 1]],
                        [[0, -1, -2], [1, 0, 0], [1, 0, -1], [-1, -1, -1]],
                        [[1, 0, -2], [1, 0, -1], [0, 0, -1], [-1, -1, -1]]])

    # Cubic
    # Laue group : M3M
    if Laue_class == 'm-3m':
        segm = n.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]])

    # Laue group : M3
    if Laue_class == 'm-3':
        segm = n.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
                        [[1, 2, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]]])

    if segm is None:
        if verbose:
            print('No Laue class found')
        return False

    nref = 0
    H = n.zeros((0, 3))
    stl = n.array([])
    sintlH = 0.0

    ##########################################################################
    # The factor of 1.1 in the sintlmax criteria for setting htest=1, ktest=1 and ltest=1
    # was added based on the following observation for triclinic cells (Laue -3) with
    # rhombohedral setting:
    # [0,-5,-6]=[0,-1,-2]+4*[1,0,0]+0*[1,0,-1]+4*[-1,-1,-1],
    # but the algorithm is such that [-4,-5,-6]=[0,-1,-2]+4*[-1,-1,-1] is tested first,
    # and this has a slightly larger sintl than [0,-5,-6]
    # If these values are on opposide sides of sintlmax [0,-5,-6] is never generated.
    # This is a quick and dirty fix, something more elegant would be good!
    # Jette Oddershede, February 2013
    ##########################################################################
    sintl_scale = 1
    if Laue_class == '-3' and cell_choice == 'rhombohedral':
        sintl_scale = 1.1

    for i in range(len(segm)):
        segn = i
        # initialize the identifiers
        htest = 0
        ktest = 0
        ltest = 0
        HLAST = segm[segn, 0, :]
        HSAVE = segm[segn, 0, :]
        HSAVE1 = segm[segn, 0, :]  # HSAVE1 =HSAVE
        sintlH = sintl(unit_cell, HSAVE)
        while ltest == 0:
            while ktest == 0:
                while htest == 0:
                    nref = nref + 1
                    if nref != 1:
                        ressss = sysabs(
                            HLAST, sysconditions, crystal_system, cell_choice)
                        if sysabs(
                                HLAST,
                                sysconditions,
                                crystal_system,
                                cell_choice) == 0:
                            if sintlH > sintlmin and sintlH <= sintlmax:
                                H = n.concatenate((H, [HLAST]))
                                stl = n.concatenate((stl, [sintlH]))
                        else:
                            nref = nref - 1
                    HNEW = HLAST + segm[segn, 1, :]
                    sintlH = sintl(unit_cell, HNEW)
                    # if (sintlH >= sintlmin) and (sintlH <= sintlmax):
                    if sintlH <= sintlmax * sintl_scale:
                        HLAST = HNEW
                    else:
                        htest = 1
#                        print HNEW,'htest',sintlH

                HSAVE = HSAVE + segm[segn, 2, :]
                HLAST = HSAVE
                HNEW = HLAST
                sintlH = sintl(unit_cell, HNEW)
                if sintlH > sintlmax * sintl_scale:
                    ktest = 1
                htest = 0

            HSAVE1 = HSAVE1 + segm[segn, 3, :]
            HSAVE = HSAVE1
            HLAST = HSAVE1
            HNEW = HLAST
            sintlH = sintl(unit_cell, HNEW)
            if sintlH > sintlmax * sintl_scale:
                ltest = 1
            ktest = 0

    stl = n.transpose([stl])
    H = n.concatenate((H, stl), 1)  # combine hkl and sintl
    H = H[n.argsort(H, 0)[:, 3], :]  # sort hkl's according to stl
    if output_stl is None:
        H = H[:, :3]
    return H


def genhkl(
        unit_cell,
        sysconditions,
        sintlmin,
        sintlmax,
        crystal_system='triclinic',
        output_stl=None):
    """

        OUTDATED SHOULD NOT BE USED ANYMORE - USE genhkl_all, genhkl_unique or genhkl_base. (July 22, 2010)

    Henning Osholm Sorensen, June 23, 2006.
    """
    segm = n.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[-1, 0, 1], [-1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    [[-1, 1, 0], [-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                    [[0, 1, -1], [1, 0, 0], [0, 1, 0], [0, 0, -1]]])

    nref = 0
    H = n.zeros((0, 3))
    stl = n.array([])
    sintlH = 0.0

    for i in range(len(segm)):
        segn = i
        # initialize the identifiers
        htest = 0
        ktest = 0
        ltest = 0
        HLAST = segm[segn, 0, :]
        HSAVE = HLAST
        sintlH = sintl(unit_cell, HSAVE)

        while ltest == 0:
            while ktest == 0:
                while htest == 0:
                    nref = nref + 1
                    if nref != 1:
                        ressss = sysabs(HLAST, sysconditions, crystal_system)
                        if sysabs(HLAST, sysconditions, crystal_system) == 0:
                            if sintlH > sintlmin and sintlH <= sintlmax:
                                H = n.concatenate((H, [HLAST]))
                                H = n.concatenate((H, [-HLAST]))
                                stl = n.concatenate((stl, [sintlH]))
                                stl = n.concatenate((stl, [sintlH]))
                        else:
                            nref = nref - 1
                    HNEW = HLAST + segm[segn, 1, :]
                    sintlH = sintl(unit_cell, HNEW)
                    # if (sintlH >= sintlmin) and (sintlH <= sintlmax):
                    if sintlH <= sintlmax:
                        HLAST = HNEW
                    else:
                        htest = 1

                HLAST[0] = HSAVE[0]
                HLAST = HLAST + segm[segn, 2, :]
                HNEW = HLAST
                sintlH = sintl(unit_cell, HNEW)
                if sintlH > sintlmax:
                    ktest = 1
                htest = 0

            HLAST[1] = HSAVE[1]
            HLAST = HLAST + segm[segn, 3, :]
            HNEW = HLAST
            sintlH = sintl(unit_cell, HNEW)
            if sintlH > sintlmax:
                ltest = 1
            ktest = 0

    stl = n.transpose([stl])
    H = n.concatenate((H, stl), 1)  # combine hkl and sintl
    H = H[n.argsort(H, 0)[:, 3], :]  # sort hkl's according to stl
    if output_stl is None:
        H = H[:, :3]
    return H


def sysabs(hkl, syscond, crystal_system='triclinic', cell_choice='standard'):
    """
    Defined as sysabs_unique with the exception that permutations in
    trigonal and hexagonal lattices are taken into account.

        INPUT: hkl     : [h k l]
           syscond : [1x26] with condition for systematic absences in this
                     space group, X in syscond should given as shown below
                   crystal_system : crystal system (string) - e.g. triclinic or hexagonal

    OUTPUT: sysbs  : if 1 the reflection is systematic absent
                     if 0 its not

    syscond:
    class        systematic abs               sysconditions[i]
    HKL          H+K=XN                            0
                 H+L=XN                            1
                 K+L=XN                            2
                 H+K,H+L,K+L = XN                  3
                 H+K+L=XN                          4
                 -H+K+L=XN                         5
    HHL          H=XN                              6
                 L=XN                              7
                 H+L=XN                            8
                 2H+L=XN                           9
    0KL          K=XN                             10
                 L=XN                             11
                 K+L=XN                           12
    H0L          H=XN                             13
                 L=XN                             14
                 H+L=XN                           15
    HK0          H=XN                             16
                 K=XN                             17
                 H+K=XN                           18
    HH0          H=XN                             19
    H00          H=XN                             20
    0K0          K=XN                             21
    00L          L=XN                             22
    H-HL         H=XN                             23
                 L=XN                             24
                 H+L=XN                           25


    """

    sys_type = sysabs_unique(hkl, syscond)
    if cell_choice == 'rhombohedral':
        if sys_type == 0:
            h = hkl[1]
            k = hkl[2]
            l = hkl[0]
            sys_type = sysabs_unique([h, k, l], syscond)
            if sys_type == 0:
                h = hkl[2]
                k = hkl[0]
                l = hkl[1]
                sys_type = sysabs_unique([h, k, l], syscond)
    elif crystal_system == 'trigonal' or crystal_system == 'hexagonal':
        if sys_type == 0:
            h = -(hkl[0] + hkl[1])
            k = hkl[0]
            l = hkl[2]
            sys_type = sysabs_unique([h, k, l], syscond)
            if sys_type == 0:
                h = hkl[1]
                k = -(hkl[0] + hkl[1])
                l = hkl[2]
                sys_type = sysabs_unique([h, k, l], syscond)

    return sys_type


def sysabs_unique(hkl, syscond):
    """
    sysabs_unique checks whether a reflection is systematic absent

    sysabs_unique = sysabs_unique(hkl,syscond)

    INPUT:  hkl     : [h k l]
            syscond : [1x26] with condition for systematic absences in this
                      space group, X in syscond should given as shown below
    OUTPUT: sysbs   :  if 1 the reflection is systematic absent
                       if 0 its not

    syscond:
    class        systematic abs               sysconditions[i]
    HKL          H+K=XN                            0
                 H+L=XN                            1
                 K+L=XN                            2
                 H+K,H+L,K+L = XN                  3
                 H+K+L=XN                          4
                 -H+K+L=XN                         5
    HHL          H=XN                              6
                 L=XN                              7
                 H+L=XN                            8
                 2H+L=XN                           9
    0KL          K=XN                             10
                 L=XN                             11
                 K+L=XN                           12
    H0L          H=XN                             13
                 L=XN                             14
                 H+L=XN                           15
    HK0          H=XN                             16
                 K=XN                             17
                 H+K=XN                           18
    HH0          H=XN                             19
    H00          H=XN                             20
    0K0          K=XN                             21
    00L          L=XN                             22
    H-HL         H=XN                             23
                 L=XN                             24
                 H+L=XN                           25

    Henning Osholm Sorensen, June 23, 2006.
    """

    (h, k, l) = hkl
    sysabs_type = 0

    # HKL class
    if syscond[0] != 0:
        condition = syscond[0]
        if (abs(h + k)) % condition != 0:
            sysabs_type = 1

    if syscond[1] != 0:
        condition = syscond[1]
        if (abs(h + l)) % condition != 0:
            sysabs_type = 2

    if syscond[2] != 0:
        condition = syscond[2]
        if (abs(k + l)) % condition != 0:
            sysabs_type = 3

    if syscond[3] != 0:
        sysabs_type = 4
        condition = syscond[3]
        if (abs(h + k)) % condition == 0:
            if (abs(h + l)) % condition == 0:
                if (abs(k + l)) % condition == 0:
                    sysabs_type = 0

    if syscond[4] != 0:
        condition = syscond[4]
        if (abs(h + k + l)) % condition != 0:
            sysabs_type = 5

    if syscond[5] != 0:
        condition = syscond[5]
        if (abs(-h + k + l)) % condition != 0:
            sysabs_type = 6

    # HHL class
    if (h - k) == 0:
        if syscond[6] != 0:
            condition = syscond[6]
            if (abs(h)) % condition != 0:
                sysabs_type = 7
        if syscond[7] != 0:
            condition = syscond[7]
            if (abs(l)) % condition != 0:
                sysabs_type = 8
        if syscond[8] != 0:
            condition = syscond[8]
            if (abs(h + l)) % condition != 0:
                sysabs_type = 9
        if syscond[9] != 0:
            condition = syscond[9]
            if (abs(h + h + l)) % condition != 0:
                sysabs_type = 10

    # 0KL class
    if h == 0:
        if syscond[10] != 0:
            condition = syscond[10]
            if (abs(k)) % condition != 0:
                sysabs_type = 11
        if syscond[11] != 0:
            condition = syscond[11]
            if (abs(l)) % condition != 0:
                sysabs_type = 12
        if syscond[12] != 0:
            condition = syscond[12]
            if (abs(k + l)) % condition != 0:
                sysabs_type = 13

    # H0L class
    if k == 0:
        if syscond[13] != 0:
            condition = syscond[13]
            if (abs(h)) % condition != 0:
                sysabs_type = 14
        if syscond[14] != 0:
            condition = syscond[14]
            if (abs(l)) % condition != 0:
                sysabs_type = 15
        if syscond[15] != 0:
            condition = syscond[15]
            if (abs(h + l)) % condition != 0:
                sysabs_type = 16

    # HK0 class
    if l == 0:
        if syscond[16] != 0:
            condition = syscond[16]
            if (abs(h)) % condition != 0:
                sysabs_type = 17
        if syscond[17] != 0:
            condition = syscond[17]
            if (abs(k)) % condition != 0:
                sysabs_type = 18
        if syscond[18] != 0:
            condition = syscond[18]
            if (abs(h + k)) % condition != 0:
                sysabs_type = 19

    # HH0 class
    if l == 0:
        if h - k == 0:
            if syscond[19] != 0:
                condition = syscond[19]
                if (abs(h)) % condition != 0:
                    sysabs_type = 20

    # H00 class
    if abs(k) + abs(l) == 0:
        if syscond[20] != 0:
            condition = syscond[20]
            if (abs(h)) % condition != 0:
                sysabs_type = 21

    # 0K0 class
    if abs(h) + abs(l) == 0:
        if syscond[21] != 0:
            condition = syscond[21]
            if (abs(k)) % condition != 0:
                sysabs_type = 22

    # 00L class
    if abs(h) + abs(k) == 0:
        if syscond[22] != 0:
            condition = syscond[22]
            if (abs(l)) % condition != 0:
                sysabs_type = 23

    # H-HL class
    if (h + k) == 0:
        if syscond[23] != 0:
            condition = syscond[23]
            if (abs(h)) % condition != 0:
                sysabs_type = 24
        if syscond[24] != 0:
            condition = syscond[24]
            if (abs(l)) % condition != 0:
                sysabs_type = 25
        if syscond[25] != 0:
            condition = syscond[25]
            if (abs(h + l)) % condition != 0:
                sysabs_type = 26


# NEW CONDITION FOR R-3c
#     # H-H(0)L class
#     if -h==k:
#         print '>>>>>>>>>>>>>>>>> DO I EVER GET HERE <<<<<<<<<<<<<<<<<<<<'
#         if syscond[23] != 0:
#             condition = syscond[23]

#             if (h+l)%condition != 0 or l%2 !=0:
#                 sysabs_type = 24

    return sysabs_type
