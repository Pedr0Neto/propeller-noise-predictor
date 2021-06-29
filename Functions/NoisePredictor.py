from xfoil import XFoil  # type: ignore
from xfoil.model import Airfoil  # type: ignore

import matplotlib.pyplot as plt  # type: ignore

from scipy import interpolate  # type: ignore
from scipy import optimize
from scipy import integrate

import numpy as np
import math
import random
import sys

import timeit


from scipy import fft



class Propeller:
    def __init__(self, propellername, B):
        ##
        # Parameters:
        #   propellername : string
        #       Name of the file and directory containing propeller data
        #   B : float
        #       Number of propeller blades
        ##

        #Number of blades
        self.B = B

        # Open the file to load data
        with open(propellername, "r") as file:
            # Read Airfoil name
            file.readline()

            # Read Airfoil data
            Table = []
            for line in file:
                Table.append([x for x in line.split("\n")[0].split("\t")])

        # Initialize arrays for better efficiency
        self.r = np.zeros(np.shape(Table)[0])
        self.c = np.zeros(np.shape(Table)[0])
        self.twist = np.zeros(np.shape(Table)[0])
        self.Foil = []

        # Organize propeller parameters
        for i in range(np.shape(Table)[0]):
            self.r[i] = Table[i][0]
            self.c[i] = Table[i][1]
            self.twist[i] = Table[i][2]
            self.Foil.append(Table[i][3])

    def import_airfoil(filename):
        ##
        # Description:
        #   Imports an airfoil from a .dat file into an array of two columns
        ##
        # Parameters:
        #   filename : string
        #       .dat file where the airfoild should be imported from
        ##
        # Returns:
        #   Table : array
        #       n x 2 array with airfoil coordinates. 1st column for x
        #           coordinates, 2nd column for y coordinates
        ##

        # Read data from file
        with open(filename, "r") as file:
            # Read Airfoil name
            file.readline()

            # Read Airfoil data
            Table = []
            for line in file:
                Table.append([float(x) for x in line.split("\n")[0].split(" ")[:] if x != ""])

            Table = np.array(Table)

        return (Table[:,0], Table[:,1])

    def interp_airfoil(self, n, arf):
        ##
        # Description:
        #   Interpolates the airfoils at each blade element
        ##
        # Parameters:
        #   n : float
        #       Number of blade elements
        #   arf : array
        #       a x 2 x b array of all imported airfoils
        #           a : number of blade sections defined by the user
        #           b : number of points in the airfoils
        ##

        # number of points in x axis (total number or points is npoints*2-1)
        npoints = 41

        # Initialize array to store airfoil coordinates at every element
        self.foilBEM = np.zeros((n, 2, npoints*2-1))

        # Interpolate airfoils
        for i in range(n):
            for j in range(len(self.r)):
                if self.r[j] >= self.rBEM[i]:
                    # eta = 0 -> Foil1 | eta = 1 -> Foil2
                    eta = (self.rBEM[i]-self.r[j-1])/(self.r[j]-self.r[j-1])

                    # Separate upper and lower surfaces on airfoil at eta=0
                    for k in range(np.shape(arf)[2]):
                        if arf[j-1,0,k+1]>arf[j-1,0,k]:
                            foil1u = arf[j-1, :, :k+1]
                            foil1l = arf[j-1, :, k:]
                            break

                    # Separate upper and lower surfaces on airfoil at eta=1
                    for k in range(np.shape(arf)[2]):
                        if arf[j,0,k+1]>arf[j,0,k]:
                            foil2u = arf[j, :, :k+1]
                            foil2l = arf[j, :, k:]
                            break

                    # New sampling points along x
                    #Initialize array
                    x_new = np.zeros(npoints)
                    # 10% of points in first 2% of chord
                    x_new[0:npoints//10] = np.linspace(0, 0.02,npoints//10, endpoint=False)
                    # 20% of points in first 10% of chord
                    x_new[npoints//10:npoints//5] = np.linspace(0.02,0.1,npoints//10, endpoint=False)
                    # Remainder points in the rest 90% of chord
                    x_new[npoints//5:] = np.linspace(0.1,1,npoints-npoints//5)

                    # Resample upper surface of foil 1
                    y1u = interpolate.interp1d(foil1u[0,:], foil1u[1,:])(x_new)

                    # Resample lower surface of foil 1
                    y1l = interpolate.interp1d(foil1l[0,:], foil1l[1,:])(x_new)

                    # Resample upper surface of foil 2
                    y2u = interpolate.interp1d(foil2u[0,:], foil2u[1,:])(x_new)

                    # Resample lower surface of foil 2
                    y2l = interpolate.interp1d(foil2l[0,:], foil2l[1,:])(x_new)

                    # Interpolate foil at position eta
                    y3u = np.zeros(npoints)
                    y3l = np.zeros(npoints)

                    for k in range(npoints):
                        # Interpolate upper surface
                        y3u[k] = y1u[k]*(1-eta) + y2u[k]*eta

                        # Interpolate lower surface
                        y3l[k] = y1l[k]*(1-eta) + y2l[k]*eta

                    # Assemble geometry in a format compatible with XFOIL
                    # X
                    self.foilBEM[i, 0, :npoints] = np.flip(x_new)
                    self.foilBEM[i, 0, npoints-1:] = x_new

                    # Y
                    self.foilBEM[i, 1, :npoints] = np.flip(y3u)
                    self.foilBEM[i, 1, npoints-1:] = y3l
                    break

    def bem_elements(self, n):
        ##
        # Description:
        #   Interpolates all propeller blade data for each BEM element
        ##
        # Parameters:
        #   n : float
        #       Number of blade elements
        ##

        # Calculate radial position of each blade element
        self.rBEM = np.array([i/n*(self.r[-1]-self.r[0])+self.r[0]+(self.r[-1]-self.r[0])/2/n for i in range(n)])

        # Interpolate blade chord at each blade element
        self.cBEM = interpolate.interp1d(self.r, self.c)(self.rBEM)

        # Interpolate blade twist at each blade element
        self.twistBEM = interpolate.interp1d(self.r, self.twist)(self.rBEM)

        # Import propeller airfoils for each blade section
        # 1 : Which .dat file?
        # 2 : Columns in the .dat file
        # 3 : Lines in the .dat file
        arf = []
        for i in range(len(self.r)):
            name = self.Foil[i] + ".dat"
            arf.append(Propeller.import_airfoil(name))

        arf = np.array(arf)

        # Interpolate airfoil coordinates
        self.interp_airfoil(n, arf)

        # Create polar objects
        self.polarsBEM = []
        for i in range(n):
            self.polarsBEM.append(Polar(self.foilBEM[i, 0, :], self.foilBEM[i, 1, :]))

    def element_loading(self, w, psi, n, airflow):
        ##
        # Description:
        #   Calculates the trangential and axial forces on a blade element on
        #       a certain azimuth and radial position
        ##
        # Parameters:
        #   w : float
        #       Angular velocity of the propeller in rpm
        #   psi : float
        #       Azimuthal position of the blade element [deg]
        #   n : integer
        #       Index of the blade element
        #   airflow : object
        #       Airflow in the propeller domain
        ##
        # Returns:
        #   Fa : float
        #       Axial force in the element
        #   Ft : float
        #       Tangential force in the element
        #   V : float
        #       Flow velocity at the element [m/s]
        #   phi : float
        #       Inflow angle [rad]
        ##

        it_limit = 1000
        count = 0
        error = False

        # Air properties
        rho = airflow.rho
        mu = airflow.mu

        # Absolute airflow velocity at the element
        un, ut, ur = airflow.get_airflow(psi, self.rBEM[n])

        # Relative airflow velocity at the element
        Un = un
        Ut = w/60*2*math.pi*self.rBEM[n]-ut

        # Induction coefficients initialization
        aa = 0
        at = 0
        # Iteration loop for induction coefficients
        while 1:
            # Increment iteration counter
            count +=1
            # Check iterarion limit
            if count>it_limit:
                error = True
                break
            # Stop in case of nan
            if np.isnan(aa) or np.isnan(at):
                error = True
                break

            # Calculate Reynolds number
            V = np.sqrt((Un*(1+aa))**2 + (Ut*(1-at))**2)
            Re = rho * V * self.cBEM[n]/mu

            #print(self.cBEM[n])
            # Calculate inflow angle
            phi = np.arctan((Un*(1+aa))/(Ut*(1-at)))

            # Calculate angle of attack
            alpha = np.deg2rad(self.twistBEM[n])-phi

            # Lift and Drag coefficients at appropriate Reynolds number and
            #   angle of attack
            Cl, Cd = self.polarsBEM[n].polar_get(np.rad2deg(alpha), Re)

            # Tangent and normal force coefficients
            Ca = Cl*np.cos(phi)-Cd*np.sin(phi)
            Ct = Cl*np.sin(phi)+Cd*np.cos(phi)

            # Calculate Prandtl's correction factor
            aux = np.exp(-self.B/2*(self.r[-1]-self.rBEM[n])/(self.rBEM[n]*np.sin(phi)))

            if aux>1:

                error = True
                break

            F = 2/math.pi * np.arccos(aux)

            # Calculate the element's solidity ratio
            s = self.cBEM[n]*self.B/2/math.pi/self.rBEM[n]

            # Calculate new induction coefficients
            at_new = ((4*F*np.sin(phi)*np.cos(phi))/(s*Ct)+1)**(-1)

            #starttime = timeit.default_timer()
            #aa_new = ((4*F*np.sin(phi)**2)/(s*Ca)-1)**(-1)      # As in silvestre2013

            #################################################################### as in hansen2017
            # explicado em: 36881.pdf

            C_T = s*(1-aa)**2*(Cl*np.cos(phi)+Cd*np.cos(phi))/(np.sin(phi)**2)

            #aa_new = float(optimize.minimize_scalar(Propeller.calculate_CT_optimize, bounds=[(0., 10.)], args=(F, C_T), method='Golden').x)
            #aa_new = float(optimize.minimize_scalar(Propeller.calculate_CT_optimize, bounds=[(0., 10.)], args=(F, C_T), method='Golden', tol=0.01).x)

            aa_new = 0
            while 1:
                deltaCT = Propeller.calculate_CT_optimize(aa_new, F, C_T)

                aa_new1 = aa_new+deltaCT*0.01

                if abs(aa_new-aa_new1)/aa_new1<0.01:
                    aa_new = aa_new1
                    break
                aa_new = aa_new1


            ####################################################################
            # Correction to aa_new due to non axial flow 36881.pdf/skewed wake correction

            # Calculate skew angle
            qui = np.arccos((un)/(np.linalg.norm([un, ut, ur])))

            # Apply correction only if flow is squewed
            if qui != 0:
                # Calculate downwing azimuth
                psi0 = np.arccos((ur)/(np.linalg.norm([ut, ur])))

                # Calculate correction factor
                fskew = 1 + 15/32*math.pi*self.rBEM[n]/self.r[-1]*np.tan(qui/2)*np.cos(np.deg2rad(psi)-psi0)

                # Apply correction factor
                aa_new = aa_new*fskew

            ####################################################################
            #print("The time difference is :", timeit.default_timer() - starttime)
            # Compare new to previous induction coefficients
            ERat = abs(at_new-at)/at_new
            ERaa = abs(aa_new-aa)/aa_new

            # Interation shift
            at = at + (at_new-at) * 0.1
            aa = aa + (aa_new-aa) * 0.1

            #print(aa)

            # Stop iterating if error is below threshold
            if ERat<0.01 and ERaa<0.01:
                V = np.sqrt((Un*(1+aa))**2 + (Ut*(1-at))**2)
                break

        # Calculate axial and tangential forces
        Fa = Ca * 0.5 * rho * V**2 * self.cBEM[n] * (self.r[-1]-self.r[0])/len(self.rBEM)
        Ft = Ct * 0.5 * rho * V**2 * self.cBEM[n] * (self.r[-1]-self.r[0])/len(self.rBEM)
        #print(Re, np.rad2deg(alpha), Cl, aa)
        # Null forces in case of nan
        if error:
            #print("ERROR in BEM, n = ", n)
            Fa = 0
            Ft = 0

        # Flow velocity at the element
        Un = Un*(1+aa)          # Positive along z axis
        Ut = Ut*(1-at)          # Positive in the direction of positive azimuth
        Ur = ur                 # Positive outwards

        return Fa, Ft, np.array([Un, Ut, Ur])

    def avg_thrust(self, w, airflow, npsi=20):
        ##
        # Description:
        #   Calculates the average thrust of the propeller
        ##
        # Parameters:
        #   w : float
        #       Angular velocity of the propeller
        #
        #   airflow : object
        #       Airflow in the propeller domain
        #   npsi (optional) : float
        #       Number of azimuthal positions to sample
        ##
        # Returns:
        #   T : float
        #       Average thrust
        ##

        # Number of blade elements
        nBEM = len(self.rBEM)

        # Azimuthal positions to sample
        psi = np.linspace(0, 360, npsi, endpoint=False)

        # Initialize Thrust variable
        T = 0.

        for i in range(len(psi)):
            for j in range(nBEM):
                # Calculate element forces
                Fa = self.element_loading(w, psi[i], j, airflow)[0]

                # Add axial force to thrust
                T += Fa

        # Average thrust
        T = T/npsi * self.B

        return T

    def calculate_CT_optimize(aa, F, C_T):
        # Calculate CT
        if aa <0.3:
            C_T_new = 4*aa*F*(1-aa)
        else:
            C_T_new = 4*aa*F*(1-(5-3*aa)*aa/4)

        return (C_T-C_T_new)

    def farassat_1A(self, psi, w, airflow, robs, vobs, m=10):
        # psi [deg] array of azimuth of each blade element
        # w = angular velocity [rpm]

        # robs [m] = observer position in propeller rectangular referential
        # vobs [m] = observer velocity in propeller rectangular referential
        # m = number of chordwise elements (used for thickness noise)

        # Air properties
        c = airflow.c       # c [m/s] speed of sound
        rho = airflow.rho

        # Number of blade elements
        n = len(self.rBEM)

        # Convert psi to radians
        for i in range(n):
            psi[i]= np.deg2rad(psi[i])

        # =========================== Loading Noise ============================

        # Initialize element load arrays
        Fa = np.zeros(n)            # Axial load
        Ft = np.zeros(n)            # Tangential load
        L = np.zeros([n,3])         # (x y z) N

        # Initialize local flow velocity array
        u = np.zeros([n, 3])

        # Initializa Inflow Angle array
        #phi = np.zeros(n)

        # Calculate load of all elements
        for i in range(n):
            Fa[i], Ft[i], u[i,:] = self.element_loading(w, np.rad2deg(psi[i]), i, airflow)

            # X direction
            L[i,0] = Ft[i] * -np.cos(psi[i])

            # Y direction
            L[i,1] = Ft[i] * -np.sin(psi[i])

            # Z direction
            L[i,2] = -Fa[i]


        # Convert flow velocity to rectangular coordinates
        for i in range(n):
            aux1 = -u[i, 2]*np.sin(psi[i]) - u[i, 1]*np.cos(psi[i])   # X
            aux2 = u[i, 2]*np.cos(psi[i]) - u[i, 1]*np.sin(psi[i])    # Y
            aux3 = u[i, 0]                                            # Z
            u[i,:] = [aux1, aux2, aux3]

        # Delta psi in degrees
        deltapsi = 1
        dt = 2*deltapsi/360*(1/(w/60))

        # Calculate load time derivatices of all elements using finite diferences, n=2. https://en.wikipedia.org/wiki/Finite_difference_coefficient
        Ldot = np.zeros([n,3])
        for i in range(n):
            # Before current time
            a1, a2 = self.element_loading(w, np.rad2deg(psi[i])-deltapsi, i, airflow)[0:2]
            # Load (x y z)
            A = [a2 * -np.cos(psi[i]), a2 * -np.sin(psi[i]), -a1]

            # After current time
            b1, b2 = self.element_loading(w, np.rad2deg(psi[i])+deltapsi, i, airflow)[0:2]
            # Load (x y z)
            B = [b2 * -np.cos(psi[i]), b2 * -np.sin(psi[i]), -b1]

            # Calculate derivative (finite difference)
            Ldot[i,:] = np.divide(np.add(np.multiply(-0.5,A), np.multiply(0.5,B)), dt)

        # Calculate load time derivatices of all elements using finite diferences, n=2. https://en.wikipedia.org/wiki/Finite_difference_coefficient
        Mdot = np.zeros([n,3])
        for i in range(n):
            # Before current time
            A = self.element_loading(w, np.rad2deg(psi[i])-deltapsi, i, airflow)[2]
            # Speed to Mach
            A = np.divide(A, c)

            # After current time
            B = self.element_loading(w, np.rad2deg(psi[i])+deltapsi, i, airflow)[2]
            # Speed to Mach
            B = np.divide(B, c)

            # Calculate derivative (finite difference)
            Mdot[i,:] = np.divide(np.add(np.multiply(-0.5,A), np.multiply(0.5,B)), dt)

            # Convert to rectangualar coordinates   # Added on 26/05/2020
            aux1 = -Mdot[i, 2]*np.sin(psi[i]) - Mdot[i, 1]*np.cos(psi[i])   # X
            aux2 = Mdot[i, 2]*np.cos(psi[i]) - Mdot[i, 1]*np.sin(psi[i])    # Y
            aux3 = Mdot[i, 0]                                            # Z
            Mdot[i,:] = [aux1, aux2, aux3]

        # Mach number
        M = np.zeros([n, 3])
        for i in range(n):
            M[i,:] = np.divide(u[i,:],c)

        # Flow velocity versor (x y z)
        uhat = np.zeros([n, 3])
        for i in range(n):
            uhat[i,:] = np.divide(u[i,:], np.linalg.norm(u[i,:]))

        # Radiation direction for each element (x y z)
        r = np.zeros([n, 3])
        for i in range(n):
            # Blade element coordinates
            x = self.rBEM[i] * -np.sin(psi[i])
            y = self.rBEM[i] * np.cos(psi[i])

            # Raditation direction = observer - element
            r[i,:] = np.subtract(robs, [x, y, 0])

        # Mach number in radiation direction
        Mr = np.zeros(n)
        for i in range(n):
            # Flow velocity in radiation direction
            V = np.dot(u[i,:], np.divide(r[i,:], np.linalg.norm(r[i,:])))

            # Mach number in radiation direction
            Mr[i] = V/c

        # Element loading in raditaion direction
        Lr = np.zeros(n)
        for i in range(n):
            Lr[i] = np.dot(L[i,:], np.divide(r[i,:], np.linalg.norm(r[i,:])))

        # 1st integral
        # Calculate integrand array
        y = np.zeros(n)
        for i in range(n):
            y[i] = np.dot(Ldot[i,:], np.divide(r[i,:], np.linalg.norm(r[i,:])))/(np.linalg.norm(r[i,:])*(1-Mr[i])**2)

        # Calculate integral
        I1 = integrate.simpson(y, self.rBEM)/c

        # 2nd integral
        # Calculate integrand array
        y = np.zeros(n)
        for i in range(n):
            # Numerator
            aux1 = Lr[i] - np.dot(L[i,:], M[i,:])

            # Denominator
            aux2 = np.linalg.norm(r[i,:])**2 * (1-Mr[i])**2

            # Integrand
            y[i] = aux1/aux2

        # Calculate integral
        I2 = integrate.simpson(y, self.rBEM)

        # 3rd integral
        # Calculate integrand array
        y = np.zeros(n)
        for i in range(n):
            # Numerator
            aux1 = Lr[i] * (np.linalg.norm(r[i,:]) * np.dot(Mdot[i,:], np.divide(r[i,:],np.linalg.norm(r[i,:]))) + c*Mr[i] - c*np.linalg.norm(M[i,:])**2)

            # Denominator
            aux2 = np.linalg.norm(r[i,:])**2 * (1-Mr[i])**3

            # Integrand
            y[i] = aux1/aux2

        # Calculate integral
        I3 = integrate.simpson(y, self.rBEM)/c

        # Loading noise pressure
        pL = I1 + I2 + I3
        pL = pL/(4*math.pi)


        # ========================== Thickness Noise ===========================
        # Calculate normal direction for every spanwise and chordwise elements
        # Upper surface
        nupper = np.zeros([m, n, 3])    # [chordwise position, spanwise position, (x y z)]
        # Upper surface
        nlower = np.zeros([m, n, 3])    # [chordwise position, spanwise position, (x y z)]

        # For every spanwise position
        for i in range(n):
            # Separate upper and lower surfaces on airfoil at the spanwise position
            arf = self.foilBEM[i,:,:]       # arf = (0/1, x index) 0 = x; 1 = y
            for k in range(np.shape(arf)[1]):
                if arf[0,k+1]>arf[0,k]:
                    foilu = arf[:, :k+1]
                    foill = arf[:, k:]
                    break

            # Coordinates of the 11 points along the chord
            samplepoints = np.linspace(0, 1, m+1)

            # Determine y coordinates
            # Upper surface
            upperpoints = interpolate.interp1d(foilu[0,:], foilu[1,:])(samplepoints)
            # Lower surface
            lowerpoints = interpolate.interp1d(foill[0,:], foill[1,:])(samplepoints)

            # Calculate the surface direction (positive pointing towars trailing edge)
            # Upper surface
            supper = np.zeros([m, 2]) # (x, y) in aifoil coordinate system
            for j in range(m):
                supper[j,:] = np.subtract([foilu[0,j+1], foilu[1,j+1]], [foilu[0,j], foilu[1,j]])

            # Lower surface
            slower = np.zeros([m, 2]) # (x, y) in aifoil coordinate system
            for j in range(m):
                slower[j,:] = np.subtract([foill[0,j+1], foill[1,j+1]], [foill[0,j], foill[1,j]])

            # Calculate owtwards pointing vector (in local airfoil coordinates)
            for j in range(m):
                # Upper surface
                nupper[j, i, :2] = [-supper[j,1], supper[j,0]]  # (x y 0)

                # Lower surface
                nlower[j, i, :2] = [slower[j,1], -slower[j,0]]  # (x y 0)

            # Transform owtwards pointing vector to propeller Coordinates
            for j in range(m):
                # Upper surface
                # Transform to propeller cylindrical coordinates
                aux1 = -(np.cos(np.deg2rad(self.twistBEM[i]))*nupper[j,i,1] + np.sin(np.deg2rad(self.twistBEM[i]))*nupper[j,i,0])         # Normal direction (points along rectangular z axis )
                aux2 = -(np.sin(np.deg2rad(self.twistBEM[i]))*nupper[j,i,1] - np.cos(np.deg2rad(self.twistBEM[i]))*nupper[j,i,0])         # Tangential (points along azimuth angle)

                # Transform to propeller rectangular coordinates
                nupper[j, i, 0] = aux2 * -np.cos(psi[i])    # X
                nupper[j, i, 1] = aux2 * -np.sin(psi[i])    # Y
                nupper[j, i, 2] = aux1                      # Z

                # Normalize vector
                nupper[j, i, :] = np.divide(nupper[j, i, :], np.linalg.norm(nupper[j, i, :]))

                # Lower surface
                # Transform to propeller cylindrical coordinates
                aux1 = -(np.cos(np.deg2rad(self.twistBEM[i]))*nlower[j,i,1] + np.sin(np.deg2rad(self.twistBEM[i]))*nlower[j,i,0])         # Normal direction (points along rectangular z axis )
                aux2 = -(np.sin(np.deg2rad(self.twistBEM[i]))*nlower[j,i,1] - np.cos(np.deg2rad(self.twistBEM[i]))*nlower[j,i,0])         # Tangential (points along azimuth angle)

                # Transform to propeller rectangular coordinates
                nlower[j, i, 0] = aux2 * -np.cos(psi[i])    # X
                nlower[j, i, 1] = aux2 * -np.sin(psi[i])    # Y
                nlower[j, i, 2] = aux1                      # Z

                # Normalize vector
                nlower[j, i, :] = np.divide(nlower[j, i, :], np.linalg.norm(nlower[j, i, :]))

        # Calculate velocity for every spanwise blade element (relative to the observer)
        v = np.zeros([n,3]) # (x y z)

        for i in range(n):  # Alterado 26/05/2021 (adicionei 2*math.pi)
            v[i,:] = np.add(np.multiply(-1, vobs), [self.rBEM[i]*2*math.pi*w/60 * -np.cos(psi[i]), self.rBEM[i]*2*math.pi*w/60 * -np.sin(psi[i]), 0])

        # Calculate velocity time derivative for every spanwise blade element (relative to the opserver) using finite diferences, n=2
        vdot = np.zeros([n,3]) # (x y z)

        for i in range(n):  # Alterado 26/05/2021 (adicionei 2*math.pi)
            # Before current time
            A = np.add(np.multiply(-1, vobs), [self.rBEM[i]*2*math.pi*w/60 * -np.cos(psi[i]-np.deg2rad(deltapsi)), self.rBEM[i]*2*math.pi*w/60 * -np.sin(psi[i]-np.deg2rad(deltapsi)), 0])

            # After current time
            B = np.add(np.multiply(-1, vobs), [self.rBEM[i]*2*math.pi*w/60 * -np.cos(psi[i]+np.deg2rad(deltapsi)), self.rBEM[i]*2*math.pi*w/60 * -np.sin(psi[i]+np.deg2rad(deltapsi)), 0])

            # Calculate derivative (finite difference)
            vdot[i,:] = np.divide(np.add(np.multiply(-0.5,A), np.multiply(0.5,B)), dt)


        # Calculate chordwise integration domain
        chordpoints = np.zeros(m)
        for i in range(m):
            chordpoints[i] = (samplepoints[i+1]+samplepoints[i])*0.5

        xc = np.zeros([m,n])
        for i in range(n):
            xc[:,i] = np.multiply(chordpoints, self.cBEM[i])


        # 1st integral
        # Calculate integrands
        y1upper = np.zeros(m)
        y2upper = np.zeros(n)
        y1lower = np.zeros(m)
        y2lower = np.zeros(n)

        for i in range(n):
            for j in range(m):
                # Upper face
                # Numerator
                aux1 = rho * np.dot(vdot[i,:], nupper[j,i,:])
                # Denominator
                aux2 = np.linalg.norm(r[i,:])*(1-Mr[i])**2

                # Chordwise Integrand
                y1upper[j] = aux1/aux2

                # Lower face
                # Numerator
                aux1 = rho * np.dot(vdot[i,:], nlower[j,i,:])

                # Chordwise Integrand
                y1lower[j] = aux1/aux2

            #Calculate chordwise integral (which is the spanwise integrand)
            # Upper face
            y2upper[i] = integrate.simpson(y1upper, xc[:,i])

            # Lower face
            y2lower[i] = integrate.simpson(y1lower, xc[:,i])

        # Calculate integral
        I1 = integrate.simpson(y2upper, self.rBEM) + integrate.simpson(y2lower, self.rBEM)


        # 2nd integral
        # Calculate integrands
        y1upper = np.zeros(m)
        y2upper = np.zeros(n)
        y1lower = np.zeros(m)
        y2lower = np.zeros(n)

        for i in range(n):
            for j in range(m):
                # Upper face
                # Numerator
                aux1 = np.linalg.norm(r[i,:]) * np.dot(Mdot[i,:],np.divide(r[i,:], np.linalg.norm(r[i,:]))) + c*Mr[i] - c*np.linalg.norm(M[i,:])**2
                aux1 = aux1 * rho * np.dot(v[i,:], nupper[j,i,:])

                # Denominator
                aux2 = np.linalg.norm(r[i,:])**2 * (1-Mr[i])**3

                # Chordwise Integrand
                y1upper[j] = aux1/aux2


                # Lower face
                # Numerator
                aux1 = np.linalg.norm(r[i,:]) * np.dot(Mdot[i,:],np.divide(r[i,:], np.linalg.norm(r[i,:]))) + c*Mr[i] - c*np.linalg.norm(M[i,:])**2
                aux1 = aux1 * rho * np.dot(v[i,:], nlower[j,i,:])

                # Chordwise Integrand
                y1lower[j] = aux1/aux2

            #Calculate chordwise integral (which is the spanwise integrand)
            # Upper face
            y2upper[i] = integrate.simpson(y1upper, xc[:,i])

            # Lower face
            y2lower[i] = integrate.simpson(y1lower, xc[:,i])

        # Calculate integral
        I2 = integrate.simpson(y2upper, self.rBEM) + integrate.simpson(y2lower, self.rBEM)

        # Thickness noise pressure
        pT = I1 + I2
        pT = pT/(4*math.pi)

        return pT, pL

    def blade_position(self, t, airflow, robs, w):
        # Position of every blade element over time. t = 0 is equivalent to the element closest to the root being at azimuth = 0

        # t[s] - observer time at which the blade position must be calculated
        # airflow - airflow class instance

        # Number of blade elements
        n = len(self.rBEM)

        # Speed of sound
        c = airflow.c

        # Find position at required t
        psi = np.zeros(n)
        tau = np.zeros(n)   # is zero when psi is equal to zero
        for i in range(n):
            while True:
                # Position of the element in propeller rectangular Coordinates
                x = self.rBEM[i] * -np.sin(psi[i])
                y = self.rBEM[i] * np.cos(psi[i])

                # Distance between the observer and the blade element
                r = np.linalg.norm(np.subtract([x, y, 0], robs))

                # Retarded time
                tau[i] = t-r/c

                # Calculate new element position
                psi_new = 2*math.pi*w/60 * tau[i]

                # Error in element position
                Ea = abs(psi[i]-psi_new)

                # Iteration carry over
                psi[i] = psi_new

                # Error must be less than 0.1 deg
                if Ea<=np.deg2rad(0.1):
                    psi[i] = np.rad2deg(psi[i])
                    break

        return psi

    def noise(self, T, t, airflow, robs, vobs, w, m=40):
        # t[s] - observer times at which to calculate propeller noise
        T = 1
        # ========================== Rotational Noise ==========================
        # Create arrays for rotational noise sound pressure
        pT = np.zeros(len(t))
        pL = np.zeros(len(t))
        pblade = np.zeros(len(t))

        # For every observer time
        for i in range(len(t)):
            print("t = " + str(t[i]) + " s" + " - " + str(t[i]/t[-1]*100)[:4] + "%")
            psi = self.blade_position(t[i], airflow, robs, w)

            pT[i], pL[i] = self.farassat_1A(psi, w, airflow, robs, vobs, m)
            pblade[i] = (pT[i]) + (pL[i])
            #print(pblade[i], pT[i], pL[i])
        """
        print("pT:")
        for i in range(len(pT)):
            print(pT[i])
        print("pL:")
        for i in range(len(pL)):
            print(pL[i])
        """

        # Interpolate rotational sound pressure
        prot = interpolate.interp1d(t, pblade)

        # Period of rotation
        Tau = 60/w

        ti = t[0]
        p = np.zeros(len(t))
        for i in range(self.B):
            # Calculate observer times for the blade i
            tphase = [x-Tau*(i)/(self.B) for x in t]
            tphase = [x + Tau if x < t[0] else x for x in tphase]

            # Calculate rotational sound pressure
            p = np.add(p, prot(tphase))

        #print("p:")
        #for i in range(len(p)):
            #print(p[i])

        # Calculate root mean square of rotational noise sound pressure
        RMSp = (np.sum([x**2 for x in p])/len(p))**0.5

        # Calculate rotational noise SPL
        SPLrot = 20*np.log10(RMSp/2e-5)

        # ============================ Total Noise =============================

        SPL = 10 * np.log10(10**(SPLrot/10))

        return SPL, SPLrot, 0, p, pT, pL

class Airflow:
    def __init__(self, airflowname, centre, rho, mu, c):
        ##
        # Parameters:
        #   airflowname : string
        #       Name of the file and directory containing airflow data at the
        #           propeller plane
        #   centre : array 2 x 1
        #       array with the coordinates [x, y] of the centre point of the
        #           radial referential
        #   rho : float
        #       air density
        #   mu : float
        #       air viscosity
        #   c : float
        #       speed of sound [m/s]
        ##

        # Centre of the radial reference frame
        self.centre = centre

        # Air density
        self.rho = rho

        # Air viscosity
        self.mu = mu

        # Speed of Sound
        self.c = c

        # Import airflow data from file
        with open(airflowname, "r") as file:
            # Read Airfoil name
            file.readline()

            # Read Airfoil data
            table = []
            for line in file:
                table.append([float(x) for x in line.split("\n")[0].split("\t")[:]])

            table = np.array(table)

        # Fix for error when u = const
        #for i in range(np.shape(table)[0]):
            #table[i,2] += random.uniform(-1e-3, 1e-3)
            #table[i,3] += random.uniform(-1e-3, 1e-3)
            #table[i,4] += random.uniform(-1e-3, 1e-3)

        # Interpolate velocity's x, y and z values in the rectangular domain
        #self.ux = interpolate.interp2d(table[:,0], table[:,1], table[:,2], kind="cubic")
        #self.uy = interpolate.interp2d(table[:,0], table[:,1], table[:,3], kind="cubic")
        #self.uz = interpolate.interp2d(table[:,0], table[:,1], table[:,4], kind="cubic")

        # Arrays of in plane position
        self.xx = table[:,0]
        self.yy = table[:,1]

        # Arrays of ux, uy and uz
        self.ux = table[:,2]
        self.uy = table[:,3]
        self.uz = table[:,4]

        # Vinf
        self.Vinf = np.mean(self.uz)

    def get_airflow(self, psi, r):
        ##
        # Description:
        #   Retrieves the tangential and normal velocity components at radial
        #       and azimuthal position
        ##
        # Method Parameters:
        #   psi : float
        #       Azimuthal position, 0 == y+ axis (in degrees)
        #   r : float
        #       radial position
        ##
        # Returns:
        #   un : float
        #       Velocity component normal to the propeller plane
        #   ut : float
        #       Velocity component perpendicular to the radius
        ##

        # Position in rectangular coordinates
        Dx = r * np.cos(np.deg2rad(psi+90))
        Dy = r * np.sin(np.deg2rad(psi+90))

        x = Dx + self.centre[0]
        y = Dy + self.centre[1]

        # Interpolate Airflow in the propeller plane
        #uxaux = self.ux(x, y)
        #uyaux = self.uy(x, y)
        uxaux = interpolate.griddata((self.xx, self.yy), self.ux, (x, y), method='linear')
        uyaux = interpolate.griddata((self.xx, self.yy), self.uy, (x, y), method='linear')

        # Tangential flow velocity
        direction = psi + 90 # Tangent direction
        versor = [np.cos(np.deg2rad(direction+90)), np.sin(np.deg2rad(direction+90))]

        ut = float(uxaux*versor[0] + uyaux*versor[1])

        # Radial flow velocity (positive pointing outwards)
        direction = psi # radial direction
        versor = [np.cos(np.deg2rad(direction+90)), np.sin(np.deg2rad(direction+90))]

        ur = float(uxaux*versor[0] + uyaux*versor[1])

        # Normal flow velocity
        #un = float(self.uz(x, y))
        un = interpolate.griddata((self.xx, self.yy), self.uz, (x, y), method='linear')

        return un, ut, ur

class Polar:
    def __init__(self, x, y):
        ##
        # Parameters:
        #   x : array n x 1
        #       array with all X coordinates in the format 1 ... 0 ... 1 from upper surface to lower surface
        #   y : array n x 1
        #       array with corresponding Y coordinates
        ##

        # x coordinate array
        self.x = x

        # y coordinate array
        self.y = y

        # Reynolds warnings
        self.warnings = False

    def extended_polar_Viterna(self, Reynolds):
        ##
        # Description:
        #   Generates the extended 360 degree polar of an airfoil using Viterna method
        ##
        # Method Parameters:
        #   Re : float
        #       Reynolds number at which the airfoil is operating
        ##
        # Returns:
        #   a_extended : array
        #       360 x 1 matrix with all sampled angles of attack in degrees
        #
        #   cl_extended : array
        #       360 x 1 matrix with all sampled lift coefficient values
        #
        #   cd_extended : array
        #       360 x 1 matrix with all sampled drag coefficient values
        ##

        # lift coefficient adjustment to account for assymetry
        cl_adj = 0.7

        # Generate airfoil class instance
        xf = XFoil()

        xf.print = True

        # Import airfoil
        xf.airfoil = Airfoil(self.x, self.y)

        # XFOIL analysis
        xf.Re = Reynolds
        xf.M = 0
        xf.n_crit = 9
        xf.max_iter = 50
        a_fine, cl_fine, cd_fine = xf.aseq(-10, 20, 0.1)[0:3]    # From -10 to 20 degrees with 1 degree steps

        # Replace nan when surrounded by converged values
        for i in range(10, len(a_fine)-10):
            for j in range(10):
                if np.isnan(a_fine[i]) and np.isnan(a_fine[i-j])==False and np.isnan(a_fine[i+j])==False:
                    a_fine[i] = (a_fine[i-j]+a_fine[i+j])/2
                    cl_fine[i] = (cl_fine[i-j]+cl_fine[i+j])/2
                    cd_fine[i] = (cd_fine[i-j]+cd_fine[i+j])/2
                    break


        # Keep values for integer angle of attack
        a = []
        cl = []
        cd = []
        for i in range(0, len(a_fine), 10):
            a.append(a_fine[i])
            cl.append(cl_fine[i])
            cd.append(cd_fine[i])

        a = np.array(a)
        cl = np.array(cl)
        cd = np.array(cd)
        print(a)

        # Trim upper limit of numerical error
        for i in range(list(a).index(0), len(a)):
            if math.isnan(cl[i]) or cl[i]<cl[i-1]:
                a = np.delete(a, [j for j in range(i, len(a))])
                cl = np.delete(cl, [j for j in range(i, len(cl))])
                cd = np.delete(cd, [j for j in range(i, len(cd))])
                break

        # Trim lower limit of polars in case of numerical error
        for i in range(list(a).index(0), -1, -1):
            if math.isnan(cl[i]) or cl[i+1]<cl[i]:
                a = np.delete(a, [j for j in range(i+1)])
                cl = np.delete(cl, [j for j in range(i+1)])
                cd = np.delete(cd, [j for j in range(i+1)])
                break

        # Upper limits
        a_s = a[-1]
        cl_s = cl[-1]
        cd_s = cd[-1]

        # Lower limits
        a_l = a[0]
        cl_l = cl[0]
        cd_l = cd[0]

        # Viterna coefficients
        B1 = 1.8
        B2 = cd_s-B1*np.sin(np.deg2rad(a_s))**2/np.cos(np.deg2rad(a_s))
        A1 = B1/2
        A2 = (cl_s-B1*np.sin(np.deg2rad(a_s))*np.cos(np.deg2rad(a_s)))*np.sin(np.deg2rad(a_s))/np.cos(np.deg2rad(a_s))**2


        # Alpha = 0 is in index 179
        a_extended = np.zeros(360)
        cl_extended = np.zeros(360)
        cd_extended = np.zeros(360)

        # Define alpha array from -179deg to 180 deg
        a_extended = [i-179 for i in range(len(a_extended))]

        # Generate extended 360 degree polars

        # Alpha min to alpha stall
        cl_extended[179-list(a).index(0):179+len(a)-list(a).index(0)] = cl[:]
        cd_extended[179-list(a).index(0):179+len(a)-list(a).index(0)] = cd[:]

        # Extend from alpha stall to 90 deg
        for i in range(179+len(a)-list(a).index(0), 270):
            cl_extended[i] = A1*np.sin(np.deg2rad(2*a_extended[i]))+A2*np.cos(np.deg2rad(a_extended[i]))**2/np.sin(np.deg2rad(a_extended[i]))
            cd_extended[i] = B1*np.sin(np.deg2rad(a_extended[i]))**2+B2*np.cos(np.deg2rad(a_extended[i]))

        # Extend from 90 deg to 180 deg - alpha stall
        for i in range(270, 360-len(a)+list(a).index(0)):
            alpha = 180-a_extended[i]
            cl_extended[i] = -cl_adj*(A1*np.sin(np.deg2rad(2*alpha))+A2*np.cos(np.deg2rad(alpha))**2/np.sin(np.deg2rad(alpha)))
            cd_extended[i] = B1*np.sin(np.deg2rad(alpha))**2+B2*np.cos(np.deg2rad(alpha))

        # Extend from 180 deg - alpha stall to 180
        for i in range(360-len(a)+list(a).index(0), 360):
            alpha = 180-a_extended[i]
            cl_extended[i] = -alpha/a_s*cl_s*cl_adj # linear variation
            cd_extended[i] = B1*np.sin(np.deg2rad(alpha))**2+B2*np.cos(np.deg2rad(alpha))

        # Extend from -180 to -180+alpha high
        for i in range(len(a)-list(a).index(0)-1):
            alpha = a_extended[i]+180
            cl_extended[i] = alpha/a_s*cl_s*cl_adj# linear variation
            cd_extended[i] = B1*np.sin(np.deg2rad(alpha))**2+B2*np.cos(np.deg2rad(alpha))

        # Extend from -180+alpha high to -90
        for i in range(len(a)-list(a).index(0)-1, 90):
            alpha = a_extended[i]+180
            cl_extended[i] = cl_adj*(A1*np.sin(np.deg2rad(2*alpha))+A2*np.cos(np.deg2rad(alpha))**2/np.sin(np.deg2rad(alpha)))
            cd_extended[i] = B1*np.sin(np.deg2rad(alpha))**2+B2*np.cos(np.deg2rad(alpha))

        # Extend from -90 to -alpha high
        for i in range(90, 179-len(a)+list(a).index(0)+1):
            alpha = -a_extended[i]
            cl_extended[i] = -cl_adj*(A1*np.sin(np.deg2rad(2*alpha))+A2*np.cos(np.deg2rad(alpha))**2/np.sin(np.deg2rad(alpha)))
            #cd_extended[i] = B1*np.sin(np.deg2rad(alpha))**2+B2*np.cos(np.deg2rad(alpha))
        cd_extended[90:179-len(a)+list(a).index(0)+1] = np.flip(cd_extended[179+len(a)-list(a).index(0)+1:270])

        # Extend from -alpha high to alpha low
        x_interp = [a_extended[179-len(a)+list(a).index(0)], a_extended[179-list(a).index(0)+3]]
        y_interp = [cd_extended[179-len(a)+list(a).index(0)], cd_extended[179-list(a).index(0)+3]]
        cd_interp = interpolate.interp1d(x_interp, y_interp, kind='linear')
        for i in range(179-len(a)+list(a).index(0)+1, 179-list(a).index(0)+3):
            alpha = a_extended[i]
            # Aproximation for good continuity
            cl_extended[i] = -cl_s*cl_adj + (alpha+a_s)/(a_l+a_s) * (cl_l+cl_s*cl_adj)
            cd_extended[i] = cd_interp(alpha)
            #cd_extended[i] = cd_l + (alpha-a_l)/(-a_s-a_l) * (cd_s-cd_l)

        return a_extended, cl_extended, cd_extended

    def generate_polars_Viterna(self, Re):
        ##
        # Description:
        #   Generates two 2D interpolation classes. One for cl and one for cd
        #       as function of Re and alpha using extended_polar_Viterna
        ##
        # Method Parameters:
        #   Re : Array
        #       Array of Reynolds numbers to sample
        ##

        # Reynolds Array
        self.Re = Re

        # Initialize arrays for better efficiency
        self.a_arr = np.zeros(360)
        self.cl_arr = np.zeros((len(self.Re), 360))
        self.cd_arr = np.zeros((len(self.Re), 360))

        # Calculate extended polar for all reynolds numbers
        for i in range(len(self.Re)):
            Reynolds = self.Re[i]
            self.a_arr, self.cl_arr[i,:], self.cd_arr[i,:] = self.extended_polar_Viterna(Reynolds)

        # Create polar interpolation
        self.cl = interpolate.interp2d(self.a_arr, self.Re, self.cl_arr, kind="linear")
        self.cd = interpolate.interp2d(self.a_arr, self.Re, self.cd_arr, kind="linear")

    def save_polars(self, filename):
        ##
        # Description:
        # Saves genetaded polars for future use
        ##
        # Method Parameters:
        #   filename : string
        #       Directory and filename of saving location
        ##

        with open(filename, "w") as file:
            # Write Reynolds Numbers
            file.write("Reynolds Numbers:\n")
            for i in range (len(self.Re)):
                file.write(str(self.Re[i]) + "\t")
            file.write("\n")

            # Write Polars
            file.write("Alpha [deg]\tCl\tCd\t...")

            # Lines
            for i in range(360):
                file.write("\n")
                file.write(str(self.a_arr[i])+"\t")

                # Columns
                for j in range(len(self.Re)):
                    file.write(str(self.cl_arr[j, i]) + "\t" + str(self.cd_arr[j, i]) + "\t")

    def load_polars(self, filename):
        ##
        # Description:
        # Loads previously saved polars
        ##
        # Method Parameters:
        #   filename : string
        #       Directory and filename of file to load
        ##

        # Open file and read
        with open(filename, "r") as file:
            # Skip first line
            file.readline()

            # Read Reynolds number array
            self.Re = np.array([float(x) for x in file.readline().split("\n")[0].split("\t")if x != ""])

            # Skip table header
            file.readline()

            # Read polars
            table = []
            for line in file:
                table.append([x for x in line.split("\n")[0].split("\t")if x != ""])

            # Initialize arrays of alpha, Cl and Cd for better efficiency
            self.a_arr = np.zeros(len(table))
            self.cl_arr = np.zeros((len(self.Re), len(table)))
            self.cd_arr = np.zeros((len(self.Re), len(table)))

            # Attribute values to correct variables
            for i in range(len(table)):
                self.a_arr[i] = table[i][0]
                for j in range(1, len(self.Re)*2+1, 2):
                    self.cl_arr[(j-1)//2, i] = table[i][j]
                    self.cd_arr[(j-1)//2, i] = table[i][j+1]

        # Create polar interpolation
        self.cl = interpolate.interp2d(self.a_arr, self.Re, self.cl_arr, kind="linear")
        self.cd = interpolate.interp2d(self.a_arr, self.Re, self.cd_arr, kind="linear")

    def polar_get(self, alpha, Re):
        ##
        # Description:
        #   Retrieves the Cl and Cd values for a givel alpha and Re
        ##
        # Method Parameters:
        #   alpha : float
        #       Angle of attack at which to sample
        #   Re : float
        #       Reynolds number at which to sample
        ##

        if (Re>self.Re[-1] or Re<self.Re[0]) and self.warnings:
            #sys.exit("Re = " + "{:.0f}".format(Re) + " outside of scope. Increase Reynolds analysis domain")
            print("Re = " + "{:.0f}".format(Re) + " outside of scope. Increase Reynolds analysis domain")

        return float(self.cl(alpha, Re)), float(self.cd(alpha, Re))

def fourier(t, y):
    A = abs(fft.fft(y))/len(y)*2

    # Normalize the amplitude by number of bins and multiply by 2
    # because we removed second half of spectrum above the Nyqist frequency
    # and energy must be preserved

    f = np.linspace(0,1/(t[1]-t[0]),len(y))

    PS = A[:int(len(A)/2)]
    f = f[:int(len(f)/2)]

    PSD = [x**2/(f[1]-f[0]) for x in PS]

    # PS = Power spectrum
    # PSD = Power spectral density
    return f, PS, PSD

################################################################################
