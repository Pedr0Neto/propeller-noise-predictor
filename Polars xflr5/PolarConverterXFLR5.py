################################################################################
# This code takes airfoil data in xfoil/sflr5 format and converts it to a format
# compatible with the code.

# This conversion also extends the airfoil polar to 360 degrees.

# The polars must be available at every blade element
################################################################################

import numpy as np

from scipy import interpolate  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import math

Re = [20000, 100000, 200000, 350000, 550000, 800000, 1000000, 1300000, 1600000]

Import = [
"NACA 0010_T1_Re0.020_M0.00_N9.0.txt",
"NACA 0010_T1_Re0.100_M0.00_N9.0.txt",
"NACA 0010_T1_Re0.200_M0.00_N9.0.txt",
"NACA 0010_T1_Re0.350_M0.00_N9.0.txt",
"NACA 0010_T1_Re0.550_M0.00_N9.0.txt",
"NACA 0010_T1_Re0.800_M0.00_N9.0.txt",
"NACA 0010_T1_Re1.000_M0.00_N9.0.txt",
"NACA 0010_T1_Re1.300_M0.00_N9.0.txt",
"NACA 0010_T1_Re1.600_M0.00_N9.0.txt",
]


for f in range(40):
    """
    print("Foil:", f)

    if f<10:
        index = "0" + str(f)
    else:
        index = str(f)


    Import = [
    index + "_T1_Re0.350_M0.00_N9.0.txt",
    index + "_T1_Re0.550_M0.00_N9.0.txt",
    index + "_T1_Re0.800_M0.00_N9.0.txt",
    index + "_T1_Re1.000_M0.00_N9.0.txt",
    index + "_T1_Re1.300_M0.00_N9.0.txt",
    index + "_T1_Re1.600_M0.00_N9.0.txt"
    ]
    """

    Export = "Polars" + str(f) + ".txt"

    ExportTable = np.zeros((360, len(Re)*2+1))

    # Import data and generate extended polars
    for k in range(len(Import)):

        with open(Import[k], "r") as file:
            # Skip first 11 lines
            for i in range(11):
                file.readline()

            # Read data
            table = []
            for line in file:
                table.append([float(x) for x in line.split("\n")[0].split(" ")if x != ""])
                #Remove line in case it's empty
                if table[-1] == []:
                    table.pop(-1)

        # Imported data
        a_import = [x[0] for x in table]
        cl_import = [x[1] for x in table]
        cd_import = [x[2] for x in table]

        # Create array of integer alphas in range
        a = np.linspace(int(a_import[0]), int(a_import[-1]), int(a_import[-1])-int(a_import[0])+1)

        # Interpolate cl and cd in the correct alphas
        cl = interpolate.interp1d(a_import, cl_import, kind="linear")(a)
        cd = interpolate.interp1d(a_import, cd_import, kind="linear")(a)

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

        # Viterna method - inspired by https://github.com/WISDEM/AirfoilPreppy/blob/master/airfoilprep/airfoilprep.py

        # lift coefficient adjustment to account for assymetry
        cl_adj = 0.7

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
        a_extended = np.array([i-179 for i in range(len(a_extended))])


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
        #x_interp = [a_extended[179-len(a)+list(a).index(0)], a_extended[179-list(a).index(0)+3]]

        #ycl_interp = [cl_extended[179-len(a)+list(a).index(0)], cl_extended[179-list(a).index(0)+3]]
        #ycd_interp = [cd_extended[179-len(a)+list(a).index(0)], cd_extended[179-list(a).index(0)+3]]

        #cl_interp = interpolate.interp1d(x_interp, ycl_interp, kind='linear')
        #cd_interp = interpolate.interp1d(x_interp, ycd_interp, kind='linear')
        for i in range(179-len(a)+list(a).index(0)+1, 179-list(a).index(0)):
            alpha = a_extended[i]
            # Aproximation for good continuity
            cl_extended[i] = -cl_s*cl_adj + (alpha+a_s)/(a_l+a_s) * (cl_l+cl_s*cl_adj)
            #cl_extended[i] = cl_interp(alpha)
            #cd_extended[i] = cd_interp(alpha)
            cd_extended[i] = cd_l + (alpha-a_l)/(-a_s-a_l) * (cd_s-cd_l)

        ExportTable[:,0] = a_extended
        ExportTable[:,k*2+1] = cl_extended
        ExportTable[:,k*2+2] = cd_extended

    # Export data
    with open(Export, "w") as file:
        # Write Reynolds Numbers
        file.write("Reynolds Numbers:\n")
        for i in range(len(Re)):
            file.write(str(Re[i]) + "\t")
        file.write("\n")

        # Write Polars
        file.write("Alpha [deg]\tCl\tCd\t...")

        # Lines
        for i in range(360):
            file.write("\n")
            for j in range(np.shape(ExportTable)[1]):
                file.write(str(ExportTable[i,j])+"\t")
