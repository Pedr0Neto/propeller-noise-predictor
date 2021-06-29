################################################################################
# This code takes the Kriging surrogate created in GenerateKriging.py and
# estimates the function values in the desired locations (or range of locations)
################################################################################

from scipy import interpolate  # type: ignore
from scipy import optimize
from scipy import integrate

import numpy as np
import math
import random
import sys

import timeit
import time

from scipy import fft

import matplotlib.pylab as plt  # type: ignore

#########
from pyDOE import * # type: ignore
from smt.sampling_methods import LHS # type: ignore
########

# Neto classes
from Functions.NoisePredictor import Propeller, Airflow, Polar, fourier  # type: ignore
from Functions.lhs import lhc_scale, jd, mphi  # type: ignore
from Functions.kriging import kriging_prediction, kriging_likelihood, corr_coef  # type: ignore

################################################################################
################################################################################
# Filename of the surrogate
filename = "Surrogates/Thrust1500_20210621-221025_2to6_0.5to2_0.5to2_-10to10_0,-10,0_0,0,75.txt"

################################################################################
################################################################################

# Read Data File
with open(filename, "r") as file:
    # Read Blade Count
    BladeCount = [int(x) for x in file.readline().split("\n")[0].split("\t")[1:]]

    # Read Chord Ratio
    ChordRatio = [float(x) for x in file.readline().split("\n")[0].split("\t")[1:]]

    # Read Span Ratio
    SpanRatio = [float(x) for x in file.readline().split("\n")[0].split("\t")[1:]]

    # Read Twist Delta [deg]
    TwistDelta = [float(x) for x in file.readline().split("\n")[0].split("\t")[1:]]

    # Read Observer Position
    ObserverPos = [float(x) for x in file.readline().split("\n")[0].split("\t")[1:]]

    # Read Observer Velocity
    ObserverVel = [float(x) for x in file.readline().split("\n")[0].split("\t")[1:]]

    # Read empty line
    file.readline()

    # Read Log Theta
    LogTheta = [float(x) for x in file.readline().split("\n")[0].split("\t")[1:-1]]

    # Read p
    p = [float(x) for x in file.readline().split("\n")[0].split("\t")[1:-1]]

    # Read Regression Lambda
    reglambda = [float(x) for x in file.readline().split("\n")[0].split("\t")[1:]]

    # Read empty line
    file.readline()

    # Read number of query points
    n_query = int(file.readline().split("\n")[0].split("\t")[1])

    # Read empty lines
    file.readline()
    file.readline()

    # Read LHC
    LHC = np.zeros([n_query, len(p)])
    for i in range(n_query):
        LHC[i,:] = [float(x) for x in file.readline().split("\n")[0].split("\t")[:-1]]

    # Read empty lines
    file.readline()
    file.readline()

    # Read Y
    Y = np.zeros(n_query)
    for i in range(n_query):
        Y[i] = float(file.readline().split("\n")[0])

    # Read empty lines
    file.readline()
    file.readline()

    # Read U
    U = np.zeros([n_query, n_query])
    for i in range(n_query):
        U[i,:] = [float(x) for x in file.readline().split("\n")[0].split("\t")[:-1]]


# 2D
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)

f = np.zeros([50,50])

for i in range(len(x)):
    for j in range(len(y)):
        f[i,j] = kriging_prediction([0, 1/3, x[i], y[j]], LHC, Y, LogTheta, p, U)

x = [a*1.5+0.5 for a in x]
y = [a*20-10 for a in y]

plt.contourf(x, y, np.transpose(f), 20)
plt.rcParams.update({'font.size': 13})
plt.xlabel("Radius ratio", size = 13)
plt.ylabel("Twist increment [deg]", size = 13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.colorbar().set_label("OASPL [dB]", rotation=270)
plt.tight_layout()
plt.savefig("Figure1.png")


"""
# Number of blades
x = np.linspace(0, 1, 50)
f = np.zeros([5,50])
for i in range(5):
    b = i/4
    for j in range(len(x)):
        f[i,j] = kriging_prediction([b, 1/3, x[j], 0], LHC, Y, LogTheta, p, U)

x = [a*1.5+0.5 for a in x]
plt.plot(x, f[0,:], "-")
plt.plot(x, f[1,:], "-")
plt.plot(x, f[2,:], "-")
plt.plot(x, f[3,:], "-")
plt.plot(x, f[4,:], "-")
font = 13
plt.legend(["B=2", "B=3", "B=4", "B=5", "B=6"], loc ="upper left", prop={'size': font})
plt.xlabel("Radius ratio", size = font)
plt.ylabel("OASPL [dB]", size = font)
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.savefig("Figure2.png")
"""
