################################################################################
# This code generates a Kriging surrogate that estimates the overall sound
# pressure level produced by a range of propeller in a design space defined by
# the user.
################################################################################

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

import copy

import timeit
import time

from scipy import fft
#########
from pyDOE import * # type: ignore
from smt.sampling_methods import LHS # type: ignore
########

# Neto classes
from Functions.NoisePredictor import Propeller, Airflow, Polar, fourier  # type: ignore
from Functions.lhs import lhc_scale, jd, mphi  # type: ignore
from Functions.kriging import kriging_prediction, kriging_likelihood, corr_coef, rmse  # type: ignore

################################################################################
################################################################################
# Define surrogate variables
# Single number if fixed value
# List with min and max in case of range

BladeCount = [2, 6]
ChordRatio = [1/2, 2]
SpanRatio = [1/2, 2]
TwistDelta = [-10, 10]        # [Deg]

# Define observer position and velocity relative to the propeller [x,y,z]
# x: Horizontal in propeller plane, positive towards the right (when looking from the front)
# y: Vertical in propeller plane, positive upwards (when looking from the front)
# z: Along propeller axis, positive towards propeller wake
ObserverPos = [0,-10,0]
ObserverVel = [0,0,75]

# Number of query points in LHC
n_query = 10

# Number of query points in validation LHC
n_validation = n_query//4

if n_validation<3:
    n_validation=3

# Kriging basis function parameters. Should be left blank when optimization is desired. It is recommended to only optimize LogTheta
LogTheta = []   # type: list
p = [2,2,2,2]
reglambda = [0]

# Bounds for the kriging basis function parameters.
# Reasonable bounds are: (-3,2) for LogTheta; (1,3) for p and (10**-6,1) for reglambda
xbounds = ((-3,2),)*4

# Select Mode "Thrust" for fixed thrust or "RPM" for fixed rpm
mode = "Thrust"
modevalue = 1500            # N or rpm

################################################################################
# Number of spanwise blade elements
n_elements = 20

# Declare generic propeller
genericprop = Propeller("Propeller.txt", 2)
genericprop.bem_elements(n_elements)

# Declare Airflow
airflowname =  "Airflow" + "Wake" + ".txt"
airf = Airflow(airflowname, [0, 0], 1.225, 1.78938e-5, 340)

# Load Polars
for i in range(n_elements):
    name = "Polars/Polars" + str(i) + ".txt"
    genericprop.polarsBEM[i].load_polars(name)

################################################################################
################################################################################

if __name__ == "__main__":
    # Determine variables
    # LHC pos | gives the index of each variable in the LHC. -1 is default for non variables
    LHCpos = [-1]*4
    k = 0   # Number of variables

    x_min = []  # type: list            # Minimum value for each variable
    x_max = []  # type: list            # Maximum value for each variable

    if isinstance(BladeCount, list):
        LHCpos[0] = k
        x_min.append(BladeCount[0])
        x_max.append(BladeCount[1])
        k += 1
    if isinstance(ChordRatio, list):
        LHCpos[1] = k
        x_min.append(ChordRatio[0])
        x_max.append(ChordRatio[1])
        k += 1
    if isinstance(SpanRatio, list):
        LHCpos[2] = k
        x_min.append(SpanRatio[0])
        x_max.append(SpanRatio[1])
        k += 1
    if isinstance(TwistDelta, list):
        LHCpos[3] = k
        x_min.append(TwistDelta[0])
        x_max.append(TwistDelta[1])
        k += 1

    # Generate latin hypercube sampling points
    start = timeit.default_timer()      # Start timer
    LHC = np.array(LHS(xlimits=np.array([[0,1] for i in range(k)]), criterion="ese")(n_query))
    stop = timeit.default_timer()       # Stop timer
    print("LHC generation time: " + str(round(stop-start, 2)) + " s")

    # Correct LHC to scaled int blade count values
    if LHCpos[0] != -1:
        for i in range(n_query):
            LHC[i][0] = round(LHC[i][0] * (BladeCount[1]-BladeCount[0])) / (BladeCount[1]-BladeCount[0])

    print("LHC:")
    print(LHC)

    # Scale sampling hypercube to expensive function domain
    LHC_scaled = lhc_scale(LHC, x_min, x_max, [0]*k, [1]*k)
    start = timeit.default_timer()      # Start timer

    # Calculate OASPL in sampling points
    Y = np.zeros(n_query)   # Initialize Y array for better optimization

    for i in range(n_query):
        # Print new propeller
        print("#"*50)
        print("LHC, Propeller ", i+1, "/", n_query)

        # Generate propeller object
        prop = copy.deepcopy(genericprop)

        # Set number of blades
        if LHCpos[0] == -1:                 # If number of blades is a fixed value
            prop.B = int(BladeCount)
        else:                                # If number of blades is a variable
            prop.B = int(LHC_scaled[i][LHCpos[0]])


        # Set blade chord
        if LHCpos[1] == -1:                 # If blade chord is a fixed value
            for j in range(n_elements):
                prop.cBEM[j] = prop.cBEM[j] * ChordRatio
        else:                                # If blade chord is a variable
            for j in range(n_elements):
                prop.cBEM[j] = prop.cBEM[j] * LHC_scaled[i][LHCpos[1]]

        # Set blade span
        if LHCpos[2] == -1:                 # If blade span is a fixed value
            for j in range(n_elements):
                prop.rBEM[j] = prop.rBEM[j] * SpanRatio
        else:                                # If blade span is a variable
            for j in range(n_elements):
                prop.rBEM[j] = prop.rBEM[j] * LHC_scaled[i][LHCpos[2]]

        # Set blade twist
        if LHCpos[3] == -1:                 # If blade span is a fixed value
            for j in range(n_elements):
                prop.twistBEM[j] = prop.twistBEM[j] + TwistDelta
        else:                                # If blade span is a variable
            for j in range(n_elements):
                prop.twistBEM[j] = prop.twistBEM[j] + LHC_scaled[i][LHCpos[3]]

        # Calculate thrust and rpm
        if mode == "RPM":
            w = modevalue
            T = prop.avg_thrust(w, airf)

        elif mode == "Thrust":
            w = 1500

            while True:
                T = prop.avg_thrust(w, airf)

                print(f"(iteration) Thrust: {T:.0f} N\t\tRPM: {w:.0f}")

                if abs(modevalue-T)/modevalue < 0.001:
                    break

                w = abs(w + ((modevalue-T)/(modevalue+T)) * w * 0.6 * np.random.rand())
                if w < 10:
                    w = 2000

        print(f"Thrust: {T:.0f} N\t\tRPM: {w:.0f}")


        # Calculate noise samplint domain (1 propeller rotation)
        t = np.linspace(0, 60/w, 128)

        # Caculate OASPL
        Y[i] = prop.noise(T, t, airf, ObserverPos, ObserverVel, w)[1]

    stop = timeit.default_timer()       # Stop timer
    print("Expensive function compute time: " + str(round(stop-start, 2)) + " s")

    # Determine best basis function parameters
    print("#"*50)
    print("Starting optimization...")

    start = timeit.default_timer()      # Start timer

    optimal = optimize.differential_evolution(kriging_likelihood, bounds=xbounds, args=(LHC, Y, "optimization", LogTheta, p, reglambda), popsize=15, updating='deferred', workers=-1)

    aux = optimal.x

    # Atribute the values of theta, p and reglambda to the correct variables
    if LogTheta == []:
        LogTheta = aux[:k]

        if p == []:
            p = aux[k:2*k]

    elif p == []:
        p = aux[:k]

    if reglambda == []:
        reglambda = aux[-1]

    stop = timeit.default_timer()       # Stop timer
    print("Optimization time: " + str(round(stop-start, 2)) + " s")

    # Compute Cholesky factorization of correlation matrix
    U = kriging_likelihood([], LHC, Y, "cholcorrmatrix", LogTheta, p, reglambda)

    ############################################################################
    start = timeit.default_timer()       # Start timer
    # Validate model
    print("#"*50)
    print("#"*50)
    print("Model Validation")

    # Generate latin hypercube sampling points
    start = timeit.default_timer()      # Start timer
    LHC_validate = np.array(LHS(xlimits=np.array([[0,1] for i in range(k)]), criterion="ese")(n_validation))
    stop = timeit.default_timer()       # Stop timer
    print("LHC generation time: " + str(round(stop-start, 2)) + " s")

    # Correct LHC to scaled int blade count values
    if LHCpos[0] != -1:
        for i in range(n_validation ):
            LHC_validate[i][0] = round(LHC_validate[i][0] * (BladeCount[1]-BladeCount[0])) / (BladeCount[1]-BladeCount[0])

    # Scale sampling hypercube to expensive function domain
    LHC_validate_scaled = lhc_scale(LHC_validate, x_min, x_max, [0]*k, [1]*k)

    funcvalue = np.zeros([n_validation])
    funcprediction = np.zeros([n_validation])


    for i in range(n_validation):
        # Print new propeller
        print("#"*50)
        print("Validation, Propeller ", i+1, "/", n_validation)

        # Prediction
        funcprediction[i] = kriging_prediction(LHC_validate[i], LHC, Y, LogTheta, p, U)

        # Real value

        # Generate propeller object
        prop = copy.deepcopy(genericprop)

        # Set number of blades
        if LHCpos[0] == -1:                 # If number of blades is a fixed value
            prop.B = int(BladeCount)
        else:                                # If number of blades is a variable
            prop.B = int(LHC_validate_scaled[i][LHCpos[0]])

        # Set blade chord
        if LHCpos[1] == -1:                 # If blade chord is a fixed value
            for j in range(n_elements):
                prop.cBEM[j] = prop.cBEM[j] * ChordRatio
        else:                                # If blade chord is a variable
            for j in range(n_elements):
                prop.cBEM[j] = prop.cBEM[j] * LHC_validate_scaled[i][LHCpos[1]]

        # Set blade span
        if LHCpos[2] == -1:                 # If blade span is a fixed value
            for j in range(n_elements):
                prop.rBEM[j] = prop.rBEM[j] * SpanRatio
        else:                                # If blade span is a variable
            for j in range(n_elements):
                prop.rBEM[j] = prop.rBEM[j] * LHC_validate_scaled[i][LHCpos[2]]

        # Set blade twist
        if LHCpos[3] == -1:                 # If blade span is a fixed value
            for j in range(n_elements):
                prop.twistBEM[j] = prop.twistBEM[j] + TwistDelta
        else:                                # If blade span is a variable
            for j in range(n_elements):
                prop.twistBEM[j] = prop.twistBEM[j] + LHC_validate_scaled[i][LHCpos[3]]

        # Calculate thrust and rpm
        if mode == "RPM":
            w = modevalue
            T = prop.avg_thrust(w, airf)

        elif mode == "Thrust":
            w = 1500

            while True:
                T = prop.avg_thrust(w, airf)

                print(f"(iteration) Thrust: {T:.0f} N\t\tRPM: {w:.0f}")

                if abs(modevalue-T)/modevalue < 0.001:
                    break

                w = abs(w + ((modevalue-T)/(modevalue+T)) * w * 0.6 * np.random.rand())
                if w < 10:
                    w = 2000

        print(f"Thrust: {T:.0f} N\t\tRPM: {w:.0f}")


        # Calculate noise samplint domain (1 propeller rotation)
        t = np.linspace(0, 60/w, 128)

        # Caculate OASPL
        funcvalue[i] = prop.noise(T, t, airf, ObserverPos, ObserverVel, w)[1]

    rsquared = corr_coef(funcprediction, funcvalue)
    RMSE = rmse(funcprediction, funcvalue)
    NRMSE = RMSE/(max(funcvalue)-min(funcvalue))
    print(f"r^2 = {rsquared:.3f}")
    print(f"RMSE = {RMSE:.3f}")
    print(f"NRMSE = {NRMSE:.3f}")
    stop = timeit.default_timer()       # Stop timer
    print("Validation time: " + str(round(stop-start, 2)) + " s")
    ############################################################################

    # Write surrogate model data to file
    # Blade count
    if LHCpos[0] == -1:
        A = str(BladeCount)
    else:
        A = str(BladeCount[0])+ "to" + str(BladeCount[1])
    # Chord Ratio
    if LHCpos[1] == -1:
        B = str(ChordRatio)
    else:
        B = str(ChordRatio[0])+ "to" + str(ChordRatio[1])
    # SpanRatio
    if LHCpos[2] == -1:
        C = str(SpanRatio)
    else:
        C = str(SpanRatio[0])+ "to" + str(SpanRatio[1])
    # TwistDelta
    if LHCpos[3] == -1:
        D = str(TwistDelta)
    else:
        D = str(TwistDelta[0])+ "to" + str(TwistDelta[1])
    # ObserverPos
    E = str(ObserverPos[0]) + "," + str(ObserverPos[1]) + "," + str(ObserverPos[2])
    # ObserverVel
    F = str(ObserverVel[0]) + "," + str(ObserverVel[1]) + "," + str(ObserverVel[2])

    filename = "Surrogates/"+mode+str(modevalue)+"_"+time.strftime("%Y%m%d-%H%M%S")+"_"+A+"_"+B+"_"+C+"_"+D+"_"+E+"_"+F+".txt"

    with open(filename, "w") as file:

        file.write("Blade Count:\t")
        if LHCpos[0] == -1:
            file.write(str(BladeCount))
        else:
            file.write(str(BladeCount[0])+"\t"+str(BladeCount[1]))

        file.write("\nChord Ratio:\t")
        if LHCpos[1] == -1:
            file.write(str(ChordRatio))
        else:
            file.write(str(ChordRatio[0])+"\t"+str(ChordRatio[1]))

        file.write("\nSpan Ratio:\t")
        if LHCpos[2] == -1:
            file.write(str(SpanRatio))
        else:
            file.write(str(SpanRatio[0])+"\t"+str(SpanRatio[1]))

        file.write("\nTwist Delta [deg]:\t")
        if LHCpos[3] == -1:
            file.write(str(TwistDelta))
        else:
            file.write(str(TwistDelta[0])+"\t"+str(TwistDelta[1]))

        file.write("\nObserver Position:\t")
        file.write(str(ObserverPos[0])+"\t"+str(ObserverPos[1])+"\t"+str(ObserverPos[2]))

        file.write("\nObserver Velocity:\t")
        file.write(str(ObserverVel[0])+"\t"+str(ObserverVel[1])+"\t"+str(ObserverVel[2]))

        file.write("\n\nLogTheta:\t")
        for i in range(k):
            file.write(str(LogTheta[i])+"\t")

        file.write("\np:\t")
        for i in range(k):
            file.write(str(p[i])+"\t")

        file.write("\nRegression Lambda:\t")
        file.write(str(reglambda[0])+"\n")

        file.write("\nNumber of query points:\t")
        file.write(str(n_query))

        file.write("\n\nLHC:")
        for i in range(n_query):
            file.write("\n")
            for j in range(k):
                file.write(str(LHC[i][j])+"\t")

        file.write("\n\nY:\n")
        for i in range(n_query):
            file.write(str(Y[i])+"\n")

        file.write("\nU:")
        for i in range(n_query):
            file.write("\n")
            for j in range(n_query):
                file.write(str(U[i][j])+"\t")

        file.write("\n\nr^2:\t"+str(rsquared))
        file.write("\nRMSE:\t"+str(RMSE))
        file.write("\nNRMSE:\t"+str(NRMSE))
        file.write("\nAirflow:\t"+airflowname)
