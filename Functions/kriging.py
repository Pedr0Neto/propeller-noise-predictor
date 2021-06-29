from pyDOE import * #https://pythonhosted.org/pyDOE/randomized.html
import math

from scipy import linalg
from scipy import optimize

import numpy as np

import timeit

def kriging_prediction(x, lhc, Y, LnTheta, p, U):
    # Calculates a Kriging prediction at x
    ##
    # Inputs:
    # x – 1 x k vetor of design variables
    # lhc – n x k matrix of sample locations
    # Y – n x 1 vector of observed data
    # LnTheta – 1 x k vector of log(theta)
    # U – n x n Cholesky factorization of Psi
    ##
    # Outputs:
    # f – scalar Kriging prediction

    theta = [10**i for i in LnTheta]

    # Calculate number of sample points
    n = lhc.shape[0]

    # Vector of ones
    one = np.ones(n)

    # Calculate mu
    a = np.linalg.lstsq(np.transpose(U), Y, rcond=None)[0]
    a = np.linalg.lstsq(U, a, rcond=None)[0]
    a = np.matmul(np.transpose(one), a)

    b = np.linalg.lstsq(np.transpose(U), one, rcond=None)[0]
    b = np.linalg.lstsq(U, b, rcond=None)[0]
    b = np.matmul(np.transpose(one), b)

    mu = a/b

    # Initialize psi to vector of ones
    Psi = np.ones(n)

    # Fill psi vector
    for i in range(n):
        Psi[i]=math.exp(-sum(np.multiply(theta, np.power(abs(np.array(lhc[i])-np.array(x)), p))))

    # Calculate prediction
    a = Y-one*mu
    a = np.linalg.lstsq(np.transpose(U), a, rcond=None)[0]
    a = np.linalg.lstsq(U, a, rcond=None)[0]

    f = mu + np.matmul(np.transpose(Psi), a)

    return f

def kriging_likelihood(x, lhc, Y, mode, LnTheta=[], p=[], reglambda=[]):
    ##
    # Parameters:
    #   x : Array
    #       Array of log(theta), p and reglambda parameters. If any of these is
    #       stipulated, it should be ommited from this array
    #
    #   lhc : Array
    #       n x k matrix of sample locations (n = number of sample locations,
    #       k = number of dimensions)
    #
    #   Y : Array
    #       n x 1 vector of observed data
    #
    #   mode : String
    #       Mode of operation of the function:
    #           - "optimization" : returns NegLnLike, used for optimization;
    #           - "cholcorrmatrix" Cholesky factorization of correlation matrix,
    #               used for the kriging prediction
    #
    #   LnTheta : Array, optional
    #       k x 1 matrix with the values of LnTheta. Must only be used when
    #           LnTheta is fixed. LnTheta should be ommited from "x" when this
    #           parameter is used.
    #
    #   p : Array, optional
    #       k x 1 matrix with the values of p. Must only be used when p is
    #       fixed. p should be ommited from "x" when this parameter is used.
    #
    #   reglambda : Float, optional
    #       Value of reglambda. If set to 0, kringing interpolation will be used.
    #           Other values will cause a degree of regression. Should be left
    #           blank if optimal regression is desired
    #
    #       fixed. p should be ommited from "x" when this parameter is used.
    ##
    # Returns:
    #   NegLnLike : float   (mode: "optimization")
    #       Concentrated ln – likelihood ∗−1. Should be minimized to optimize
    #           the variables in x
    #   U : Array   (mode: "cholcorrmatrix")
    #       Cholesky factorization of correlation matrix. Is used in the
    #       kriging prediction
    ##

    # Determine number of sample points
    n = lhc.shape[0]

    # Determine number of variables
    k = lhc.shape[1]

    # Atribute the values of theta, p and reglambda to the correct variables
    if LnTheta == []:
        theta = [10**x for x in x[:k]]

        if p == []:
            p = x[k:2*k]

    else:
        theta = [10**x for x in LnTheta]

        if p == []:
            p = x[:k]

    if reglambda == []:
        reglambda = x[-1]


    # n x 1 array of ones
    one = np.ones(n)

    # Pre-allocate memory
    Psi = np.zeros([n, n])

    # Build upper half of correlation matrix
    for i in range(n):
        for j in range(i+1, n):
            Psi[i,j] = math.exp(-sum(theta*np.power((abs(lhc[i,:] - lhc[j,:])), p)))

    # Add upper and lower halves and diagonal of ones plus small number to reduce ill conditioning
    Psi = Psi + np.matrix.transpose(Psi) + np.identity(n)*(1+np.finfo(float).eps+reglambda)

    # Cholesky factorization
    try:
        U = linalg.cholesky(Psi)

        # Sum lns of diagonal to find ln(det(Psi))
        LnDetPsi = 2*sum([math.log(i) for i in abs(np.diag(U))])

        # Use back–substitution of Cholesky instead of inverse
        # Calculate mu
        a = np.linalg.lstsq(np.transpose(U), Y, rcond=None)[0]
        a = np.linalg.lstsq(U, a, rcond=None)[0]
        a = np.matmul(np.transpose(one), a)

        b = np.linalg.lstsq(np.transpose(U), one, rcond=None)[0]
        b = np.linalg.lstsq(U, b, rcond=None)[0]
        b = np.matmul(np.transpose(one), b)

        mu = a/b

        # Calculate SigmaSqr
        a = np.transpose(Y-one*mu)

        b = Y-one*mu
        b = np.linalg.lstsq(np.transpose(U), b, rcond=None)[0]
        b = np.linalg.lstsq(U, b, rcond=None)[0]

        SigmaSqr = np.matmul(a, b)/n

        # Calculate NegLnLike
        NegLnLike = -1*(-(n/2)*np.log(SigmaSqr)-0.5*LnDetPsi)

    # In case of the matrix not being positive definite:
    except np.linalg.LinAlgError as err:
        NegLnLike = 1E100

    # Determine type of return
    if mode == "optimization":
        return NegLnLike
    elif mode == "cholcorrmatrix":
        return U

def corr_coef(ymeta, yreal):
    # Determine number of sample points
    n = ymeta.shape[0]

    ymeta=np.array(ymeta)
    yreal=np.array(yreal)

    # Compute r^2
    # Numerator
    a = sum(yreal*ymeta)
    b = sum(yreal)
    c = sum(ymeta)
    #Denominator
    d = sum(yreal**2)
    e = sum(ymeta**2)

    f = ((n*a-b*c)/np.sqrt((n*d-b**2)*(n*e-c**2)))**2

    return f

def rmse(ymeta, yreal):
    # Determine number of sample points
    n = ymeta.shape[0]

    return np.sqrt(sum((yreal-ymeta)**2)/n)
