import numpy as np

def lhc_scale(lhc, x_min_out, x_max_out, x_min_in, x_max_in):
    # Define scaled array
    lhc_scaled = np.zeros([lhc.shape[0], lhc.shape[1]])

    for i in range(lhc.shape[0]):
        for j in range(lhc.shape[1]):
            lhc_scaled[i,j] = lhc[i,j] * (x_max_out[j]-x_min_out[j])/(x_max_in[j]-x_min_in[j]) + x_min_out[j] - x_min_in[j]/(x_max_in[j]-x_min_in[j])

    return lhc_scaled

def jd(X,p):
    # Computes the distances between all pairs of points in a sampling
    # plan X using the p–norm, sorts them in ascending order and
    # removes multiple occurrences.
    ##
    # Inputs:
    # X – sampling plan being evaluated
    # p – distance norm (p=1 rectangular – default, p=2 Euclidean)
    ##
    # Outputs:
    # J – multiplicity array (that is, the number of pairs
    # separated by each distance value).
    # distinct_d – list of distinct distance values

    # Number of points in the sampling plan
    n = X.shape[0]

    # Compute the distances between all pairs of points
    d = np.zeros(int(n*(n-1)/2))

    for i in range(n-1):
        for j in range(i+1, n):
            # Distance metric: p–norm
            d[int(i*n-i*(i+1)/2+j-i-1)] = np.linalg.norm(X[i, :]-X[j, :], p)

    # Remove multiple occurrences
    distinct_d = np.unique(d)

    # Pre-allocate memory for J
    J = np.zeros(len(distinct_d))

    # Generate multiplicity array
    for i in range(len(distinct_d)):
        # J(i) will contain the number of pairs separated by the distance distinct_d(i)
        J[i] = list(d).count(distinct_d[i])

    return J, distinct_d

def mphi(X, q, p):
    # Calculates the sampling plan quality criterion of Morris and
    # Mitchell.
    ##
    # Inputs:
    # X – sampling plan
    # q – exponent used in the calculation of the metric
    # p – the distance metric to be used (p=1 rectangular –
    # (default, p=2 Euclidean)
    ##
    # Output:
    # Phiq – sampling plan ‘space–fillingness’ metric

    # Calculate the distances between all pairs of points (using the p–norm) and build multiplicity array J
    [J, d] = jd(X, p)

    # The sampling plan quality criterion
    Phiq = sum([a*b**(-q) for a,b in zip(J,d)])**(1/q)

    return Phiq
