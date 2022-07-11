import numpy as np
from scipy.stats import poisson


def sample_poisson_data(T, N1, N2, eps, mu):
    """
    sizes :N1: and :N2:

    
    Args:
    -----
    :T:    number of features
    :N1:   total in group 1 at t=0
    :N2:   total in group 2 at t=0
    :eps:  fraction of non-null events
    :mu:   intensity of non-null events

    """

    P = np.ones(T) / T  # `base` Poisson rates (does not have to be fixed)
    Q = P.copy()
    theta = np.random.rand(T) < eps
    Q[theta] = (np.sqrt(mu) + np.sqrt(P[theta])) ** 2   # perturbed Poisson rates

    O1 = poisson.rvs(N1 * P)
    O2 = poisson.rvs(N2 * Q)

    return O1, O2
