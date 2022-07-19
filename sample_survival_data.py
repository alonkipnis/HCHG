import numpy as np
from scipy.stats import poisson


def sample_survival_data(T, N1, N2, lam0, eps, r):
    """
    Sample :T: times from two survival populations with 
    initial sizes :N1: and :N2:
    In each time t and group j, the reduction `Oj`
    is a Possion RV. Usually, the Poisson rates
    are identical. Although in :eps: fraction the times the 
    Poisson rate of O2 are elevated by :mu:
    
    Args:
    -----
    :T:    number of events
    :N1:   total in group 1 at t=0
    :N2:   total in group 2 at t=0
    :eps:  fraction of non-null events
    :lam0: base Poisson rate
    :r:   intensity of non-null events
    
    Note that since we sample from two Poisson distributions 
    in each 'event', there is some possibility that we draw (O1,O2) = (0,0),
    hence there is no change in that event. This situation is different 
    than standard notation. 
    
    """

    Nt1 = np.zeros(T+1)
    Nt2 = np.zeros(T+1)

    lam1 = lam0  # `base` Poisson rates (does not have to be fixed)
    theta = np.random.rand(T) < eps

    #lam2 = lam1.copy()
    #lam2[theta] = (np.sqrt(mu) + np.sqrt(lam1[theta])) ** 2   # perturbed Poisson rates

    Nt1[0] = N1
    Nt2[0] = N2

    for t in np.arange(T):
        N0 = 2 * Nt1[0] * Nt2[0] / (Nt1[0] + Nt2[0])
        lam2 = (np.sqrt(lam1[t]) + theta[t] * np.sqrt( r * np.log(T) / 2 / N0) ) ** 2

        O1 = poisson.rvs(Nt1[t] * lam1[t] * (Nt1[t] > 0))
        O2 = poisson.rvs(Nt2[t] * lam2 * (Nt2[t] > 0))

        Nt1[t+1] = np.maximum(Nt1[t] - O1, 0)
        Nt2[t+1] = np.maximum(Nt2[t] - O2, 0)
    return Nt1, Nt2
