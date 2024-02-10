import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
konp_test = importr('KONPsurv')

def konp_testR(times, status, groups):
    res = konp_test.konp_test(times, status, groups, n_perm=1)
    return dict(chisq_test_stat=res[3][0], lr_test_stat=res[4][0],cauchy_test_stat=res[5][0])


def time2event(times, num_events, num_censored):
    """
    Convert number of events and censored to time-to-event data.

    The number of events and censored at each time point is converted to a list of event times and event status.
    All events are assumed to occur at the same time as the time point.
    Events are coded as 1 and censored as 0.
    All at risk at the last time point are assumed to be censored.
    """
    assert len(times) == len(num_events) == len(num_censored) 

    r_times = []
    r_status = []
    for i,t in enumerate(times):
        ne = int(num_events[i])
        nc = int(num_censored[i])
        r_times += ([t] * (ne + nc))
        r_status += ([1] * ne + [0] * nc)
    
    return r_times, r_status

def evaluate_test_stats_konp(Ot1, Ot2, Ct1, Ct2):
    assert len(Ot1) == len(Ot2) == len(Ct1) == len(Ct2)
    T = len(Ot1)
    times1, status1 = time2event(np.arange(1,T+1), Ot1, Ct1)
    times2, status2 = time2event(np.arange(1,T+1), Ot2, Ct2)
    times = FloatVector(times1 + times2)
    status = FloatVector(status1 + status2)
    groups = FloatVector([0] * len(times1) + [1] * len(times2))
    return konp_testR(times, status, groups)
