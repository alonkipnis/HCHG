
import numpy as np
import matplotlib as mpl
from illustrate_survival_curve import illustrate_survival_curve
from phase_transition_experiment.sample_survival_poisson import sample_survival_poisson_censorship


output_figure_path = 'survival_curve_example_with_censoring.png'
np.random.seed(12)

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] =  [10, 5]
mpl.style.use('ggplot')



T = 84
N1 = 1000
N2 = 1000
eps = 0.05
lam_bar = 0.6
lam0 = lam_bar / T * np.ones(T)
lamc = 0.2
r = 1.2
Nt1, Nt2, Ot1, Ot2 = sample_survival_poisson_censorship(T, N1, N2, lam0, lamc, eps, r)
Ct1 = -np.append(np.diff(Nt1), 0) - Ot1
Ct2 = -np.append(np.diff(Nt2), 0) - Ot2

df = pd.DataFrame({
    'at_risk:0': Nt1,
    'at_risk:1': Nt2,
    'events1': Ot1,
    'events2': Ot2
})

df_HCT, df_all = illustrate_survival_curve(Nt1, Nt2, Ot1, Ot2, Ct1, Ct2, stbl=True)

plt.xlim((0,T))
plt.ylim((0.45,1))
#plt.xtickslabel(fontsize=16)
plt.ylabel("Survival Proportion", fontsize=16)
plt.savefig(output_figure_path)
plt.show()


df_all['pvalue'] = np.round(df_all['pvalue'], 4)
df_all['at_risk:0'] = df_all['at_risk:0'].astype(int)
df_all['at_risk:1'] = df_all['at_risk:1'].astype(int)
df_all['events1'] = df_all['events1'].astype(int)
df_all['events2'] = df_all['events2'].astype(int)
a = df_all.reset_index().filter(['index', 'at_risk:0', 'at_risk:1', 'events1', 'events2', 'pvalue']).rename(columns={'index':'$t-1$','at_risk:0': '$n_x(t-1)$',
                                     'at_risk:1': '$n_y(t-1)$',
                                     'events1': '$o_x(t)$',
                                    'events2': '$o_y(t)$', 
                                    'pvalue': '$p_t$'
                                           }).to_latex(index=False, bold_rows=True,float_format="%.4f")

def highlight_line(a, i):
       # get the i-th line in a:
       num_header_lines = a.split('\n').index(r"\midrule")+1
       l = a.split('\n')[i+num_header_lines]
       new_l = '\\rowcolor{lightgray} ' +  l
       return a.replace(l, new_l)

prev = 5
num_removed = 0
for ii in df_HCT.index:
    i = ii - num_removed
    a = highlight_line(a, i)
    prev = i

print(a)