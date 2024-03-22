"""

MEC8211 - Homework 3 - Group 4

"""

import numpy as np

def compute_gci_with_provided_p(f1, f2, p_hat, pf=2):
    r = 2
    p = p_hat if p_hat < pf else pf
    gci = abs(f2 - f1) / (r**(p) - 1) * (3)
    return gci

f1_values = [24.75619676816559, 27.2219521774702, 22.8593931172917, 29.76136640312751, 26.18068416463144]
f2_values = [24.87401289862732, 27.1319539088010, 22.96779398297336, 29.94377140828306, 25.75178760715440]
p_hat_values = [2.3220, 1.7994, 2.4481, 2.2340, 1.4635]

gci_results_with_p_hat = [compute_gci_with_provided_p(f1, f2, p_hat) for f1, f2, p_hat in zip(f1_values, f2_values, p_hat_values)]

for i, gci in enumerate(gci_results_with_p_hat, start=1):
    print(f"Case {i}: GCI = {gci}")
    
gci_mean = np.mean(gci_results_with_p_hat)
print(f"Mean GCI = {gci_mean}")
"0.24989212594045124"