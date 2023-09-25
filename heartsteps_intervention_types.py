''''
How do the inactive and active bandits for HeartSteps behave if you took no/some/all intervention?
'''
from src.dataset import Heartstep
import numpy as np
from src.utils import reset_bandits, no_intervention_heartstep, all_intervention_heartstep, random_intervention_heartstep
import pickle as pkl

num_bandits_per_group = 10
A = 2; T = 100; S = 200; D_s = 5
all_bandits = []
test_bandits = []
test_groups = []
for j, group in enumerate(['active', 'inactive']):#, 'inactive', 'driver', 'ww']):
    for i in range(num_bandits_per_group):
        if group == 'active': sig=150; y0=89 + np.random.randn()*2 # So that we can get a spectrum of users
        elif group == 'inactive': sig=150; y0=70 + np.random.randn()*2
        elif group == 'driver': sig = 0.05
        elif group == 'ww': sig = 0.05
        bandit = Heartstep(T=T, patient_type=group, sig=sig, A=A, y0=y0)
        all_bandits.append(bandit)
        test_bandit = Heartstep(T=T, patient_type=group, sig=sig, A=A, y0=y0)
        test_bandits.append(test_bandit)
        test_groups.append(test_bandit.group)

date = "09072023"
all_bandits, test_bandits = reset_bandits(all_bandits, test_bandits)
results_no_intervention = no_intervention_heartstep(all_bandits, T)
pkl.dump(results_no_intervention, open("data/" + date + "no_intervention_fig3.pkl", 'wb'))
all_bandits, test_bandits = reset_bandits(all_bandits, test_bandits)
results_all_intervention = all_intervention_heartstep(all_bandits, T)
pkl.dump(results_all_intervention, open("data/" + date + "all_intervention_fig3.pkl", 'wb'))
all_bandits, test_bandits = reset_bandits(all_bandits, test_bandits)
results_random_intervention = random_intervention_heartstep(all_bandits, T)
pkl.dump(results_random_intervention, open("data/" + date + "random_intervention_fig3.pkl", 'wb'))