import pickle as pkl
import numpy as np
from src.dataset import get_DKT_CMAB
from src.utils import reset_education_bandits
# Run the Education CMAB using the Hybrid OD setup

def run_OD_education(all_bandits, S, K, D_s, T, A):
    # Generate train and test bandits
    N = len(all_bandits)
    D = D_s
    inst_regret = {
        'TS_hybrid_y': np.zeros((S, T, N)),
        'TS_hybrid_r': np.zeros((S, T, N)),
        'OD': np.zeros((S, T, N))
    }
    simple_regret = {
        'TS_hybrid_y': np.zeros((S, int(T * N))),
        'TS_hybrid_r': np.zeros((S, int(T * N))),
        'OD': np.zeros((S, int(T * N)))
    }
    for s in range(S):
        # Reset all bandits
        all_bandits = reset_education_bandits(all_bandits)
        pi_hybrid_ts_y = Hybrid(K=K, D=D)
        pi_hybrid_ts_r = Hybrid(K=K, D=D)
        pi_od = Hybrid(K=1, D=D)
        for kk, t in enumerate(tqdm.tqdm(range(T))):
            for i in range(N):
                bandit = all_bandits[i]

                # Find the optimal action, or set of optimal actions
                opt_a, r_star = find_opt_a(bandit=bandit)

                # Choose optimal action w.r.t TS optimizing for outcome (highest probability answer)
                Phi_train = np.asarray([phi_sa_education(a=a, s=bandit.state) for a in
                                        range(A)])
                a_it = pi_hybrid_ts_y.choose_ts_education(Phi_train, choose_outcome=True)
                ns, r_it, _, _ = bandit.step(a_it)
                y_it = ns[:51]
                bandit.state = ns
                phi_it = phi_sa_education(a_it, bandit.state)
                pi_hybrid_ts_y.update_individual(phi_it, y_it)
                inst_regret['TS_hybrid_y'][s, t, i] = (r_star - r_it)

    return inst_regret, simple_regret
bandits = [] # Each of them has a different curriculum
graphs = [8, 30, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
graph_dir = "/Users/amandyam/Documents/Research/HeartSteps/CSRL/environments_and_constraints/education/graph_structures/"
saved_data_folder = '/Users/amandyam/Documents/Research/HeartSteps/CSRL/environments_and_constraints/education/saved_data'
THRESH = 0.85
H = 300
ws = 10
for i in range(10):
    fname = graph_dir + "prereq_graph_C" + str(graphs[i]) + ".json"
    with open(fname, 'r') as f:
        prereq_dict = json.load(f)
    bandit = get_DKT_CMAB(prereq_graph=prereq_dict, saved_data_folder=saved_data_folder, H=H)
    bandits.append(bandit)
inst_regret, _ = run_OD_education(bandits, S=1, K=51, D_s=102, T=100, A=51)
pkl.dump(inst_regret, open("data/inst_regret_education.pkl", 'wb'))
