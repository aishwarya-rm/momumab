import numpy as np

def oracle(bandit, t):
    r0 = bandit.pull(t, int(0))
    r1 = bandit.pull(t, int(1))
    return max(r0, r1)

def phi_sa(a, group, n_notif, prev_y, T=200):
    if group == 0:
        inv_sigmoid = lambda x: np.exp(-x) / (1 + np.exp(-x))
        map_t = lambda x: (x - T / 1.9) * (10 / T)
    else:
        inv_sigmoid = lambda x: np.exp(-x) / (1 + np.exp(-x))
        map_t = lambda x: (x - T / 1.3) * (4/ T)
    s = inv_sigmoid(map_t(n_notif))
    # group_te = [0, 0, 0, 0]
    group_te = [0, 0]
    group_te[group] = s * a
    phi = [1, prev_y] + group_te + [n_notif]# + np.random.randn(3).tolist()# There are 10 elements to the state, last three are junk
    return np.asarray(phi)

def u1(y, phi_it, group):
    if group == 0:
        intercept = 0.42
        goal = np.sqrt(10000)
    elif group == 1:
        goal = np.sqrt(5600)
        intercept = 0.3
    if y < goal:
        return 0.005 * y
    else:
        return 0.001 * y + intercept
def u2(y, phi_it, group):
    num_notifications = phi_it[-1] # TODO change if we add junk params back
    return -(num_notifications ** 2) * 0.00003  # *num_notifications

def reset_bandits(all_bandits, test_bandits):
    new_bandits = []
    new_test = []
    for b in all_bandits:
        b.reset()
        new_bandits.append(b)
    for t in test_bandits:
        t.reset()
        new_test.append(t)
    return new_bandits, new_test

def no_intervention_heartstep(all_bandits, T=200):
    R_active = []
    Y_active = []
    R_inactive = []
    Y_inactive = []
    for b in all_bandits:
        r_b = []
        y_b = []
        for t in range(T):
            noise = np.random.randn() * b.sig
            r, y = b.pull(a=0, n_notif=b.ts_r_notifications, noise=noise, prev_y=b.yprev_ts_r)
            b.ts_r_notifications += 0
            r_b.append(r)
            y_b.append(y)
            b.yprev_ts_r = y
        if b.patient_type == 'active':
            R_active.append(r_b)
            Y_active.append(y_b)
        else:
            R_inactive.append(r_b)
            Y_inactive.append(y_b)
    results = {"R_active": R_active, "R_inactive": R_inactive, "Y_active": Y_active, "Y_inactive": Y_inactive}
    return results

def all_intervention_heartstep(all_bandits, T=200):
    R_active = []
    Y_active = []
    R_inactive = []
    Y_inactive = []
    for b in all_bandits:
        r_b = []
        y_b = []
        for t in range(T):
            noise = np.random.randn() * b.sig
            r, y = b.pull(a=1, n_notif=b.ts_r_notifications, noise=noise, prev_y=b.yprev_ts_r)
            b.ts_r_notifications += 1
            r_b.append(r)
            y_b.append(y)
            b.yprev_ts_r = y
        if b.patient_type == 'active':
            R_active.append(r_b)
            Y_active.append(y_b)
        else:
            R_inactive.append(r_b)
            Y_inactive.append(y_b)
    results = {"R_active": R_active, "R_inactive": R_inactive, "Y_active": Y_active, "Y_inactive": Y_inactive}
    return results

def random_intervention_heartstep(all_bandits, T=200):
    R_active = []
    Y_active = []
    R_inactive = []
    Y_inactive = []
    for b in all_bandits:
        r_b = []
        y_b = []
        for t in range(T):
            a = np.random.choice([0, 1])
            noise = np.random.randn() * b.sig
            r, y = b.pull(a=a, n_notif=0, noise=noise, prev_y=b.yprev_ts_r)
            b.ts_r_notifications += a
            r_b.append(r)
            y_b.append(y)
            b.yprev_ts_r = y
        if b.patient_type == 'active':
            R_active.append(r_b)
            Y_active.append(y_b)
        else:
            R_inactive.append(r_b)
            Y_inactive.append(y_b)
    results = {"R_active": R_active, "R_inactive": R_inactive, "Y_active": Y_active, "Y_inactive": Y_inactive}
    return results

def reset_education_bandits(all_bandits):
    for b in all_bandits:
        b.reset()
    return all_bandits