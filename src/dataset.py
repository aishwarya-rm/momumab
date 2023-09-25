import numpy as np
import json

class Dataset():
    def __init__(self, T):
        pass
    def pull(self, a):
        pass
    def reset(self):
        pass

class Heartstep(Dataset):
    def __init__(self, T, patient_type, sig, A, y0):
        '''
        :param T: number of timesteps
        :param sig: noise
        :param patient_type: ['active', 'inactive', 'driver', 'ww']
        '''
        # Generate patient states
        self.group = patient_type
        self.A = A
        self.patient_type = patient_type
        self.sig = sig
        self.T = T
        self.y0 = y0
        base_W = np.asarray([[0.6, 0.4], [0.4, 0.6], [0.6, 0.4], [0.6, 0.4]] ) # Weights must add up to 1
        # T,
        #phi_it = [t, indicators ofr each group, # notifications, a_it, indicator for ww and weekend]
        # groups are active, inactive, driver, ww
        c = 0.001
        # self.Theta = [-0.02] + [1-c] + [0.3, 0.21, 0.1, 0.08] + [0] #+ [0, 0, 0]
        self.Theta = [-0.02] + [1-c] + [0.3, 0.21] + [0]
        # self.Theta = [-0.3, 80, 70, 60, 80, 0.32, 10, 20, 1]
        # self.Theta = [-0.1, 100, 70, 60, 80, 0.15, 3, 60]

        self.od_notifications = 0
        self.ts_y_notifications = 0
        self.ts_r_notifications = 0
        self.random_notifications = 0

        self.yprev_od = y0
        self.yprev_ts_y = y0
        self.yprev_ts_r = y0
        self.yprev_random = y0
        self.yprev_baseline = y0

        if patient_type == 'active':
            self.weights = base_W[0]
            self.group = 0
        elif patient_type == 'inactive':
            self.weights = base_W[1]
            self.group = 1
        elif patient_type == 'driver':
            self.weights = base_W[2]
            self.group = 2
        elif patient_type == 'ww':
            self.weights = base_W[3]
            self.group = 3

        self.utilities = [u1, u2]
        self.sig = sig

    def pull(self, a, n_notif, prev_y, noise):
        p_sa = phi_sa(a=a, group=self.group, n_notif=n_notif, prev_y=prev_y)
        # Calculate the outcome
        y_it = np.dot(p_sa, self.Theta) + 0.001 * noise
        # Calculate the reward
        r = 0
        for i in range(self.weights.shape[0]):
            r += self.weights[i] * self.utilities[i](y_it, p_sa, self.group)
        return r, y_it

    def reset(self):
        self.od_notifications = 0
        self.ts_y_notifications = 0
        self.ts_r_notifications = 0
        self.random_notifications = 0

        self.yprev_od = self.y0
        self.yprev_ts_y = self.y0
        self.yprev_ts_r = self.y0
        self.yprev_random = self.y0
        self.yprev_baseline = self.y0

class Gym(Dataset):
    def __init__(self, age, gender, state, is_new_member, visits=None, num_treatments=None):
        # Load the prediction model
        pth = "/Users/amandyam/Documents/Research/HeartSteps/policy-grad-boed/notebooks/regression_model.joblib"
        self.model = load(pth)

        # Initialize the state
        data = pd.read_csv(
            "/Users/amandyam/Documents/Research/HeartSteps/policy-grad-boed/data/24h_fitness/StepUp_Data/pptdata.csv")
        data = data.drop('Unnamed: 0', axis='columns')
        state_counts = Counter()
        for p_id, d in data.groupby('participant_id'):
            state = d['customer_state'].unique()[0]
            if np.any(d['customer_state'].isnull()):
                continue
            state_counts[state] += 1
        top_states = ['CA', 'TX', 'CO', 'WA', 'OR', 'FL', 'NY', 'HI', 'NJ']
        self.state_idx = {}
        for state in state_counts.keys():
            if state in top_states:
                self.state_idx[state] = top_states.index(state)
            else:
                self.state_idx[state] = 9

        self.included_treatments = ["Placebo Control", "Planning, Reminders & Micro-Incentives to Exercise"]
        treatments = [
            "Bonus for Returning after Missed Workouts b",
            "Higher Incentives a",
            "Exercise Social Norms Shared (High and Increasing)",
            "Free Audiobook Provided",
            "Bonus for Returning after Missed Workouts a",
            "Planning Fallacy Described and Planning Revision Encouraged",
            "Choice of Gain- or Loss-Framed Micro-Incentives",
            "Exercise Commitment Contract Explained",
            "Free Audiobook Provided, Temptation Bundling Explained",
            "Following Workout Plan Encouraged",
            "Fitness Questionnaire with Decision Support & Cognitive Reappraisal Prompt",
            "Values Affirmation",
            "Asked Questions about Workouts",
            "Rigidity Rewarded a",
            "Defaulted into 3 Weekly Workouts",
            "Exercise Advice Solicited",
            "Exercise Fun Facts Shared",
            "Fitness Questionnaire",
            "Planning Revision Encouraged",
            "Exercise Social Norms Shared (Low)",
            "Exercise Encouraged with Typed Pledge",
            "Gain-Framed Micro-Incentives",
            "Higher Incentives b",
            "Rigidity Rewarded e",
            "Exercise Encouraged with Signed Pledge",
            "Values Affirmation Followed by Diagnosis as Gritty",
            "Bonus for Consistent Exercise Schedule",
            "Rigidity Rewarded c",
            "Loss-Framed Micro-Incentives",
            # "Planning, Reminders & Micro-Incentives to Exercise",
            "Fitness Questionnaire with Cognitive Reappraisal Prompt",
            "Exercise Encouraged",
            "Planning Workouts Encouraged",
            "Gym Routine Encouraged",
            "Reflecting on Workouts Encouraged",
            "Planning Workouts Rewarded",
            "Effective Workouts Encouraged",
            "Planning Benefits Explained",
            "Reflecting on Workouts Rewarded",
            "Fun Workouts Encouraged",
            "Mon-Fri Consistency Rewarded, Sat-Sun Consistency Rewarded",
            "Exercise Encouraged with E-Signed Pledge",
            "Bonus for Variable Exercise Schedule",
            "Exercise Commitment Contract Explained Post-Intervention",
            "Rewarded for Responding to Questions about Workouts",
            "Defaulted into 1 Weekly Workout",
            "Exercise Social Norms Shared (Low but Increasing)",
            "Rigidity Rewarded d",
            "Exercise Commitment Contract Encouraged",
            "Fitness Questionnaire with Decision Support",
            "Rigidity Rewarded b",
            "Exercise Advice Solicited, Shared with Others",
            "Exercise Social Norms Shared (High)"
        ]
        self.included_treatments += treatments

        treatment_id_to_name = {}
        self.treatment_name_to_id = {}

        for (t_id, name), _ in data.groupby(['treatment', 'exp_condition']):
            treatment_id_to_name[t_id] = name
            self.treatment_name_to_id[name] = t_id

        self.treatment_idx = {
            '1233A': 0, '1233B': 0, '1234B': 0, '1252C': 0,  # placebo
            '1234A': 1, '1241C': 1, '1251C': 1, '1253A': 1  # baseline
        }
        K = len(self.included_treatments)
        self.treatment_feats = np.zeros((K, K - 1))
        self.treatment_feats[1:, :] = np.eye(K - 1)
        i = 2
        for t_name in self.included_treatments:
            t_id = self.treatment_name_to_id[t_name]
            if t_id not in self.treatment_idx:
                self.treatment_idx[t_id] = i
                i += 1
        self.covariates = [age] + list(np.eye(10)[self.state_idx[state]]) + [int(gender=='F'), is_new_member]
        self.state = None

        # Elements of the state that need to be kept updated
        self.num_treatments = 0
        self.week = 0
        self.previous_visits = []

        if visits is not None:
            self.previous_visits = visits
        if num_treatments is not None:
            self.num_treatments = num_treatments

    def pull(self, a):
        treatment_id = self.treatment_name_to_id[a]
        t_idx = self.treatment_idx[treatment_id]
        # Use the model to predict the new number of gym visits
        avg_visits = 0 if len(self.previous_visits) == 0 else np.mean(self.previous_visits)
        min_visits = 0 if len(self.previous_visits) == 0 else np.min(self.previous_visits)
        max_visits = 0 if len(self.previous_visits) == 0 else np.max(self.previous_visits)
        num_visits_last_week = 0 if len(self.previous_visits) < 1 else self.previous_visits[-1]
        num_visits_lastlast_week = 0 if len(self.previous_visits) < 2 else self.previous_visits[-2]
        longest_streak, num_streaks = self.streaks(self.previous_visits)
        receiving_treatment = 1
        self.num_treatments += 1
        state = [self.covariates + [avg_visits, min_visits, max_visits, num_visits_last_week, num_visits_lastlast_week] + [longest_streak, num_streaks] + [receiving_treatment, self.num_treatments] + self.treatment_feats[t_idx].tolist()]
        num_visits = self.model.predict(state)
        self.week += 1
        self.previous_visits.append(num_visits[0])
        return int(num_visits)

    def pull_no_update(self, a):
        treatment_id = self.treatment_name_to_id[a]
        t_idx = self.treatment_idx[treatment_id]
        # Use the model to predict the new number of gym visits
        avg_visits = 0 if len(self.previous_visits) == 0 else np.mean(self.previous_visits)
        min_visits = 0 if len(self.previous_visits) == 0 else np.min(self.previous_visits)
        max_visits = 0 if len(self.previous_visits) == 0 else np.max(self.previous_visits)
        num_visits_last_week = 0 if len(self.previous_visits) < 1 else self.previous_visits[-1]
        num_visits_lastlast_week = 0 if len(self.previous_visits) < 2 else self.previous_visits[-2]
        longest_streak, num_streaks = self.streaks(self.previous_visits)
        receiving_treatment = 1
        self.num_treatments = 1
        try:
            state = [
                self.covariates + [avg_visits, min_visits, max_visits, num_visits_last_week, num_visits_lastlast_week] + [
                    longest_streak, num_streaks] + [receiving_treatment, self.num_treatments] + self.treatment_feats[
                    t_idx].tolist()]
            num_visits = self.model.predict(state)
        except:
            print(str(state))
            num_visits = 0
        return int(num_visits)

    def streaks(self, lst):
        if len(lst) == 0:
            return 0, 0
        else:
            tmp = []
            for n, c in groupby(lst):
                num, count = n, sum(1 for i in c)
                if num == 1:
                    tmp.append((num, count))
            if len(tmp) == 0:
                return 0, 0
            longest_streak = max([y for x, y in tmp])
            num_streaks = len(tmp)
            return longest_streak, num_streaks

def bits_list(n, val = None):
    bits = [int(x) for x in bin(n)[2:]]
    if val is None or len(bits) >= val:
        return bits
    else:
        return [0] * (val-len(bits)) + bits

class Education(object):
    def __init__(self, student_model, best_model_weights, idx_to_problem, problem_to_idx,  n_features, curriculum, H):

        self.student_model = student_model
        self.ws = 10
        self.thresh = 0.85  # Threshold for mastery

        self.student_model.load_weights(best_model_weights)  # Predicts whether showing an action will give you mastery
        self.idx_to_problem = idx_to_problem
        self.problem_to_idx = problem_to_idx
        self.H = H
        self.n_actions = len(self.problem_to_idx)  # Number of problems to present

        self.n_time_features = len(bits_list(self.H))
        self.n_state_features = self.n_actions + self.n_time_features
        self.n_features = n_features
        self.curriculum = curriculum
        self.mastered_problems = []
        self.goalset_idx = 0
        self.reset()

    def format_input(self, problem_seq, answer_seq):
        if len(problem_seq) == 0:
            return np.array([[[-1] * self.n_features]])
        x_seq = []
        l_seq = len(problem_seq)
        for problem, answer in zip(problem_seq, answer_seq):
            idx = problem * 2 + answer
            x = np.zeros(self.n_features)
            x[idx] = 1
            x_seq.append(x)

        return tf.convert_to_tensor(np.array([x_seq]))

    def constrained_actions(self):
        constrained_problems = []
        for a_idx in range(self.n_actions):
            problem = self.idx_to_problem[a_idx]
            prereqs = self.curriculum[str(problem)]
            all_satisfied = True
            for p in prereqs:
                p_idx = self.problem_to_idx[p]
                if p_idx not in self.mastered_problems:
                    all_satisfied = False
            if all_satisfied:
                constrained_problems.append(a_idx)
        return constrained_problems

    def goal_sets(self):
        # Get the first level of prereqs


        # Get the second level
        pass

    def reset(self):
        self.treatments = np.asarray([0]*51)
        self.state = np.concatenate((np.asarray([0]*self.n_actions), self.treatments)) # Mastered problems, treatments
        self.problems = []
        self.answers = []
        self.h = 0
        self.problem_probs = [0.0 for i in range(self.n_actions)]
        self.mastered = {i: 0 for i in range(self.n_actions)}
        return np.array(self.problem_probs + bits_list(self.h, val=self.n_time_features))

    def step(self, problem_idx):
        self.treatments[problem_idx] += 1 # Accumulate the treatment
        problem = self.idx_to_problem[problem_idx]
        prob = 0
        num_attempts = 0
        # Pick the problem that exceeds the threshold of mastery according to the prediction model
        while prob < self.thresh and self.h < self.H and num_attempts < self.ws:
            self.problems.append(problem)
            self.answers.append(1)
            prob = self.student_model.predict(self.format_input(self.problems, self.answers))[0][-1][problem]
            self.h += 1
            num_attempts += 1

        terminate = self.h == self.H
        reward = int((prob >= self.thresh) and (self.problem_probs[problem_idx] < self.thresh))
        # This depends on action, prob is predicting the probability of answering a problem correctly
        self.problem_probs[problem_idx] = prob
        #next_state = np.array(self.problem_probs + bits_list(self.h, val=self.n_time_features)) # Old simulator

        # Actual next state is problem probs + self.treatments
        next_state = np.concatenate((np.asarray(self.problem_probs), np.asarray(self.treatments)))
        if reward == 1:
            self.mastered_problems.append(problem_idx)
        return next_state, reward, terminate, None


def get_custom_DKT_model(saved_data_folder='saved_data'):
    all_problems_file_name = f'{saved_data_folder}/all_problems.txt'
    all_problems = []

    with open(all_problems_file_name, 'r') as filehandle:
        for line in filehandle:
            problem = line[:-1]  # remove linebreak which is the last character of the string
            all_problems.append(int(problem))

    n_problems = len(all_problems)
    n_features = 2 * n_problems

    MASK = -1.
    val_fraction = 0.2
    verbose = 1  # Verbose = {0,1,2}
    optimizer = "adam"  # Optimizer to use
    lstm_units = 200  # Number of LSTM units
    dropout_rate = 0.1  # Dropout rate
    metrics = [MetricWrapper(tf.keras.metrics.AUC()),
               MetricWrapper(tf.keras.metrics.BinaryAccuracy())]

    student_model = DKT_model(n_features, lstm_units, dropout_rate, n_problems)
    student_model.compile_model(custom_loss_func, metrics, optimizer, verbose)

    return student_model


def get_DKT_CMAB(prereq_graph, saved_data_folder='environments_and_constraints/education/saved_data', THRESH=0.85, H=100, ws=10):
    load_model_weights = f"{saved_data_folder}/saved_model_weights/bestvalmodel"

    with open(f'{saved_data_folder}/idx_to_problem.json', 'r') as f:
        idx_to_problem = json.load(f)
    with open(f'{saved_data_folder}/problem_to_idx.json', 'r') as f:
        problem_to_idx = json.load(f)

    all_problems_file_name = f'{saved_data_folder}/all_problems.txt'
    all_problems = []

    with open(all_problems_file_name, 'r') as filehandle:
        for line in filehandle:
            problem = line[:-1]  # remove linebreak which is the last character of the string
            all_problems.append(int(problem))

    n_problems = len(all_problems)
    n_features = 2 * n_problems

    idx_to_problem = convert_keys_to_int(idx_to_problem)
    problem_to_idx = convert_keys_to_int(problem_to_idx)

    student_model = get_custom_DKT_model(saved_data_folder=saved_data_folder)

    return Education(student_model, load_model_weights, idx_to_problem, problem_to_idx, n_features=n_features, curriculum=prereq_graph, H=H)