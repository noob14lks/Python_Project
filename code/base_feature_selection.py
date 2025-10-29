import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

def filter_anova(X, y, k=20):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    scores = selector.scores_
    selected_idx = selector.get_support(indices=True)
    return selected_idx, scores

def filter_mutual_info(X, y, k=20):
    mi = mutual_info_classif(X, y)
    idx = np.argsort(mi)[-k:]
    return idx, mi

def embedded_rf(X, y, k=20):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X, y)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[-k:]
    return idx, importances

def generate_initial_population(X, y, pop_size=50, k=20):
    num_features = X.shape[1]
    idx_anova, _ = filter_anova(X, y, k)
    idx_mi, _ = filter_mutual_info(X, y, k)
    idx_rf, _ = embedded_rf(X, y, k)

    base_idxs = np.unique(np.concatenate([idx_anova, idx_mi, idx_rf]))

    population = []
    for _ in range(pop_size):
        individual = np.zeros(num_features, dtype=int)
        individual[base_idxs] = 1
        flip_percentage = 0.2
        flipidx = np.random.choice(num_features, size=int(flip_percentage*num_features), replace=False)
        individual[flipidx] = 1 - individual[flipidx]
        population.append(individual.tolist())

    population.append([1]*num_features)
    population.append([0]*num_features)

    return population
