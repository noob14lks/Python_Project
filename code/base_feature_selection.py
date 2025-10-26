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
    rf = RandomForestClassifier(n_estimators=30, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[-k:]
    return idx, importances

def generate_initial_population(X, y, pop_size=20, k=15):
    num_features = X.shape[1]
    idx_anova, score_anova = filter_anova(X, y, k)
    idx_mi, score_mi = filter_mutual_info(X, y, k)
    idx_rf, score_rf = embedded_rf(X, y, k)
    combined_idx = np.unique(np.concatenate([idx_anova, idx_mi, idx_rf]))
    base_chs = np.zeros(num_features)
    base_chs[combined_idx] = 1
    pop = []
    for _ in range(pop_size):
        rand_chs = (base_chs + np.random.binomial(1, 0.3, num_features)) > 0
        pop.append(rand_chs.astype(int).tolist())
    return pop
