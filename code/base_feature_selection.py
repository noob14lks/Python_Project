import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

def filter_anova(X, y, k=50):
    """ANOVA F-score filter"""
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    scores = selector.scores_
    selected_idx = selector.get_support(indices=True)
    return selected_idx, scores

def filter_mutual_info(X, y, k=50):
    """Mutual Information filter"""
    mi = mutual_info_classif(X, y, random_state=42)
    k = min(k, X.shape[1])
    idx = np.argsort(mi)[-k:]
    return idx, mi

def embedded_rf(X, y, k=50):
    """Random Forest feature importance"""
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    k = min(k, X.shape[1])
    idx = np.argsort(importances)[-k:]
    return idx, importances

def generate_initial_population(X, y, pop_size=50, k=50):
    """Generate hybrid initial population guided by filter methods"""
    num_features = X.shape[1]
    k = min(k, num_features)
    
    idx_anova, _ = filter_anova(X, y, k)
    idx_mi, _ = filter_mutual_info(X, y, k)
    idx_rf, _ = embedded_rf(X, y, k)
    
    combined_idx = np.unique(np.concatenate([idx_anova, idx_mi, idx_rf]))
    
    base_chs = np.zeros(num_features)
    base_chs[combined_idx] = 1
    
    pop = []
    for _ in range(pop_size):
        rand_chs = (base_chs + np.random.binomial(1, 0.2, num_features)) > 0
        pop.append(rand_chs.astype(int).tolist())
    
    return pop
