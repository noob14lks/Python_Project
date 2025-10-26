import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def evaluate_baselines(X, y, idx_selected=None):
    score_all = np.mean(cross_val_score(RandomForestClassifier(), X, y, cv=2))
    print(f'RF accuracy (all features): {score_all:.4f}')
    if idx_selected is not None and len(idx_selected) > 0:
        X_s = X[:, idx_selected]
        score_sel = np.mean(cross_val_score(RandomForestClassifier(), X_s, y, cv=2))
        print(f'RF accuracy ({len(idx_selected)} features): {score_sel:.4f}')
