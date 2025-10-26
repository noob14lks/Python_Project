from base_feature_selection import filter_anova, filter_mutual_info, embedded_rf

def run_filter_selection(X, y, k=20):
    idx_anova, scores_anova = filter_anova(X, y, k)
    idx_mi, scores_mi = filter_mutual_info(X, y, k)
    idx_rf, importances_rf = embedded_rf(X, y, k)
    print('ANOVA top:', idx_anova)
    print('MI top:', idx_mi)
    print('RF top:', idx_rf)
    return {
        'anova': (idx_anova, scores_anova),
        'mi': (idx_mi, scores_mi),
        'rf': (idx_rf, importances_rf)
    }
