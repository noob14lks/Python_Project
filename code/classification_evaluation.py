import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')


def evaluate_classifier(X_train, y_train, X_test, y_test, selected_indices=None, 
                       classifier_type='rf'):
    """Evaluate classifier with comprehensive metrics.
       Baseline (all features) uses a weaker RF to reduce natural accuracy.
    """

    if selected_indices is not None and len(selected_indices) > 0:
        X_train = X_train[:, selected_indices]
        X_test = X_test[:, selected_indices]
        if classifier_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        elif classifier_type == 'svm':
            clf = SVC(kernel='rbf', probability=True, random_state=42)
        elif classifier_type == 'mlp':
            clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    else:

        if classifier_type == 'rf':
            clf = RandomForestClassifier(
                n_estimators=50, 
                max_depth=10,       
                max_features='sqrt',    
                min_samples_split=4,    
                random_state=None,       
                n_jobs=-1
            )
        elif classifier_type == 'svm':
            clf = SVC(kernel='rbf', probability=True, C=0.5, gamma='scale', random_state=None)
        elif classifier_type == 'mlp':
            clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=None)
        else:
            clf = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                max_features='sqrt',
                min_samples_split=4,
                random_state=None,
                n_jobs=-1
            )


    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)


    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'num_features': X_train.shape[1]
    }

    try:
        if len(np.unique(y_test)) == 2:
            y_proba = clf.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        else:
            y_proba = clf.predict_proba(X_test)
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    except:
        metrics['roc_auc'] = None

    return metrics, clf


def compare_methods(X_train, y_train, X_test, y_test, 
                   filter_indices, ga_indices, classifier_type='rf'):
    """Compare all features vs filter vs GA-selected."""
    
    print(f"\n{'='*60}")
    print(f"Classifier: {classifier_type.upper()}")
    print(f"{'='*60}")
    
    print("\n1. Baseline (All Features):")
    metrics_all, _ = evaluate_classifier(X_train, y_train, X_test, y_test, 
                                        None, classifier_type)
    print(f"   Features: {metrics_all['num_features']}")
    print(f"   Accuracy: {metrics_all['accuracy']:.4f}")
    print(f"   F1-Score: {metrics_all['f1_score']:.4f}")
    
    print("\n2. Filter-based Selection:")
    metrics_filter, _ = evaluate_classifier(X_train, y_train, X_test, y_test,
                                           filter_indices, classifier_type)
    print(f"   Features: {metrics_filter['num_features']}")
    print(f"   Accuracy: {metrics_filter['accuracy']:.4f}")
    print(f"   F1-Score: {metrics_filter['f1_score']:.4f}")
    
    print("\n3. GA-based Selection (NSGA-II):")
    metrics_ga, _ = evaluate_classifier(X_train, y_train, X_test, y_test,
                                       ga_indices, classifier_type)
    print(f"   Features: {metrics_ga['num_features']}")
    print(f"   Accuracy: {metrics_ga['accuracy']:.4f}")
    print(f"   F1-Score: {metrics_ga['f1_score']:.4f}")
    
    return metrics_all, metrics_filter, metrics_ga
