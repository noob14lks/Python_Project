from dataset_loader import load_all_features
from base_feature_selection import generate_initial_population, embedded_rf
from nsga2_optimization import run_nsga2, evaluate
from visualization import plot_pareto_front, export_results_csv
from baseline_comparison import evaluate_baselines
import numpy as np

datasets, file_paths = load_all_features("./features")

for i, (X, y) in enumerate(datasets):
    print(f"\n=== Dataset {i+1} ===")
    print(f"Source file: {file_paths[i]}")
    initial_pop = generate_initial_population(X, y, pop_size=20, k=15)
    _, importance_scores = embedded_rf(X, y, k=X.shape[1])
    pareto_front = run_nsga2(X, y, population_size=20, ngen=10,
                            initial_pop=initial_pop,
                            importance_scores=importance_scores)
    pareto_results = []
    for ind in pareto_front:
        acc, num_feat = evaluate(ind, X, y)
        pareto_results.append({'indices': [idx for idx, bit in enumerate(ind) if bit == 1],
                               'accuracy': acc, 'num_features': num_feat})
        print(f"Accuracy: {acc:.4f}, Features: {num_feat}")

    ds_prefix = file_paths[i].split('/')[-1].replace('.npz','')
    plot_pareto_front(pareto_results, fname=f"{ds_prefix}_pareto_front.png")
    export_results_csv(pareto_results, fname=f"{ds_prefix}_pareto_results.csv")

    best = max(pareto_results, key=lambda x: x['accuracy'] / (x['num_features']+1))
    print(f"Best subset selected indices: {best['indices']}")
    evaluate_baselines(X, y, idx_selected=best['indices']) 
