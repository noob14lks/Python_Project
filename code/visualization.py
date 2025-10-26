import matplotlib.pyplot as plt
import csv

def plot_pareto_front(pareto_results, fname='pareto_front.png'):
    acc = [res['accuracy'] for res in pareto_results]
    num_feats = [res['num_features'] for res in pareto_results]
    plt.figure(figsize=(6,4))
    plt.scatter(num_feats, acc, c='blue')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title('Pareto Front')
    plt.savefig(fname)
    plt.close()
    print(f'Pareto front plot saved as {fname}')

def export_results_csv(pareto_results, fname='pareto_results.csv'):
    with open(fname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Indices', 'Accuracy', 'NumFeatures'])
        for res in pareto_results:
            writer.writerow([res['indices'], res['accuracy'], res['num_features']])
    print(f'Results exported to {fname}')
