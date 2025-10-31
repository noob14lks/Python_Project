import os
import sys
import numpy as np


def check_dependencies():
    """Check if all required packages are installed"""
    print("="*70)
    print("CHECKING DEPENDENCIES...")
    print("="*70)
    
    missing = []
    
    try:
        import tensorflow
    except ImportError:
        missing.append('tensorflow')
    
    try:
        import sklearn
    except ImportError:
        missing.append('scikit-learn')
    
    try:
        import matplotlib
    except ImportError:
        missing.append('matplotlib')
    
    try:
        import pandas
    except ImportError:
        missing.append('pandas')
    
    try:
        from deap import base
    except ImportError:
        missing.append('deap')
    
    try:
        from PIL import Image
    except ImportError:
        missing.append('pillow')
    
    if len(missing) > 0:
        print(f"✗ Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing)}")
        sys.exit(1)
    
    print("✓ All dependencies installed\n")


# Check dependencies first
check_dependencies()


# Import modules
from feature_extraction import process_all_datasets
from base_feature_selection import generate_initial_population, filter_anova
from nsga2_optimization import run_nsga2, evaluate
from classification_evaluation import compare_methods
from visualization import (plot_pareto_front, plot_confusion_matrix,
                          plot_comparison_bar, export_results_to_csv)


def main():
    # """Main pipeline execution - AUGMENTED DATA ONLY"""
    # print("\n" + "="*70)
    # print("MULTI-OBJECTIVE GENETIC ALGORITHM FOR FEATURE SELECTION")
    # print("Based on Bohrer et al. (2024)")
    # print("MODE: AUGMENTED DATA ONLY")
    # print("="*70)
    
    # # Create necessary folders
    # os.makedirs('./features', exist_ok=True)
    # os.makedirs('./results', exist_ok=True)
    
    # # ========================================================================
    # # STEP 1: DIRECT FEATURE EXTRACTION (AUGMENTED ONLY)
    # # ========================================================================
    # print("\n" + "="*70)
    # print("[STEP 1] EXTRACTING CNN FEATURES FROM AUGMENTED IMAGES")
    # print("="*70)
    
    # data_folder = './data/extracteddata'
    
    # if not os.path.exists(data_folder):
    #     print(f"✗ Error: '{data_folder}' not found!")
    #     sys.exit(1)
    
    # # Extract features from augmented data only
    # feature_files = process_all_datasets(
    #     input_folder=data_folder,
    #     output_folder='./features',
    #     only_augmented=True  # ← AUGMENTED DATA ONLY
    # )
    
    # if len(feature_files) == 0:
    #     print("\n✗ No datasets were successfully processed!")
    #     sys.exit(1)
    
    # print(f"\n✓ Extracted features from {len(feature_files)} datasets (AUGMENTED ONLY)")
    
    # ========================================================================
    # STEP 2: FEATURE SELECTION WITH NSGA-II
    # ========================================================================
    print("\n" + "="*70)
    print("[STEP 2] MULTI-OBJECTIVE FEATURE SELECTION")
    print("="*70)
    
    # Get ALL feature files
    all_feature_files = [f for f in os.listdir('./features') 
                        if f.endswith('.npz')]


    if len(all_feature_files) == 0:
        print("✗ No feature files found!")
        sys.exit(1)


    print(f"Processing {len(all_feature_files)} datasets\n")


    all_results = []


    for idx, feature_file in enumerate(all_feature_files, 1):
        # Extract dataset name by removing all possible suffixes
        dataset_name = feature_file
        for suffix in ['_augmented_densenet121_features.npz', '_densenet121_features.npz', 
                      '_vgg16_features.npz', '_features.npz']:
            dataset_name = dataset_name.replace(suffix, '')
        
        print(f"\n{'='*70}")
        print(f"DATASET {idx}/{len(all_feature_files)}: {dataset_name} (AUGMENTED)")
        print(f"{'='*70}")
        
        try:
            # Load features
            data = np.load(os.path.join('./features', feature_file))
            X_train, y_train = data['X_train'], data['y_train']
            X_val, y_val = data['X_val'], data['y_val']
            X_test, y_test = data['X_test'], data['y_test']
            
            print(f"  Features: {X_train.shape[1]} (DenseNet121)")
            print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
            
            # Generate initial population
            k_features = min(100, max(10, X_train.shape[1] // 10))
            
            print(f"\n  Generating initial population...")
            initial_pop = generate_initial_population(X_train, y_train, 
                                                     pop_size=50, k=k_features)
            _, importance_scores = embedded_rf(X_train, y_train, k=X_train.shape[1])
            filter_indices, _ = filter_anova(X_train, y_train, k=k_features)
            print(f"  ✓ Population ready")
            
            # Run NSGA-II
            print(f"\n  Running NSGA-II optimization...")
            pareto_front = run_nsga2(X_train, y_train, X_val, y_val,
                                     population_size=50, ngen=30,
                                     initial_pop=initial_pop,
                                     importance_scores=importance_scores,
                                     classifier_type='svm')  # Changed to 'svm'
            
            # Extract Pareto results
            print(f"\n  Extracting Pareto solutions...")
            pareto_results = []
            for ind in pareto_front:
                acc, num_feat = evaluate(ind, X_train, y_train, X_val, y_val, 'svm')  # Changed to 'svm'
                pareto_results.append({
                    'indices': [i for i, bit in enumerate(ind) if bit == 1],
                    'accuracy': acc,
                    'num_features': num_feat
                })
            
            best_solution = max(pareto_results, 
                                key=lambda x: x['accuracy'] / (x['num_features'] + 1))
            ga_indices = best_solution['indices']
            
            print(f"  ✓ Best solution: {len(ga_indices)} features, "
                  f"Val Accuracy: {best_solution['accuracy']:.4f}")
            
            # Generate visualizations
            print(f"\n  Generating visualizations...")
            plot_pareto_front(pareto_results, 
                              f'./results/{dataset_name}_augmented_pareto.png',
                              f"{dataset_name} (Augmented)")
            
            # Final evaluation on test set
            print(f"\n  Final evaluation on test set...")
            metrics_all, metrics_filter, metrics_ga = compare_methods(
                X_train, y_train, X_test, y_test,
                filter_indices, ga_indices, 'svm'  # Changed to 'svm'
            )
            
            plot_confusion_matrix(metrics_ga['confusion_matrix'],
                                  f'./results/{dataset_name}_augmented_confusion.png',
                                  f"{dataset_name} (Augmented)")
            
            comparison = {
                'All Features': metrics_all,
                'Filter (ANOVA)': metrics_filter,
                'GA (NSGA-II)': metrics_ga
            }
            plot_comparison_bar(comparison,
                               f'./results/{dataset_name}_augmented_comparison.png',
                               f"{dataset_name} (Augmented)")
            
            # Store results
            all_results.append({
                'Dataset': f"{dataset_name}_augmented",
                'Total_Features': X_train.shape[1],
                'GA_Features': metrics_ga['num_features'],
                'GA_Test_Accuracy': metrics_ga['accuracy'],
                'GA_F1_Score': metrics_ga['f1_score'],
                'GA_Precision': metrics_ga['precision'],
                'GA_Recall': metrics_ga['recall']
            })
            
            print(f"\n✓ Completed: {dataset_name} (Augmented)")
            
        except Exception as e:
            print(f"\n✗ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


    print(f"\n{'='*70}")
    print("EXPORTING RESULTS SUMMARY")
    print(f"{'='*70}")
    
    if len(all_results) > 0:
        export_results_to_csv(all_results, './results/summary_augmented.csv')
        
        print("\n" + "="*70)
        print("FINAL RESULTS (AUGMENTED DATA)")
        print("="*70)
        print(f"{'Dataset':<35} {'Test Accuracy':<15} {'Features':<10}")
        print("-"*70)
        for result in all_results:
            print(f"{result['Dataset']:<35} "
                  f"{result['GA_Test_Accuracy']:<15.4f} "
                  f"{result['GA_Features']:<10}")
        print("="*70)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Results saved in './results/' directory")
    print("All results use AUGMENTED DATA ONLY")
    print("="*70)


# Run the main function
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
