import random
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np


def setup_nsga2(num_features):
    if not hasattr(setup_nsga2, "initialized"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        setup_nsga2.initialized = True
    
    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: creator.Individual(
        [random.choice([0, 1]) for _ in range(num_features)]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox


def guided_crossover(ind1, ind2, indpb=0.7):
    for i in range(len(ind1)):
        if ind1[i] == ind2[i]:
            continue
        elif random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


def guided_mutation(individual, importance_scores=None, mutpb=0.15):
    for i in range(len(individual)):
        if random.random() < mutpb:
            prob = 1.0
            if importance_scores is not None and max(importance_scores) > 0:
                prob = 1 - importance_scores[i] / max(importance_scores)
            if random.random() < prob:
                individual[i] = 1 - individual[i]
    return (individual,)


def evaluate(individual, X_train, y_train, X_val, y_val, classifier_type='rf'):
    selected_idx = [i for i, bit in enumerate(individual) if bit == 1]
    
    if len(selected_idx) == 0:
        return (0.0, len(individual))
    
    X_train_selected = X_train[:, selected_idx]
    X_val_selected = X_val[:, selected_idx]
    
    try:
        if classifier_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        elif classifier_type == 'svm':
            clf = SVC(kernel='rbf', random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        
        clf.fit(X_train_selected, y_train)
        accuracy = clf.score(X_val_selected, y_val)
    except Exception as e:
        print(f"Evaluation error: {e}")
        accuracy = 0.0
    
    return (accuracy, len(selected_idx))


def run_nsga2(X_train, y_train, X_val, y_val, population_size=120, ngen=60, 
              initial_pop=None, importance_scores=None, classifier_type='rf'):
    num_features = X_train.shape[1]
    toolbox = setup_nsga2(num_features)
    
    toolbox.register("mate", guided_crossover, indpb=0.7)
    toolbox.register("mutate", guided_mutation, 
                    importance_scores=importance_scores, mutpb=0.15)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate, X_train=X_train, y_train=y_train,
                     X_val=X_val, y_val=y_val, classifier_type=classifier_type)
    
    if initial_pop is not None:
        population = [creator.Individual(ind) for ind in initial_pop]
    else:
        population = toolbox.population(n=population_size)
    
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    for ind in invalid_ind:
        ind.fitness.values = toolbox.evaluate(ind)
    
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    print(f"Starting NSGA-II: pop_size={population_size}, generations={ngen}")
    
    for gen in range(ngen):
        print(f"  Generation {gen+1}/{ngen}", end='\r')
        
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.3)
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)
        
        population = toolbox.select(population + offspring, population_size)
        hof.update(population)
    
    print(f"\n  NSGA-II completed. Pareto front size: {len(hof)}")
    return hof
