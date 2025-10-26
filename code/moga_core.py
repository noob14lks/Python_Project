import random
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def setup_nsga2(num_features):
    if not hasattr(setup_nsga2, "initialized"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        setup_nsga2.initialized = True

    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: creator.Individual([random.choice([0, 1]) for _ in range(num_features)]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox

def guided_crossover(ind1, ind2, indpb=0.5):
    for i in range(len(ind1)):
        if ind1[i] == ind2[i]:
            continue
        elif random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

def guided_mutation(individual, importance_scores=None, mutpb=0.1):
    for i in range(len(individual)):
        if random.random() < mutpb:
            prob = 1.0
            if importance_scores is not None:
                prob = 1 - importance_scores[i]/max(importance_scores)
            if random.random() < prob:
                individual[i] = 1 - individual[i]
    return individual

def evaluate(individual, X, y, classifier=RandomForestClassifier(n_estimators=30, n_jobs=-1)):
    selected_idx = [i for i, bit in enumerate(individual) if bit == 1]
    if len(selected_idx) == 0:
        return 0.0, len(individual)
    X_selected = X[:, selected_idx]
    scores = cross_val_score(classifier, X_selected, y, cv=2, scoring='accuracy')
    accuracy = scores.mean()
    return accuracy, len(selected_idx)

def run_nsga2(X, y, population_size=20, ngen=10, initial_pop=None, importance_scores=None):
    num_features = X.shape[1]
    toolbox = setup_nsga2(num_features)
    toolbox.register("mate", guided_crossover, indpb=0.5)
    toolbox.register("mutate", guided_mutation, importance_scores=importance_scores, mutpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate, X=X, y=y)

    if initial_pop is not None:
        population = [creator.Individual(ind) for ind in initial_pop]
    else:
        population = toolbox.population(n=population_size)

    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    for gen in range(ngen):
        print(f"[NSGA-II] Generation {gen+1}/{ngen}")
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.3)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring + population, population_size)
        hof.update(population)
    print("[NSGA-II] Optimization finished.")
    return hof
