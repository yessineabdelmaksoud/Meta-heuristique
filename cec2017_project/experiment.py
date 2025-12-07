import sys
import os
import numpy as np
import pandas as pd
import json
import time
from concurrent.futures import ProcessPoolExecutor
import itertools

# Add the cec2017-py directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cec2017-py'))

import cec2017.functions as functions
from algorithms.hs import HarmonySearch, AdaptiveHS, HybridHS
from algorithms.others import PSO, GA, Greedy, GSA, ABC

def run_single_experiment(args):
    func_name, alg_name, run_id, dim, max_evals = args
    
    # Re-import or get function/class based on name to avoid pickling issues if any
    # But classes should be picklable.
    
    # Map names to objects
    algorithms = {
        'HS': HarmonySearch,
        'AdaptiveHS': AdaptiveHS,
        'HybridHS': HybridHS,
        'PSO': PSO,
        'GA': GA,
        'Greedy': Greedy,
        'GSA': GSA,
        'ABC': ABC
    }
    
    # Get function by name from cec2017.functions
    # functions.all_functions is a list, so we need to find it by name or index
    # Let's assume func_name is "F1", "F2", etc.
    f_idx = int(func_name[1:]) - 1
    func = functions.all_functions[f_idx]
    
    alg_class = algorithms[alg_name]
    
    alg = alg_class(func, dim, max_evals)
    best_fitness, _, history = alg.run()
    
    return {
        'Function': func_name,
        'Algorithm': alg_name,
        'Run': run_id,
        'BestFitness': best_fitness,
        'History': history
    }

def run_experiment():
    # Configuration
    DIM = 30
    MAX_EVALS = 30000
    RUNS = 2 # Reduced for demonstration purposes. Set to 30 for full experiment.
    
    # Algorithms to test
    alg_names = ['HS', 'AdaptiveHS', 'HybridHS', 'PSO', 'GA', 'Greedy', 'GSA', 'ABC']
    
    # Functions to test (F1 to F30)
    func_names = [f"F{i+1}" for i in range(30)]
    
    # Create tasks
    tasks = []
    for func_name in func_names:
        for alg_name in alg_names:
            for run in range(RUNS):
                tasks.append((func_name, alg_name, run, DIM, MAX_EVALS))
    
    print(f"Total tasks: {len(tasks)}")
    
    results = []
    convergence_data = {}
    
    # Specific functions for convergence curves
    convergence_funcs = ['F2', 'F4', 'F12', 'F25']
    
    # Run in parallel
    # Adjust max_workers based on CPU cores.
    max_workers = os.cpu_count() or 4
    print(f"Running with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, result in enumerate(executor.map(run_single_experiment, tasks)):
            if i % 100 == 0:
                print(f"Completed {i}/{len(tasks)} tasks")
            
            # Aggregate results
            # We need to aggregate by Function and Algorithm later
            # For now just store the raw result or partial aggregation?
            # Storing 7200 histories might be too much memory?
            # Only store history if it's one of the convergence functions
            
            if result['Function'] in convergence_funcs:
                fname = result['Function']
                aname = result['Algorithm']
                if fname not in convergence_data:
                    convergence_data[fname] = {}
                if aname not in convergence_data[fname]:
                    convergence_data[fname][aname] = []
                convergence_data[fname][aname].append(result['History'])
            
            # We only need the BestFitness for the table
            results.append({
                'Function': result['Function'],
                'Algorithm': result['Algorithm'],
                'BestFitness': result['BestFitness']
            })

    # Process results for CSV
    df = pd.DataFrame(results)
    
    # Group by Function and Algorithm to get Mean and Std
    summary = df.groupby(['Function', 'Algorithm'])['BestFitness'].agg(['mean', 'std', 'min', 'max']).reset_index()
    summary.rename(columns={'mean': 'Mean', 'std': 'Std', 'min': 'Best', 'max': 'Worst'}, inplace=True)
    
    summary.to_csv('cec2017_results.csv', index=False)
    
    # Process convergence data
    avg_convergence = {}
    for fname, algs in convergence_data.items():
        avg_convergence[fname] = {}
        for alg_name, histories in algs.items():
            max_len = max(len(h) for h in histories)
            padded_histories = []
            for h in histories:
                if len(h) < max_len:
                    h = h + [h[-1]] * (max_len - len(h))
                padded_histories.append(h)
            
            avg_hist = np.mean(padded_histories, axis=0).tolist()
            avg_convergence[fname][alg_name] = avg_hist
            
    with open('convergence_data.json', 'w') as f:
        json.dump(avg_convergence, f)
    
    print("Experiment completed.")

if __name__ == "__main__":
    run_experiment()
