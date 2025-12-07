import sys
import os
import numpy as np
import pandas as pd
import json

# Add the cec2017-py directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cec2017-py'))

import cec2017.functions as functions
from algorithms.hs import HarmonySearch, AdaptiveHS, HybridHS
from algorithms.others import PSO, GA, Greedy, GSA, ABC

def run_test():
    # Configuration
    DIM = 30
    MAX_EVALS = 1000 # Reduced for test
    RUNS = 2 # Reduced for test
    
    # Algorithms to test
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
    
    # Test only F1
    funcs = [functions.f1]
    
    results = []
    
    for f_idx, func in enumerate(funcs):
        func_name = f"F{f_idx+1}"
        print(f"Testing {func_name}...")
        
        for alg_name, alg_class in algorithms.items():
            print(f"  Running {alg_name}...")
            fitness_values = []
            
            for run in range(RUNS):
                alg = alg_class(func, DIM, MAX_EVALS)
                best_fitness, _, _ = alg.run()
                fitness_values.append(best_fitness)
                
            print(f"    Best: {np.min(fitness_values)}")
            
    print("Test completed successfully.")

if __name__ == "__main__":
    run_test()
