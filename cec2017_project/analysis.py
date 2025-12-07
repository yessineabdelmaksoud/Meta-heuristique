import pandas as pd
import json
import matplotlib.pyplot as plt
import os

def generate_table():
    if not os.path.exists('cec2017_results.csv'):
        print("No results file found.")
        return

    df = pd.read_csv('cec2017_results.csv')
    
    # Pivot table to show Mean and Std for each algorithm and function
    pivot_mean = df.pivot(index='Function', columns='Algorithm', values='Mean')
    pivot_std = df.pivot(index='Function', columns='Algorithm', values='Std')
    
    print("Results Table (Mean):")
    print(pivot_mean)
    
    pivot_mean.to_csv('results_summary_mean.csv')
    pivot_std.to_csv('results_summary_std.csv')

def plot_convergence():
    if not os.path.exists('convergence_data.json'):
        print("No convergence data found.")
        return

    with open('convergence_data.json', 'r') as f:
        data = json.load(f)
        
    for func_name, algs in data.items():
        plt.figure(figsize=(10, 6))
        
        for alg_name, history in algs.items():
            plt.plot(history, label=alg_name)
            
        plt.title(f'Convergence Curve - {func_name}')
        plt.xlabel('Evaluations')
        plt.ylabel('Fitness')
        plt.yscale('log') # Log scale is often better for convergence
        plt.legend()
        plt.grid(True)
        plt.savefig(f'convergence_{func_name}.png')
        plt.close()

if __name__ == "__main__":
    generate_table()
    plot_convergence()
