import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Add the project directories to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cec2017-py'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import cec2017.functions as functions
    from cec2017_project.algorithms.hs import HarmonySearch, AdaptiveHS, HybridHS
    from cec2017_project.algorithms.others import PSO, GA, Greedy, GSA, ABC
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

st.set_page_config(page_title="CEC2017 Optimization Benchmark", layout="wide")

st.title("🧬 CEC2017 Optimization Benchmark Visualizer")
st.markdown("""
This application allows you to interactively run and visualize meta-heuristic algorithms on the CEC2017 benchmark functions.
""")

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# Mode Selection
mode = st.sidebar.radio("Select Mode", ["Single Algorithm Run", "Algorithm Comparison"])

# Common Parameters (Shared across modes)
st.sidebar.subheader("Common Parameters")
dim = st.sidebar.number_input("Dimension (D)", min_value=2, max_value=100, value=30, step=1)
max_evals = st.sidebar.number_input("Max Evaluations", min_value=1000, max_value=100000, value=30000, step=1000)
runs = st.sidebar.number_input("Number of Runs", min_value=1, max_value=30, value=1, step=1)

# Function Selection (Shared across modes)
st.sidebar.subheader("Function Selection")
func_idx = st.sidebar.selectbox("Select Function", [f"F{i+1}" for i in range(30)])
f_index = int(func_idx[1:]) - 1
selected_func = functions.all_functions[f_index]

alg_options = ['Harmony Search (HS)', 'Adaptive HS', 'Hybrid HS', 'PSO', 'GA', 'Greedy', 'GSA', 'ABC']

if mode == "Single Algorithm Run":
    # Algorithm Selection
    selected_alg = st.sidebar.selectbox("Select Algorithm", alg_options)

    # Algorithm Specific Parameters
    params = {}
    if selected_alg == 'Harmony Search (HS)':
        st.sidebar.subheader("HS Parameters")
        params['hms'] = st.sidebar.number_input("HMS (Memory Size)", value=50)
        params['hmcr'] = st.sidebar.slider("HMCR", 0.0, 1.0, 0.9)
        params['par'] = st.sidebar.slider("PAR", 0.0, 1.0, 0.3)
        params['bw'] = st.sidebar.number_input("Bandwidth (BW)", value=0.01, format="%.4f")

    elif selected_alg == 'Adaptive HS':
        st.sidebar.subheader("Adaptive HS Parameters")
        params['hms'] = st.sidebar.number_input("HMS", value=50)
        params['hmcr_min'] = st.sidebar.slider("HMCR Min", 0.0, 1.0, 0.9)
        params['hmcr_max'] = st.sidebar.slider("HMCR Max", 0.0, 1.0, 0.99)
        params['par_min'] = st.sidebar.slider("PAR Min", 0.0, 1.0, 0.3)
        params['par_max'] = st.sidebar.slider("PAR Max", 0.0, 1.0, 0.99)
        params['bw_min'] = st.sidebar.number_input("BW Min", value=0.0001, format="%.5f")
        params['bw_max'] = st.sidebar.number_input("BW Max", value=0.5, format="%.5f")

    elif selected_alg == 'Hybrid HS':
        st.sidebar.subheader("Hybrid HS Parameters")
        params['hms'] = st.sidebar.number_input("HMS", value=50)
        params['hmcr'] = st.sidebar.slider("HMCR", 0.0, 1.0, 0.9)
        params['par'] = st.sidebar.slider("PAR", 0.0, 1.0, 0.3)
        params['bw'] = st.sidebar.number_input("BW", value=0.01, format="%.4f")
        params['local_search_rate'] = st.sidebar.slider("Local Search Rate", 0.0, 1.0, 0.01)

    elif selected_alg == 'PSO':
        st.sidebar.subheader("PSO Parameters")
        params['pop_size'] = st.sidebar.number_input("Population Size", value=30)
        params['w'] = st.sidebar.slider("Inertia Weight (w)", 0.0, 1.0, 0.729)
        params['c1'] = st.sidebar.number_input("C1", value=1.49445)
        params['c2'] = st.sidebar.number_input("C2", value=1.49445)

    elif selected_alg == 'GA':
        st.sidebar.subheader("GA Parameters")
        params['pop_size'] = st.sidebar.number_input("Population Size", value=50)
        params['mutation_rate'] = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.05)
        params['crossover_rate'] = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)

    elif selected_alg == 'GSA':
        st.sidebar.subheader("GSA Parameters")
        params['pop_size'] = st.sidebar.number_input("Population Size", value=50)
        params['G0'] = st.sidebar.number_input("G0", value=100)
        params['alpha'] = st.sidebar.number_input("Alpha", value=20)

    elif selected_alg == 'ABC':
        st.sidebar.subheader("ABC Parameters")
        params['pop_size'] = st.sidebar.number_input("Colony Size", value=50)
        params['limit'] = st.sidebar.number_input("Limit", value=100)

    # Run Button
    if st.sidebar.button("🚀 Run Optimization"):
        st.header(f"Results for {selected_alg} on {func_idx}")
        
        col1, col2 = st.columns(2)
        
        with st.spinner("Running optimization..."):
            all_histories = []
            best_fitnesses = []
            best_solutions = []
            
            progress_bar = st.progress(0)
            
            for r in range(runs):
                # Instantiate Algorithm
                if selected_alg == 'Harmony Search (HS)':
                    alg = HarmonySearch(selected_func, dim, max_evals, **params)
                elif selected_alg == 'Adaptive HS':
                    alg = AdaptiveHS(selected_func, dim, max_evals, **params)
                elif selected_alg == 'Hybrid HS':
                    alg = HybridHS(selected_func, dim, max_evals, **params)
                elif selected_alg == 'PSO':
                    alg = PSO(selected_func, dim, max_evals, **params)
                elif selected_alg == 'GA':
                    alg = GA(selected_func, dim, max_evals, **params)
                elif selected_alg == 'Greedy':
                    alg = Greedy(selected_func, dim, max_evals)
                elif selected_alg == 'GSA':
                    alg = GSA(selected_func, dim, max_evals, **params)
                elif selected_alg == 'ABC':
                    alg = ABC(selected_func, dim, max_evals, **params)
                else:
                    st.error(f"Unknown algorithm: {selected_alg}")
                    st.stop()
                
                best_fit, best_sol, history = alg.run()
                
                all_histories.append(history)
                best_fitnesses.append(best_fit)
                best_solutions.append(best_sol)
                
                progress_bar.progress((r + 1) / runs)
                
            # Statistics
            mean_best = np.mean(best_fitnesses)
            std_best = np.std(best_fitnesses)
            min_best = np.min(best_fitnesses)
            
            with col1:
                st.metric("Mean Best Fitness", f"{mean_best:.4e}")
                st.metric("Std Dev", f"{std_best:.4e}")
                
            with col2:
                st.metric("Best Run Fitness", f"{min_best:.4e}")
                
            # Convergence Plot
            st.subheader("Convergence Curve")
            
            # Pad histories to same length for averaging
            max_len = max(len(h) for h in all_histories)
            padded_histories = []
            for h in all_histories:
                if len(h) < max_len:
                    h = h + [h[-1]] * (max_len - len(h))
                padded_histories.append(h)
                
            avg_history = np.mean(padded_histories, axis=0)
            
            fig, ax = plt.subplots()
            ax.plot(range(len(avg_history)), avg_history)
            ax.set_xlabel('Evaluations')
            ax.set_ylabel('Fitness')
            ax.set_title('Convergence Curve')
            ax.set_yscale('log')
            ax.grid(True)
            
            st.pyplot(fig)
            
            # Best Solution
            st.subheader("Best Solution Found (from best run)")
            best_run_idx = np.argmin(best_fitnesses)
            st.code(str(best_solutions[best_run_idx]))

elif mode == "Algorithm Comparison":
    st.sidebar.info("Note: Default parameters are used for comparison.")
    selected_algs = st.sidebar.multiselect("Select Algorithms to Compare", alg_options, default=['Harmony Search (HS)', 'PSO'])
    
    if st.sidebar.button("🚀 Run Comparison"):
        if not selected_algs:
            st.error("Please select at least one algorithm.")
        else:
            st.header(f"Comparison on {func_idx}")
            
            results_data = []
            convergence_data = {}
            
            progress_bar = st.progress(0)
            total_steps = len(selected_algs) * runs
            current_step = 0
            
            with st.spinner("Running comparison..."):
                for alg_name in selected_algs:
                    fitness_values = []
                    histories = []
                    
                    for r in range(runs):
                        # Instantiate with default parameters
                        if alg_name == 'Harmony Search (HS)':
                            alg = HarmonySearch(selected_func, dim, max_evals)
                        elif alg_name == 'Adaptive HS':
                            alg = AdaptiveHS(selected_func, dim, max_evals)
                        elif alg_name == 'Hybrid HS':
                            alg = HybridHS(selected_func, dim, max_evals)
                        elif alg_name == 'PSO':
                            alg = PSO(selected_func, dim, max_evals)
                        elif alg_name == 'GA':
                            alg = GA(selected_func, dim, max_evals)
                        elif alg_name == 'Greedy':
                            alg = Greedy(selected_func, dim, max_evals)
                        elif alg_name == 'GSA':
                            alg = GSA(selected_func, dim, max_evals)
                        elif alg_name == 'ABC':
                            alg = ABC(selected_func, dim, max_evals)
                        else:
                            st.error(f"Unknown algorithm: {alg_name}")
                            st.stop()
                        
                        best_fit, _, history = alg.run()
                        fitness_values.append(best_fit)
                        histories.append(history)
                        
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)
                    
                    # Store results
                    results_data.append({
                        "Algorithm": alg_name,
                        "Mean Fitness": np.mean(fitness_values),
                        "Std Dev": np.std(fitness_values),
                        "Best Fitness": np.min(fitness_values)
                    })
                    
                    # Process convergence for this algo
                    max_len = max(len(h) for h in histories)
                    padded = []
                    for h in histories:
                        if len(h) < max_len:
                            h = h + [h[-1]] * (max_len - len(h))
                        padded.append(h)
                    convergence_data[alg_name] = np.mean(padded, axis=0)
            
            # Display Results Table
            st.subheader("Results Summary")
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results.style.format({
                "Mean Fitness": "{:.4e}",
                "Std Dev": "{:.4e}",
                "Best Fitness": "{:.4e}"
            }))
            
            # Display Comparison Plot
            st.subheader("Convergence Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for alg_name, history in convergence_data.items():
                ax.plot(range(len(history)), history, label=alg_name)
                
            ax.set_xlabel('Evaluations')
            ax.set_ylabel('Fitness')
            ax.set_title(f'Convergence Comparison on {func_idx}')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True)
            
            st.pyplot(fig)

else:
    st.info("Select parameters and click 'Run Optimization' to start.")
