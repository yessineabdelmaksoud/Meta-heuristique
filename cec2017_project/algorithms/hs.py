import numpy as np

class HarmonySearch:
    def __init__(self, func, dim, max_evals=30000, hms=50, hmcr=0.9, par=0.3, bw=0.01, lb=-100, ub=100):
        self.func = func
        self.dim = dim
        self.max_evals = max_evals
        self.hms = hms  # Harmony Memory Size
        self.hmcr = hmcr  # Harmony Memory Consideration Rate
        self.par = par  # Pitch Adjusting Rate
        self.bw = bw  # Bandwidth
        self.lb = lb
        self.ub = ub
        
        self.HM = None
        self.fitness = None
        self.eval_count = 0
        self.best_fitness_history = []
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize(self):
        self.HM = np.random.uniform(self.lb, self.ub, (self.hms, self.dim))
        self.fitness = np.zeros(self.hms)
        for i in range(self.hms):
            if self.eval_count < self.max_evals:
                self.fitness[i] = self.func([self.HM[i]])[0]
                self.eval_count += 1
            else:
                self.fitness[i] = float('inf')
        
        self._update_best()

    def _update_best(self):
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_solution = self.HM[best_idx].copy()
        
    def improvise(self):
        new_harmony = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                # Memory consideration
                index = np.random.randint(0, self.hms)
                new_harmony[i] = self.HM[index, i]
                
                # Pitch adjustment
                if np.random.rand() < self.par:
                    new_harmony[i] += np.random.uniform(-1, 1) * self.bw
                    # Boundary check
                    new_harmony[i] = np.clip(new_harmony[i], self.lb, self.ub)
            else:
                # Random selection
                new_harmony[i] = np.random.uniform(self.lb, self.ub)
        
        return new_harmony

    def update_hm(self, new_harmony, new_fitness):
        worst_idx = np.argmax(self.fitness)
        if new_fitness < self.fitness[worst_idx]:
            self.HM[worst_idx] = new_harmony
            self.fitness[worst_idx] = new_fitness
            self._update_best()

    def run(self):
        self.initialize()
        
        while self.eval_count < self.max_evals:
            new_harmony = self.improvise()
            
            # Evaluate new harmony
            new_fitness = self.func([new_harmony])[0]
            self.eval_count += 1
            
            self.update_hm(new_harmony, new_fitness)
            self.best_fitness_history.append(self.best_fitness)
            
        return self.best_fitness, self.best_solution, self.best_fitness_history

class AdaptiveHS(HarmonySearch):
    def __init__(self, func, dim, max_evals=30000, hms=50, 
                 hmcr_min=0.9, hmcr_max=0.99, 
                 par_min=0.3, par_max=0.99, 
                 bw_min=0.0001, bw_max=0.5, 
                 lb=-100, ub=100):
        super().__init__(func, dim, max_evals, hms, hmcr_max, par_min, bw_max, lb, ub)
        self.hmcr_min = hmcr_min
        self.hmcr_max = hmcr_max
        self.par_min = par_min
        self.par_max = par_max
        self.bw_min = bw_min
        self.bw_max = bw_max

    def run(self):
        self.initialize()
        
        while self.eval_count < self.max_evals:
            # Dynamic parameter adjustment
            progress = self.eval_count / self.max_evals
            self.par = self.par_min + (self.par_max - self.par_min) * progress
            self.bw = self.bw_max * np.exp(np.log(self.bw_min / self.bw_max) * progress)
            
            new_harmony = self.improvise()
            
            new_fitness = self.func([new_harmony])[0]
            self.eval_count += 1
            
            self.update_hm(new_harmony, new_fitness)
            self.best_fitness_history.append(self.best_fitness)
            
        return self.best_fitness, self.best_solution, self.best_fitness_history

class HybridHS(HarmonySearch):
    def __init__(self, func, dim, max_evals=30000, hms=50, hmcr=0.9, par=0.3, bw=0.01, lb=-100, ub=100, local_search_rate=0.01):
        super().__init__(func, dim, max_evals, hms, hmcr, par, bw, lb, ub)
        self.local_search_rate = local_search_rate

    def local_search(self, solution):
        # Simple local search (e.g., small random perturbation)
        new_solution = solution + np.random.uniform(-self.bw, self.bw, self.dim)
        new_solution = np.clip(new_solution, self.lb, self.ub)
        return new_solution

    def run(self):
        self.initialize()
        
        while self.eval_count < self.max_evals:
            new_harmony = self.improvise()
            
            # Apply local search with some probability
            if np.random.rand() < self.local_search_rate:
                new_harmony = self.local_search(new_harmony)

            new_fitness = self.func([new_harmony])[0]
            self.eval_count += 1
            
            self.update_hm(new_harmony, new_fitness)
            self.best_fitness_history.append(self.best_fitness)
            
        return self.best_fitness, self.best_solution, self.best_fitness_history
