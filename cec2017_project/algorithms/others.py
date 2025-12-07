import numpy as np

class Algorithm:
    def __init__(self, func, dim, max_evals=30000, lb=-100, ub=100):
        self.func = func
        self.dim = dim
        self.max_evals = max_evals
        self.lb = lb
        self.ub = ub
        self.eval_count = 0
        self.best_fitness_history = []
        self.best_solution = None
        self.best_fitness = float('inf')

    def evaluate(self, solution):
        if self.eval_count >= self.max_evals:
            return float('inf')
        
        fitness = self.func([solution])[0]
        self.eval_count += 1
        
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_solution = solution.copy()
            
        self.best_fitness_history.append(self.best_fitness)
        return fitness

class PSO(Algorithm):
    def __init__(self, func, dim, max_evals=30000, pop_size=30, w=0.729, c1=1.49445, c2=1.49445, lb=-100, ub=100):
        super().__init__(func, dim, max_evals, lb, ub)
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def run(self):
        # Initialization
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        V = np.zeros((self.pop_size, self.dim))
        P = X.copy()
        P_fitness = np.array([self.evaluate(x) for x in X])
        
        # Global best is already updated in evaluate() via self.best_fitness
        # But we need the index for the loop
        g_best = self.best_solution.copy()
        
        while self.eval_count < self.max_evals:
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            
            V = self.w * V + self.c1 * r1 * (P - X) + self.c2 * r2 * (g_best - X)
            X = X + V
            X = np.clip(X, self.lb, self.ub)
            
            for i in range(self.pop_size):
                if self.eval_count >= self.max_evals:
                    break
                
                fitness = self.evaluate(X[i])
                
                if fitness < P_fitness[i]:
                    P[i] = X[i].copy()
                    P_fitness[i] = fitness
                    
            g_best = self.best_solution.copy()
            
        return self.best_fitness, self.best_solution, self.best_fitness_history

class GA(Algorithm):
    def __init__(self, func, dim, max_evals=30000, pop_size=50, mutation_rate=0.05, crossover_rate=0.8, lb=-100, ub=100):
        super().__init__(func, dim, max_evals, lb, ub)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def run(self):
        # Initialization
        population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitnesses = np.array([self.evaluate(ind) for ind in population])
        
        while self.eval_count < self.max_evals:
            new_population = []
            
            # Elitism: keep the best
            best_idx = np.argmin(fitnesses)
            new_population.append(population[best_idx].copy())
            
            while len(new_population) < self.pop_size:
                if self.eval_count >= self.max_evals:
                    break

                # Selection (Tournament)
                p1 = population[np.random.randint(0, self.pop_size)]
                p2 = population[np.random.randint(0, self.pop_size)]
                parent1 = p1 if self.evaluate(p1) < self.evaluate(p2) else p2 # Note: this re-evaluates, which is bad for budget. 
                # Let's use stored fitnesses for selection to save evaluations
                
                idx1, idx2 = np.random.randint(0, self.pop_size, 2)
                parent1 = population[idx1] if fitnesses[idx1] < fitnesses[idx2] else population[idx2]
                
                idx3, idx4 = np.random.randint(0, self.pop_size, 2)
                parent2 = population[idx3] if fitnesses[idx3] < fitnesses[idx4] else population[idx4]
                
                # Crossover
                if np.random.rand() < self.crossover_rate:
                    point = np.random.randint(1, self.dim)
                    child = np.concatenate((parent1[:point], parent2[point:]))
                else:
                    child = parent1.copy()
                
                # Mutation
                if np.random.rand() < self.mutation_rate:
                    mutation_point = np.random.randint(0, self.dim)
                    child[mutation_point] += np.random.uniform(-1, 1)
                    child[mutation_point] = np.clip(child[mutation_point], self.lb, self.ub)
                
                new_population.append(child)
            
            population = np.array(new_population)[:self.pop_size]
            # Evaluate new population
            for i in range(len(population)):
                 if self.eval_count < self.max_evals:
                    fitnesses[i] = self.evaluate(population[i])
            
        return self.best_fitness, self.best_solution, self.best_fitness_history

class Greedy(Algorithm):
    def __init__(self, func, dim, max_evals=30000, lb=-100, ub=100):
        super().__init__(func, dim, max_evals, lb, ub)

    def run(self):
        # Simple Random Search / Greedy
        # Generate random solution, if better keep it
        
        current_solution = np.random.uniform(self.lb, self.ub, self.dim)
        current_fitness = self.evaluate(current_solution)
        
        while self.eval_count < self.max_evals:
            new_solution = np.random.uniform(self.lb, self.ub, self.dim)
            new_fitness = self.evaluate(new_solution)
            
            if new_fitness < current_fitness:
                current_solution = new_solution
                current_fitness = new_fitness
                
        return self.best_fitness, self.best_solution, self.best_fitness_history

class GSA(Algorithm):
    def __init__(self, func, dim, max_evals=30000, pop_size=50, G0=100, alpha=20, lb=-100, ub=100):
        super().__init__(func, dim, max_evals, lb, ub)
        self.pop_size = pop_size
        self.G0 = G0
        self.alpha = alpha

    def run(self):
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        V = np.zeros((self.pop_size, self.dim))
        fitness = np.array([self.evaluate(x) for x in X])
        
        while self.eval_count < self.max_evals:
            t = self.eval_count # Approximation of iteration, but GSA uses time t. 
            # Let's use iteration count. We need to track iterations.
            # Since we don't have explicit iterations, we can use eval_count / pop_size
            iteration = self.eval_count / self.pop_size
            max_iter = self.max_evals / self.pop_size
            
            G = self.G0 * np.exp(-self.alpha * iteration / max_iter)
            
            best = np.min(fitness)
            worst = np.max(fitness)
            
            epsilon = 1e-10
            M = (fitness - worst) / (best - worst + epsilon)
            M = M / np.sum(M)
            
            F = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if i != j:
                        R = np.linalg.norm(X[i] - X[j])
                        force = G * (M[i] * M[j]) / (R + epsilon) * (X[j] - X[i])
                        F[i] += np.random.rand() * force
            
            a = F / (M[:, None] + epsilon)
            V = np.random.rand(self.pop_size, self.dim) * V + a
            X = X + V
            X = np.clip(X, self.lb, self.ub)
            
            for i in range(self.pop_size):
                if self.eval_count < self.max_evals:
                    fitness[i] = self.evaluate(X[i])
                    
        return self.best_fitness, self.best_solution, self.best_fitness_history

class ABC(Algorithm):
    def __init__(self, func, dim, max_evals=30000, pop_size=50, limit=100, lb=-100, ub=100):
        super().__init__(func, dim, max_evals, lb, ub)
        self.pop_size = pop_size # Colony size
        self.n_food = pop_size // 2
        self.limit = limit

    def run(self):
        # Initialization
        foods = np.random.uniform(self.lb, self.ub, (self.n_food, self.dim))
        fitness = np.array([self.evaluate(x) for x in foods])
        trial = np.zeros(self.n_food)
        
        while self.eval_count < self.max_evals:
            # Employed Bees
            for i in range(self.n_food):
                if self.eval_count >= self.max_evals: break
                
                k = np.random.randint(0, self.n_food)
                while k == i: k = np.random.randint(0, self.n_food)
                
                phi = np.random.uniform(-1, 1, self.dim)
                new_solution = foods[i] + phi * (foods[i] - foods[k])
                new_solution = np.clip(new_solution, self.lb, self.ub)
                
                new_fitness = self.evaluate(new_solution)
                
                if new_fitness < fitness[i]:
                    foods[i] = new_solution
                    fitness[i] = new_fitness
                    trial[i] = 0
                else:
                    trial[i] += 1
            
            # Onlooker Bees
            # Calculate probabilities
            if np.max(fitness) != np.min(fitness):
                 probs = (np.max(fitness) - fitness) / (np.max(fitness) - np.min(fitness) + 1e-10)
            else:
                 probs = np.ones(self.n_food)
            probs = probs / np.sum(probs)
            
            for i in range(self.n_food): # Number of onlookers = n_food
                if self.eval_count >= self.max_evals: break
                
                # Select food source based on probability
                selected = np.random.choice(self.n_food, p=probs)
                
                k = np.random.randint(0, self.n_food)
                while k == selected: k = np.random.randint(0, self.n_food)
                
                phi = np.random.uniform(-1, 1, self.dim)
                new_solution = foods[selected] + phi * (foods[selected] - foods[k])
                new_solution = np.clip(new_solution, self.lb, self.ub)
                
                new_fitness = self.evaluate(new_solution)
                
                if new_fitness < fitness[selected]:
                    foods[selected] = new_solution
                    fitness[selected] = new_fitness
                    trial[selected] = 0
                else:
                    trial[selected] += 1
            
            # Scout Bees
            best_idx = np.argmin(fitness)
            for i in range(self.n_food):
                if trial[i] > self.limit and i != best_idx:
                    if self.eval_count >= self.max_evals: break
                    foods[i] = np.random.uniform(self.lb, self.ub, self.dim)
                    fitness[i] = self.evaluate(foods[i])
                    trial[i] = 0
                    
        return self.best_fitness, self.best_solution, self.best_fitness_history
