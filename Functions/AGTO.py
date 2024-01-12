#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalAGTO(Optimizer):
    """
    The original version of: Artificial Gorilla Troops Optimization (AGTO)

    Links:
        1. https://doi.org/10.1002/int.22535
        2. https://www.mathworks.com/matlabcentral/fileexchange/95953-artificial-gorilla-troops-optimizer

    Notes (parameters):
        1. p1 (float): the probability of transition in exploration phase (p in the paper), default = 0.03
        2. p2 (float): the probability of transition in exploitation phase (w in the paper), default = 0.8
        3. beta (float): coefficient in updating equation, should be in [-5.0, 5.0], default = 3.0

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, AGTO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = AGTO.OriginalAGTO(epoch=1000, pop_size=50, p1=0.03, p2=0.8, beta=3.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdollahzadeh, B., Soleimanian Gharehchopogh, F., & Mirjalili, S. (2021). Artificial gorilla troops optimizer: a new
    nature‐inspired metaheuristic algorithm for global optimization problems. International Journal of Intelligent Systems, 36(10), 5887-5958.
    """
    def __init__(self, epoch: int = 10000, pop_size: int = 100, p1: float = 0.03, p2: float = 0.8, beta: float = 3.0, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p1 = self.validator.check_float("p1", p1, (0, 1))      # p in the paper
        self.p2 = self.validator.check_float("p2", p2, (0, 1))      # w in the paper
        self.beta = self.validator.check_float("beta", beta, [-10.0, 10.0])
        self.set_parameters(["epoch", "pop_size", "p1", "p2", "beta"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        a = (np.cos(2*self.generator.random())+1) * (1 - epoch/self.epoch)
        c = a * (2 * self.generator.random() - 1)
        ## Exploration
        pop_new = []
        for idx in range(0, self.pop_size):
            if self.generator.random() < self.p1:
                pos_new = self.problem.generate_solution()
            else:
                if self.generator.random() >= 0.5:
                    z = self.generator.uniform(-a, a, self.problem.n_dims)
                    rand_idx = self.generator.integers(0, self.pop_size)
                    pos_new = (self.generator.random() - a) * self.pop[rand_idx].solution + c * z * self.pop[idx].solution
                else:
                    id1, id2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                    pos_new = self.pop[idx].solution - c*(c*self.pop[idx].solution - self.pop[id1].solution) + \
                        self.generator.random() * (self.pop[idx].solution - self.pop[id2].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)
        _, self.g_best = self.update_global_best_agent(self.pop, save=False)

        pos_list = np.array([agent.solution for agent in self.pop])
        ## Exploitation
        pop_new = []
        for idx in range(0, self.pop_size):
            if a >= self.p2:
                g = 2 ** c
                delta = (np.abs(np.mean(pos_list, axis=0)) ** g) ** (1.0 / g)
                pos_new = c*delta*(self.pop[idx].solution - self.g_best.solution) + self.pop[idx].solution
            else:
                if self.generator.random() >= 0.5:
                    h = self.generator.normal(0, 1, self.problem.n_dims)
                else:
                    h = self.generator.normal(0, 1)
                r1 = self.generator.random()
                pos_new = self.g_best.solution - (2*r1-1)*(self.g_best.solution - self.pop[idx].solution) * (self.beta * h)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

# aralık
from enumFunctions import Functions
from functions import selectFunction
import numpy as np
from mealpy import FloatVar, AGTO
import argparse


parser = argparse.ArgumentParser(description='Process AGTO arguments')
parser.add_argument('nVars', type=float, help='N Vars')
parser.add_argument('lb', type=float, help='Lower Bound')
parser.add_argument('ub', type=float, help='Upper Bound')
parser.add_argument('selectedBenchmarkFunction', type=str, help='Selected benchmark function')
parser.add_argument('minmax', type=str, help='Min Max')
parser.add_argument('epoch', type=float, help='Number Of Generations')
parser.add_argument('populationSize', type=float, help='Population Size')
parser.add_argument('p1', type=float, help='p1')
parser.add_argument('p2', type=float, help='p2')
parser.add_argument('beta', type=float, help='beta')

# Argümanları ayrıştır
args = parser.parse_args()

# Değişkenlere atama
nVars = args.nVars
lb = args.lb
ub = args.ub
selectedBenchmarkFunction = args.selectedBenchmarkFunction
minmax = args.minmax
epoch = args.epoch
populationSize = args.populationSize
p1 = args.p1
p2 = args.p2
beta = args.p2
if minmax == "Min":
    minmax = "min"
if minmax == "Max":
    minmax = "max"


def select_function_by_name(func_name, *args):
    function_mapping = {
    'Ackley': selectFunction(Functions.ackley),
    'Dixonprice': selectFunction(Functions.dixonprice),
    'Griewank': selectFunction(Functions.griewank),
    'Michalewicz': selectFunction(Functions.michalewicz),
    'Perm': selectFunction(Functions.perm),
    'Powell': selectFunction(Functions.powell),
    'Powersum': selectFunction(Functions.powersum),
    'Rastrigin': selectFunction(Functions.rastrigin),
    'Rosenbrock': selectFunction(Functions.rosenbrock),
    'Schwefel': selectFunction(Functions.schwefel),
    'Sphere': selectFunction(Functions.sphere),
    'Sum2': selectFunction(Functions.sum2),
    'Trid': selectFunction(Functions.trid),
    'Zakharov': selectFunction(Functions.zakharov),
    'ellipse': selectFunction(Functions.ellipse),
    'nesterov': selectFunction(Functions.nesterov),
    'saddle': selectFunction(Functions.saddle) 
}
    
    selected_function = function_mapping.get(func_name)
    
    if selected_function:
        def custom_objective_function(solution):
            return selected_function(solution, *args)  
        
        return custom_objective_function
    else:
        raise ValueError("Belirtilen fonksiyon adı geçerli değil.")

selected_function_name = selectedBenchmarkFunction
custom_objective_func = select_function_by_name(selected_function_name)

bounds = FloatVar(lb=(lb,) * int(nVars), ub=(ub,) * int(nVars), name="delta")
problem_dict = {
    "bounds": bounds,
    "minmax": minmax,
    "obj_func": custom_objective_func
}

model = AGTO.OriginalAGTO(epoch=epoch, pop_size=populationSize, p1=p1, p2=p2, beta=beta)
g_best = model.solve(problem_dict)