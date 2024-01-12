#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalEO(Optimizer):
    """
    The original version of: Equilibrium Optimizer (EO)

    Links:
        1. https://doi.org/10.1016/j.knosys.2019.105190
        2. https://www.mathworks.com/matlabcentral/fileexchange/73352-equilibrium-optimizer-eo

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, EO
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(n_vars=30, lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = EO.OriginalEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Faramarzi, A., Heidarinejad, M., Stephens, B. and Mirjalili, S., 2020. Equilibrium optimizer: A novel
    optimization algorithm. Knowledge-Based Systems, 191, p.105190.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        ## Fixed parameter proposed by authors
        self.V = 1
        self.a1 = 2
        self.a2 = 1
        self.GP = 0.5

    def make_equilibrium_pool__(self, list_equilibrium=None):
        pos_list = [agent.solution for agent in list_equilibrium]
        pos_mean = np.mean(pos_list, axis=0)
        pos_mean = self.correct_solution(pos_mean)
        agent = self.generate_agent(pos_mean)
        list_equilibrium.append(agent)
        return list_equilibrium

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # ---------------- Memory saving-------------------  make equilibrium pool
        _, c_eq_list, _ = self.get_special_agents(self.pop, n_best=4, minmax=self.problem.minmax)
        c_pool = self.make_equilibrium_pool__(c_eq_list)
        # Eq. 9
        t = (1 - epoch / self.epoch) ** (self.a2 * epoch / self.epoch)
        pop_new = []
        for idx in range(0, self.pop_size):
            lamda = self.generator.uniform(0, 1, self.problem.n_dims)  # lambda in Eq. 11
            r = self.generator.uniform(0, 1, self.problem.n_dims)  # r in Eq. 11
            c_eq = c_pool[self.generator.integers(0, len(c_pool))].solution  # random selection 1 of candidate from the pool
            f = self.a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1.0)  # Eq. 11
            r1 = self.generator.uniform()
            r2 = self.generator.uniform()  # r1, r2 in Eq. 15
            gcp = 0.5 * r1 * np.ones(self.problem.n_dims) * (r2 >= self.GP)  # Eq. 15
            g0 = gcp * (c_eq - lamda * self.pop[idx].solution)  # Eq. 14
            g = g0 * f  # Eq. 13
            pos_new = c_eq + (self.pop[idx].solution - c_eq) * f + (g * self.V / lamda) * (1.0 - f)  # Eq. 16
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
from mealpy import FloatVar, EO
import argparse


parser = argparse.ArgumentParser(description='Process EO arguments')
parser.add_argument('nVars', type=float, help='N Vars')
parser.add_argument('lb', type=float, help='Lower Bound')
parser.add_argument('ub', type=float, help='Upper Bound')
parser.add_argument('selectedBenchmarkFunction', type=str, help='Selected benchmark function')
parser.add_argument('epoch', type=float, help='Number Of Generations')
parser.add_argument('populationSize', type=float, help='Population Size')
parser.add_argument('minmax', type=str, help='Min Max')

# Argümanları ayrıştır
args = parser.parse_args()

# Değişkenlere atama
n_vars = args.nVars
lb = args.lb
ub = args.ub
benchmarkFunction = args.selectedBenchmarkFunction
epoch = args.epoch
populationSize = args.populationSize
minmax = args.minmax
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

selected_function_name = benchmarkFunction
custom_objective_func = select_function_by_name(selected_function_name)

import numpy as np
from mealpy import FloatVar, EO

bounds = FloatVar(lb=(lb,) * int(n_vars), ub=(ub,) * int(n_vars), name="delta")
problem_dict = {
    "bounds": bounds,
    "minmax": minmax,
    "obj_func": custom_objective_func
}

model = OriginalEO(epoch=epoch, pop_size=populationSize)
g_best = model.solve(problem_dict)