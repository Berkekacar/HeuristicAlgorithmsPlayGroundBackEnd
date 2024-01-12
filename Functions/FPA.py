#!/usr/bin/env python
# Created by "Thieu" at 19:34, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalFPA(Optimizer):
    """
    The original version of: Flower Pollination Algorithm (FPA)

    Links:
        1. https://doi.org/10.1007/978-3-642-32894-7_27

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_s (float): [0.5, 0.95], switch probability, default = 0.8
        + levy_multiplier: [0.0001, 1000], mutiplier factor of Levy-flight trajectory, depends on the problem

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, FPA
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
    >>> model = FPA.OriginalFPA(epoch=1000, pop_size=50, p_s = 0.8, levy_multiplier = 0.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S., 2012, September. Flower pollination algorithm for global optimization. In International
    conference on unconventional computing and natural computation (pp. 240-249). Springer, Berlin, Heidelberg.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, p_s: float = 0.8, levy_multiplier: float = 0.1, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_s (float): switch probability, default = 0.8
            levy_multiplier (float): multiplier factor of Levy-flight trajectory, default = 0.2
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.p_s = self.validator.check_float("p_s", p_s, (0, 1.0))
        self.levy_multiplier = self.validator.check_float("levy_multiplier", levy_multiplier, (-10000, 10000))
        self.set_parameters(["epoch", "pop_size", "p_s", "levy_multiplier"])
        self.sort_flag = False

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        random_pos = self.problem.generate_solution()
        return np.where(condition, solution, random_pos)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = []
        for idx in range(0, self.pop_size):
            if self.generator.uniform() < self.p_s:
                levy = self.get_levy_flight_step(multiplier=self.levy_multiplier, size=self.problem.n_dims, case=-1)
                pos_new = self.pop[idx].solution + 1.0 / np.sqrt(epoch) * levy * (self.pop[idx].solution - self.g_best.solution)
            else:
                id1, id2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = self.pop[idx].solution + self.generator.uniform() * (self.pop[id1].solution - self.pop[id2].solution)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(self.pop, pop, self.problem.minmax)

# aralık
from enumFunctions import Functions
from functions import selectFunction
import numpy as np
from mealpy import FloatVar, FPA
import argparse


parser = argparse.ArgumentParser(description='Process CRO arguments')
parser.add_argument('nVars', type=float, help='N Vars')
parser.add_argument('lb', type=float, help='Lower Bound')
parser.add_argument('ub', type=float, help='Upper Bound')
parser.add_argument('name', type=str, help='Name of work')
parser.add_argument('minmax', type=str, help='Min Max')
parser.add_argument('selectedBenchmarkFunction', type=str, help='Selected benchmark function')
parser.add_argument('epoch', type=float, help='Number Of Generations')
parser.add_argument('popSize', type=float, help='Population Size')
parser.add_argument('pS', type=float, help='pS')
parser.add_argument('levyMultiplier', type=float, help='levyMultiplier')
# Argümanları ayrıştır
args = parser.parse_args()

# Değişkenlere atama
n_vars = args.nVars
lb = args.lb
ub = args.ub
name = args.name
minmax = args.minmax
selectedBenchmarkFunction = args.selectedBenchmarkFunction
epoch = args.epoch
popSize = args.popSize
pS = args.pS
levyMultiplier = args.levyMultiplier


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

import numpy as np
from mealpy import FloatVar, FPA

bounds = FloatVar(lb=(lb,) * int(n_vars), ub=(ub,) * int(n_vars), name="delta")
problem_dict = {
    "bounds": bounds,
    "minmax": minmax,
    "obj_func": custom_objective_func
}

model = FPA.OriginalFPA(epoch=epoch, pop_size=popSize, p_s = pS, levy_multiplier = levyMultiplier)
g_best = model.solve(problem_dict)