#!/usr/bin/env python
# Created by "Thieu" at 19:27, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalEP(Optimizer):
    """
    The original version of: Evolutionary Programming (EP)

    Links:
        1. https://www.cleveralgorithms.com/nature-inspired/evolution/evolutionary_programming.html
        2. https://github.com/clever-algorithms/CleverAlgorithms

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + bout_size (float): [0.05, 0.2], percentage of child agents implement tournament selection

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, EP
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
    >>> model = EP.OriginalEP(epoch=1000, pop_size=50, bout_size = 0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yao, X., Liu, Y. and Lin, G., 1999. Evolutionary programming made faster.
    IEEE Transactions on Evolutionary computation, 3(2), pp.82-102.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, bout_size: float = 0.05, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            bout_size (float): percentage of child agents implement tournament selection
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.bout_size = self.validator.check_float("bout_size", bout_size, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "bout_size"])
        self.sort_flag = True

    def initialize_variables(self):
        self.n_bout_size = int(self.bout_size * self.pop_size)
        self.distance = 0.05 * (self.problem.ub - self.problem.lb)

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        strategy = self.generator.uniform(0, self.distance, self.problem.n_dims)
        times_win = 0
        return Agent(solution=solution, strategy=strategy, win=times_win)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        child = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx].solution + self.pop[idx].strategy * self.generator.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            s_old = self.pop[idx].strategy + self.generator.normal(0, 1.0, self.problem.n_dims) * np.abs(self.pop[idx].strategy) ** 0.5
            agent.update(solution=pos_new, strategy=s_old, win=0)
            child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                child[-1].target = self.get_target(pos_new)
        child = self.update_target_for_population(child)
        # Update the global best
        children = self.get_sorted_population(child, self.problem.minmax)
        pop = children + self.pop
        for i in range(0, len(pop)):
            ## Tournament winner (Tried with bout_size times)
            for idx in range(0, self.n_bout_size):
                rand_idx = self.generator.integers(0, len(pop))
                if self.compare_target(pop[i].target, pop[rand_idx].target, self.problem.minmax):
                    pop[i].win += 1
                else:
                    pop[rand_idx].win += 1
        pop = sorted(pop, key=lambda agent: agent.win, reverse=True)
        self.pop = pop[:self.pop_size]


class LevyEP(OriginalEP):
    """
    The developed Levy-flight version: Evolutionary Programming (LevyEP)

    Notes:
        + Levy-flight is applied to EP, flow and some equations is changed.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + bout_size (float): [0.05, 0.2], percentage of child agents implement tournament selection

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, EP
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
    >>> model = EP.LevyEP(epoch=1000, pop_size=50, bout_size = 0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, bout_size: float = 0.05, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            bout_size (float): percentage of child agents implement tournament selection
        """
        super().__init__(epoch, pop_size, bout_size, **kwargs)
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        child = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx].solution + self.pop[idx].strategy * self.generator.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.correct_solution(pos_new)
            s_old = self.pop[idx].strategy + self.generator.normal(0, 1.0, self.problem.n_dims) * np.abs(self.pop[idx].strategy) ** 0.5
            agent = self.generate_empty_agent(pos_new)
            agent.update(solution=pos_new, strategy=s_old, win=0)
            child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                child[-1].target = self.get_target(pos_new)
        child = self.update_target_for_population(child)
        # Update the global best
        children = self.get_sorted_population(child, self.problem.minmax)
        pop = children + self.pop
        for i in range(0, len(pop)):
            ## Tournament winner (Tried with bout_size times)
            for idx in range(0, self.n_bout_size):
                rand_idx = self.generator.integers(0, len(pop))
                if self.compare_target(pop[i].target, pop[rand_idx].target, self.problem.minmax):
                    pop[i].win += 1
                else:
                    pop[rand_idx].win += 1
        ## Keep the top population, but 50% of left population will make a comeback an take the good position
        pop = sorted(pop, key=lambda agent: agent.win, reverse=True)
        pop_new = pop[:self.pop_size]
        pop_left = pop[self.pop_size:]
        ## Choice random 50% of population left
        pop_comeback = []
        idx_list = self.generator.choice(range(0, len(pop_left)), int(0.5 * len(pop_left)), replace=False)
        for idx in idx_list:
            pos_new = pop_left[idx].solution + self.get_levy_flight_step(multiplier=0.01, size=self.problem.n_dims, case=0)
            pos_new = self.correct_solution(pos_new)
            strategy = self.distance = 0.05 * (self.problem.ub - self.problem.lb)
            agent = self.generate_empty_agent(pos_new)
            agent.update(solution=pos_new, strategy=strategy, win=0)
            pop_comeback.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_comeback[-1].target = self.get_target(pos_new)
        pop_comeback = self.update_target_for_population(pop_comeback)
        self.pop = self.get_sorted_and_trimmed_population(pop_new + pop_comeback, self.pop_size, self.problem.minmax)

# aralık
from enumFunctions import Functions
from functions import selectFunction
import numpy as np
from mealpy import FloatVar, EP
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
parser.add_argument('boutSize', type=float, help='boutSize')
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
boutSize = args.boutSize


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
from mealpy import FloatVar, EP

bounds = FloatVar(lb=(lb,) * int(n_vars), ub=(ub,) * int(n_vars), name="delta")
problem_dict = {
    "bounds": bounds,
    "minmax": minmax,
    "obj_func": custom_objective_func
}

model = EP.OriginalEP(epoch=epoch, pop_size=popSize, bout_size = boutSize)
g_best = model.solve(problem_dict)