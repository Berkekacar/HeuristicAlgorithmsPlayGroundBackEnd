#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalGWO(Optimizer):
    """
    The original version of: Grey Wolf Optimizer (GWO)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2013.12.007
        2. https://www.mathworks.com/matlabcentral/fileexchange/44974-grey-wolf-optimizer-gwo?s_tid=FX_rc3_behav

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GWO
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
    >>> model = GWO.OriginalGWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., Mirjalili, S.M. and Lewis, A., 2014. Grey wolf optimizer. Advances in engineering software, 69, pp.46-61.
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

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # linearly decreased from 2 to 0
        a = 2 - 2. * epoch / self.epoch
        _, list_best, _ = self.get_special_agents(self.pop, n_best=3, minmax=self.problem.minmax)
        pop_new = []
        for idx in range(0, self.pop_size):
            A1 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A2 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            A3 = a * (2 * self.generator.random(self.problem.n_dims) - 1)
            C1 = 2 * self.generator.random(self.problem.n_dims)
            C2 = 2 * self.generator.random(self.problem.n_dims)
            C3 = 2 * self.generator.random(self.problem.n_dims)
            X1 = list_best[0].solution - A1 * np.abs(C1 * list_best[0].solution - self.pop[idx].solution)
            X2 = list_best[1].solution - A2 * np.abs(C2 * list_best[1].solution - self.pop[idx].solution)
            X3 = list_best[2].solution - A3 * np.abs(C3 * list_best[2].solution - self.pop[idx].solution)
            pos_new = (X1 + X2 + X3) / 3.0
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)

from enumFunctions import Functions
from functions import selectFunction
import numpy as np
from mealpy import FloatVar, GWO
import argparse


parser = argparse.ArgumentParser(description='Process GWO arguments')
parser.add_argument('populationSize', type=float, help='Population Size')
parser.add_argument('lb', type=float, help='Lower Bound')
parser.add_argument('ub', type=float, help='Upper Bound')
parser.add_argument('nVars', type=float, help='Dimension')
parser.add_argument('minmax', type=str, help='Min Max')
parser.add_argument('numberOfGenerations', type=int, help='Number of Generations')
parser.add_argument('decreaseFrom',type=int,help='Decrease From')
parser.add_argument('selectedBenchmarkFunction',type=str,help='Selected Benchmark Function')

# Argümanları ayrıştır
args = parser.parse_args()

# Değişkenlere atama
populationSize = args.populationSize
lb = args.lb
ub = args.ub
nVars = args.nVars
minmax = args.minmax
numberOfGenerations = args.numberOfGenerations
decreaseFrom = args.decreaseFrom
selectedBenchmarkFunction = args.selectedBenchmarkFunction
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
from mealpy import FloatVar, GWO

bounds = FloatVar(lb=(lb,) * int(nVars), ub=(ub,) * int(nVars), name="delta")
problem_dict = {
    "bounds": bounds,
    "minmax": minmax,
    "obj_func": custom_objective_func
}

model = GWO.OriginalGWO(epoch=numberOfGenerations, pop_size=populationSize)
g_best = model.solve(problem_dict)