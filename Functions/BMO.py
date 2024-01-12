#!/usr/bin/env python
# Created by "Thieu" at 11:10, 15/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalBMO(Optimizer):
    """
    The original version: Barnacles Mating Optimizer (BMO)

    Links:
        1. https://ieeexplore.ieee.org/document/8441097

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pl (int): [1, pop_size - 1], barnacle’s threshold

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, BMO
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
    >>> model = BMO.OriginalBMO(epoch=1000, pop_size=50, pl = 4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., Deb, S. and Coelho, L.D.S., 2018. Earthworm optimisation algorithm: a bio-inspired metaheuristic algorithm
    for global optimisation problems. International journal of bio-inspired computation, 12(1), pp.1-22.
    """

    def __init__(self, epoch=10000, pop_size=100, pl=5, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pl = self.validator.check_int("pl", pl, [1, self.pop_size-1])
        self.set_parameters(["epoch", "pop_size", "pl"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        k1 = self.generator.permutation(self.pop_size)
        k2 = self.generator.permutation(self.pop_size)
        temp = np.abs(k1 - k2)
        pop_new = []
        for idx in range(0, self.pop_size):
            if temp[idx] <= self.pl:
                p = self.generator.uniform(0, 1)
                pos_new = p * self.pop[k1[idx]].solution + (1 - p) * self.pop[k2[idx]].solution
            else:
                pos_new = self.generator.uniform(0, 1) * self.pop[k2[idx]].solution
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        self.pop = self.update_target_for_population(pop_new)


# aralık
from enumFunctions import Functions
from functions import selectFunction
import numpy as np
from mealpy import FloatVar, BMO
import argparse


parser = argparse.ArgumentParser(description='Process BMO arguments')
parser.add_argument('nVars', type=float, help='N Vars')
parser.add_argument('lb', type=float, help='Lower Bound')
parser.add_argument('ub', type=float, help='Upper Bound')
parser.add_argument('name', type=str, help='Name of work')
parser.add_argument('minmax', type=str, help='Min Max')
parser.add_argument('selectedBenchmarkFunction', type=str, help='Selected benchmark function')
parser.add_argument('epoch', type=float, help='Number Of Generations')
parser.add_argument('popSize', type=float, help='Population Size')
parser.add_argument('pl', type=float, help='pl')

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
pl = args.pl
if minmax == "Min":
    minmax = "max"
if minmax == "Max":
    minmax = "min"


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
from mealpy import FloatVar, BMO

bounds = FloatVar(lb=(lb,) * int(n_vars), ub=(ub,) * int(n_vars), name="delta")
problem_dict = {
    "bounds": bounds,
    "minmax": minmax,
    "obj_func": custom_objective_func
}

model = BMO.OriginalBMO(epoch=epoch, pop_size=popSize, pl = pl)
g_best = model.solve(problem_dict)