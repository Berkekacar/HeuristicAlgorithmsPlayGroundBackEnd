#!/usr/bin/env python
# Created by "Thieu" at 17:23, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class OriginalTSA(Optimizer):
    """
    The original version: Tunicate Swarm Algorithm (TSA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0952197620300385?via%3Dihub
        2. https://www.mathworks.com/matlabcentral/fileexchange/75182-tunicate-swarm-algorithm-tsa

    Notes:
        1. This algorithm has some limitations
        2. The paper has several wrong equations in algorithm
        3. The implementation in Matlab code has some difference to the paper
        4. This algorithm shares some similarities with the Barnacles Mating Optimizer (BMO)

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, TSA
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
    >>> model = TSA.OriginalTSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kaur, S., Awasthi, L. K., Sangal, A. L., & Dhiman, G. (2020). Tunicate Swarm Algorithm: A new bio-inspired
    based metaheuristic paradigm for global optimization. Engineering Applications of Artificial Intelligence, 90, 103541.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
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
        pmin, pmax = 1, 4
        pop_new = []
        for idx in range(0, self.pop_size):
            c3 = self.generator.random(self.problem.n_dims)
            c2 = self.generator.random(self.problem.n_dims)
            c1 = self.generator.random(self.problem.n_dims)
            M = np.fix(pmin + self.generator.random() * (pmax - pmin))
            A = (c2 + c3 - 2 * c1) / M
            t1 = self.g_best.solution + A * np.abs(self.g_best.solution - c2 * self.pop[idx].solution)
            t2 = self.g_best.solution - A * np.abs(self.g_best.solution - c2 * self.pop[idx].solution)
            pos_new = np.where(c3 >= 0.5, t1, t2)
            if idx != 0:
                pos_new = (pos_new + self.pop[idx-1].solution) / 2
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        self.pop = self.update_target_for_population(pop_new)

from enumFunctions import Functions
from functions import selectFunction
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Process PSO arguments')
parser.add_argument('nVars', type=int, help='N Vars')
parser.add_argument('lb', type=float, help='Lower Bound')
parser.add_argument('ub', type=float, help='Upper Bound')
parser.add_argument('name',type=str, help='name')
parser.add_argument('minmax', type=str, help='Min Max')
parser.add_argument('selectedBenchmarkFunction', type=str, help='Selected benchmark function')
parser.add_argument('epoch', type=int, help='Epoch')
parser.add_argument('popSize', type=int, help='Population Size')
# Argümanları ayrıştır
args = parser.parse_args()

# Değişkenlere atama
nVars = args.nVars
lb = args.lb
ub = args.ub
name = args.name
minmax = args.minmax
selectedBenchmarkFunction = args.selectedBenchmarkFunction
epoch = args.epoch
popSize = args.popSize

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
from mealpy import FloatVar, TSA

def objective_function(solution):
    return np.sum(solution**2)
    
bounds = FloatVar(lb=(lb,) * int(nVars), ub=(ub,) * int(nVars), name="delta")
problem_dict = {
    "bounds": bounds,
    "minmax": minmax,
    "obj_func": custom_objective_func
}
    
model = TSA.OriginalTSA(epoch=epoch, pop_size=popSize)
g_best = model.solve(problem_dict)