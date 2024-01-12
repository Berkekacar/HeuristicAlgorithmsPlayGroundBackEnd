#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalPSO(Optimizer):
    """
    The original version of: Particle Swarm Optimization (PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [1, 3], local coefficient, default = 2.05
        + c2 (float): [1, 3], global coefficient, default = 2.05
        + w (float): (0., 1.0), Weight min of bird, default = 0.4

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, PSO
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
    >>> model = PSO.OriginalPSO(epoch=1000, pop_size=50, c1=2.05, c2=20.5, w=0.4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kennedy, J. and Eberhart, R., 1995, November. Particle swarm optimization. In Proceedings of
    ICNN'95-international conference on neural networks (Vol. 4, pp. 1942-1948). IEEE.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, c1: float = 2.05, c2: float = 2.05, w: float = 0.4, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c1: [0-2] local coefficient
            c2: [0-2] global coefficient
            w_min: Weight min of bird, default = 0.4
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.w = self.validator.check_float("w", w, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "c1", "c2", "w"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.v_min, self.v_max)
        local_pos = solution.copy()
        return Agent(solution=solution, velocity=velocity, local_solution=local_pos)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        pos_rand = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, pos_rand)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Update weight after each move count  (weight down)
        for idx in range(0, self.pop_size):
            cognitive = self.c1 * self.generator.random(self.problem.n_dims) * (self.pop[idx].local_solution - self.pop[idx].solution)
            social = self.c2 * self.generator.random(self.problem.n_dims) * (self.g_best.solution - self.pop[idx].solution)
            self.pop[idx].velocity = self.w * self.pop[idx].velocity + cognitive + social
            pos_new = self.pop[idx].solution + self.pop[idx].velocity
            pos_new = self.correct_solution(pos_new)
            target = self.get_target(pos_new)
            if self.compare_target(target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=pos_new.copy(), target=target.copy())
            if self.compare_target(target, self.pop[idx].local_target, self.problem.minmax):
                self.pop[idx].update(local_solution=pos_new.copy(), local_target=target.copy())


# aralık
from enumFunctions import Functions
from functions import selectFunction
import numpy as np
from mealpy import FloatVar, PSO
import argparse


parser = argparse.ArgumentParser(description='Process PSO arguments')
parser.add_argument('lb', type=float, help='lb')
parser.add_argument('ub', type=float, help='ub')
parser.add_argument('populationSize', type=float, help='Population Size')
parser.add_argument('numberOfGenerations', type=float, help='Number Of Generations')
parser.add_argument('c1', type=float, help='lb')
parser.add_argument('c2', type=float, help='lb')
parser.add_argument('w', type=float, help='lb')
parser.add_argument('selectedBenchmarkFunction', type=str, help='Selected benchmark function')
parser.add_argument('minmax',type=str,help='wwwwwwww')
parser.add_argument('nVars',type=float,help='wwwwwwww')

# Argümanları ayrıştır
args = parser.parse_args()

# Değişkenlere atama
lb = args.lb
ub = args.ub
populationSize = args.populationSize
numberOfGenerations = args.numberOfGenerations
c1 = args.c1
c2 = args.c2
w = args.w
selectedBenchmarkFunction = args.selectedBenchmarkFunction
minmax = args.minmax
nVars = args.nVars
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
# aralık
def objective_function(solution):
    return np.sum(solution**2)

bounds = FloatVar(lb=(lb,) * int(nVars), ub=(ub,) * int(nVars), name="delta")
problem_dict = {
    "bounds": bounds,
    "minmax": minmax,
    "obj_func": custom_objective_func
}

model = PSO.OriginalPSO(epoch=numberOfGenerations, pop_size=populationSize, c1=c1, c2=c2, w=w)
g_best = model.solve(problem_dict)
