#!/usr/bin/env python
# Created by "Thieu" at 17:21, 21/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class DevSOA(Optimizer):
    """
    The developed version: Seagull Optimization Algorithm (SOA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0950705118305768

    Notes:
        1. The original one will not work because their operators always make the solution out of bound.
        2. I added the normal random number in Eq. 14 to make its work
        3. Besides, I will check keep the better one and remove the worst

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + fc (float): [1.0, 10.0] -> better [1, 5], freequency of employing variable A (A linear decreased from fc to 0), default = 2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SOA
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
    >>> model = SOA.DevSOA(epoch=1000, pop_size=50, fc = 2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(self, epoch=10000, pop_size=100, fc=2, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.fc = self.validator.check_float("fc", fc, [1.0, 10.])
        self.set_parameters(["epoch", "pop_size", "fc"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        A = self.fc - epoch*self.fc / self.epoch    # Eq. 6
        uu = vv = 1
        pop_new = []
        for idx in range(0, self.pop_size):
            B = 2 * A**2 * self.generator.random()                          # Eq. 8
            M = B * (self.g_best.solution - self.pop[idx].solution)         # Eq. 7
            C = A * self.pop[idx].solution                                  # Eq. 5
            D = np.abs(C + M)                                               # Eq. 9
            k = self.generator.uniform(0, 2*np.pi)
            r = uu * np.exp(k*vv)
            xx = r * np.cos(k)
            yy = r * np.sin(k)
            zz = r * k
            pos_new = xx * yy * zz * D + self.generator.normal(0, 1) * self.g_best.solution                 # Eq. 14
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(agent, self.pop[idx], self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(self.pop, pop_new, self.problem.minmax)


class OriginalSOA(Optimizer):
    """
    The original version: Seagull Optimization Algorithm (SOA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0950705118305768

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + fc (float): [1.0, 10.0] -> better [1, 5], freequency of employing variable A (A linear decreased from fc to 0), default = 2

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SOA
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
    >>> model = SOA.OriginalSOA(epoch=1000, pop_size=50, fc = 2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Dhiman, G., & Kumar, V. (2019). Seagull optimization algorithm: Theory and its applications
    for large-scale industrial engineering problems. Knowledge-based systems, 165, 169-196.
    """

    def __init__(self, epoch=10000, pop_size=100, fc=2, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.fc = self.validator.check_float("fc", fc, [1.0, 10.])
        self.set_parameters(["epoch", "pop_size", "fc"])
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        A = self.fc - epoch*self.fc / self.epoch    # Eq. 6
        uu = vv = 1
        pop_new = []
        for idx in range(0, self.pop_size):
            B = 2 * A**2 * self.generator.random()                      # Eq. 8
            M = B * (self.g_best.solution - self.pop[idx].solution)     # Eq. 7
            C = A * self.pop[idx].solution                              # Eq. 5
            D = np.abs(C + M)                                           # Eq. 9
            k = self.generator.uniform(0, 2*np.pi)
            r = uu * np.exp(k*vv)
            xx = r * np.cos(k)
            yy = r * np.sin(k)
            zz = r * k
            pos_new = xx * yy * zz * D + self.g_best.solution           # Eq. 14
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        self.pop = self.update_target_for_population(pop_new)


from enumFunctions import Functions
from functions import selectFunction
from mealpy import FloatVar, SOA
import numpy as np
from mealpy import FloatVar
import argparse


parser = argparse.ArgumentParser(description='Process PSO arguments')
parser.add_argument('nVars', type=int, help="n vars")
parser.add_argument('lb', type=float, help="lower bound")
parser.add_argument('ub', type=float, help="upper bound")
parser.add_argument('name', type=str, help=" name")
parser.add_argument('minmax', type=str, help=" minmax")
parser.add_argument('selectedBenchmarkFunction', type=str, help='Selected benchmark function')
parser.add_argument('epoch', type=int, help="epoch")
parser.add_argument('popSize', type=int, help="pop size")
parser.add_argument('fc', type=float, help="fc bound")




# Argümanları ayrıştır
args = parser.parse_args()

# Değişkenlere atama
nVars=args.nVars
lb = args.lb
ub = args.ub
name = args.name
minmax = args.minmax
selectedBenchmarkFunction = args.selectedBenchmarkFunction
epoch= args.epoch
popSize=args.popSize
fc = args.fc

if minmax == "Min":
    minmax = "min"
else: 
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
from mealpy import FloatVar, SOA
bounds = FloatVar(lb=(lb,) * int(nVars), ub=(ub,) * int(nVars), name=name)
problem_dict = {
    "bounds": bounds,
    "minmax": minmax,
    "obj_func": custom_objective_func
}
model = SOA.OriginalSOA(epoch=epoch, pop_size=popSize, fc = fc)
g_best = model.solve(problem_dict)
