#!/usr/bin/env python
# Created by "Thieu" at 08:37, 17/06/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from scipy.stats import cauchy


class OriginalSHADE(Optimizer):
    """
    The original version of: Success-History Adaptation Differential Evolution (OriginalSHADE)

    Links:
        1. https://doi.org/10.1109/CEC.2013.6557555

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + miu_f (float): [0.4, 0.6], initial weighting factor, default = 0.5
        + miu_cr (float): [0.4, 0.6], initial cross-over probability, default = 0.5

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SHADE
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
    >>> model = SHADE.OriginalSHADE(epoch=1000, pop_size=50, miu_f = 0.5, miu_cr = 0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Tanabe, R. and Fukunaga, A., 2013, June. Success-history based parameter adaptation for
    differential evolution. In 2013 IEEE congress on evolutionary computation (pp. 71-78). IEEE.
    """

    def __init__(self, epoch: int = 750, pop_size: int = 100, miu_f: float = 0.5, miu_cr: float = 0.5, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_f (float): initial weighting factor, default = 0.5
            miu_cr (float): initial cross-over probability, default = 0.5
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        # the initial f, location is changed then that f is good
        self.miu_f = self.validator.check_float("miu_f", miu_f, (0, 1.0))
        # the initial cr,
        self.miu_cr = self.validator.check_float("miu_cr", miu_cr, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "miu_f", "miu_cr"])
        self.sort_flag = False

    def initialize_variables(self):
        self.dyn_miu_f = self.miu_f * np.ones(self.pop_size)  # list the initial f,
        self.dyn_miu_cr = self.miu_cr * np.ones(self.pop_size)  # list the initial cr,
        self.dyn_pop_archive = list()
        self.k_counter = 0

    ### Survivor Selection
    def weighted_lehmer_mean__(self, list_objects, list_weights):
        up = list_weights * list_objects ** 2
        down = list_weights * list_objects
        return np.sum(up) / np.sum(down)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        list_f = list()
        list_cr = list()
        list_f_index = list()
        list_cr_index = list()
        list_f_new = np.ones(self.pop_size)
        list_cr_new = np.ones(self.pop_size)
        pop_old = [agent.copy() for agent in self.pop]
        pop_sorted = self.get_sorted_population(self.pop, self.problem.minmax)
        pop = []
        for idx in range(0, self.pop_size):
            ## Calculate adaptive parameter cr and f
            idx_rand = self.generator.integers(0, self.pop_size)
            cr = self.generator.normal(self.dyn_miu_cr[idx_rand], 0.1)
            cr = np.clip(cr, 0, 1)
            while True:
                f = cauchy.rvs(self.dyn_miu_f[idx_rand], 0.1)
                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break
            list_cr_new[idx] = cr
            list_f_new[idx] = f
            p = self.generator.uniform(2 / self.pop_size, 0.2)
            top = int(self.pop_size * p)
            x_best = pop_sorted[self.generator.integers(0, top)]
            x_r1 = self.pop[self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))]
            new_pop = self.pop + self.dyn_pop_archive
            while True:
                x_r2 = new_pop[self.generator.integers(0, len(new_pop))]
                if np.any(x_r2.solution - x_r1.solution) and np.any(x_r2.solution - self.pop[idx].solution):
                    break
            x_new = self.pop[idx].solution + f * (x_best.solution - self.pop[idx].solution) + f * (x_r1.solution - x_r2.solution)
            condition = self.generator.random(self.problem.n_dims) < cr
            pos_new = np.where(condition, x_new, self.pop[idx].solution)
            j_rand = self.generator.integers(0, self.problem.n_dims)
            pos_new[j_rand] = x_new[j_rand]
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop[-1].target = self.get_target(pos_new)
        pop = self.update_target_for_population(pop)
        for idx in range(0, self.pop_size):
            if self.compare_target(pop[idx].target, self.pop[idx].target, self.problem.minmax):
                list_cr.append(list_cr_new[idx])
                list_f.append(list_f_new[idx])
                list_f_index.append(idx)
                list_cr_index.append(idx)
                self.pop[idx] = pop[idx].copy()
                self.dyn_pop_archive.append(pop[idx].copy())
        # Randomly remove solution
        temp = len(self.dyn_pop_archive) - self.pop_size
        if temp > 0:
            idx_list = self.generator.choice(range(0, len(self.dyn_pop_archive)), temp, replace=False)
            archive_pop_new = []
            for idx, agent in enumerate(self.dyn_pop_archive):
                if idx not in idx_list:
                    archive_pop_new.append(agent.copy())
            self.dyn_pop_archive = archive_pop_new

        # Update miu_cr and miu_f
        if len(list_f) != 0 and len(list_cr) != 0:
            # Eq.13, 14, 10
            list_fit_old = np.ones(len(list_cr_index))
            list_fit_new = np.ones(len(list_cr_index))
            idx_increase = 0
            for idx in range(0, self.pop_size):
                if idx in list_cr_index:
                    list_fit_old[idx_increase] = pop_old[idx].target.fitness
                    list_fit_new[idx_increase] = self.pop[idx].target.fitness
                    idx_increase += 1
            temp = np.sum(np.abs(list_fit_new - list_fit_old))
            if temp == 0:
                list_weights = 1.0 / len(list_fit_new) * np.ones(len(list_fit_new))
            else:
                list_weights = np.abs(list_fit_new - list_fit_old) / temp
            self.dyn_miu_cr[self.k_counter] = np.sum(list_weights * np.array(list_cr))
            self.dyn_miu_f[self.k_counter] = self.weighted_lehmer_mean__(np.array(list_f), list_weights)
            self.k_counter += 1
            if self.k_counter >= self.pop_size:
                self.k_counter = 0


from enumFunctions import Functions
from functions import selectFunction
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
parser.add_argument('miuF', type=float, help="miuF bound")
parser.add_argument('miuCR', type=float, help="miuCR bound")



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
miuF = args.miuF
miuCR = args.miuCR



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
from mealpy import FloatVar, SHADE

def objective_function(solution):
    return np.sum(solution**2)
bounds = FloatVar(lb=(lb,) * int(nVars), ub=(ub,) * int(nVars), name="delta")
problem_dict = {
    "bounds": bounds,
    "obj_func": custom_objective_func,
    "minmax": minmax,
}

model = SHADE.OriginalSHADE(epoch=epoch, pop_size=popSize, miu_f = miuF, miu_cr = miuCR)
g_best = model.solve(problem_dict)