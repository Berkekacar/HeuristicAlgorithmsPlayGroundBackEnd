#!/usr/bin/env python
# Created by "Thieu" at 14:22, 11/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent


class OriginalMA(Optimizer):
    """
    The original version of: Memetic Algorithm (MA)

    Links:
        1. https://www.cleveralgorithms.com/nature-inspired/physical/memetic_algorithm.html
        2. https://github.com/clever-algorithms/CleverAlgorithms

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.85
        + pm (float): [0.05, 0.3], mutation probability, default = 0.15
        + p_local (float): [0.3, 0.7], Probability of local search for each agent, default=0.5
        + max_local_gens (int): [5, 25], number of local search agent will be created during local search mechanism, default=10
        + bits_per_param (int): [2, 4, 8, 16], number of bits to decode a real number to 0-1 bitstring, default=4

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, MA
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
    >>> model = MA.OriginalMA(epoch=1000, pop_size=50, pc = 0.85, pm = 0.15, p_local = 0.5, max_local_gens = 10, bits_per_param = 4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Moscato, P., 1989. On evolution, search, optimization, genetic algorithms and martial arts:
    Towards memetic algorithms. Caltech concurrent computation program, C3P Report, 826, p.1989.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, pc: float = 0.85, pm: float = 0.15,
                 p_local: float = 0.5, max_local_gens: int = 10, bits_per_param: int = 4, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pc (float): cross-over probability, default = 0.85
            pm (float): mutation probability, default = 0.15
            p_local (float): Probability of local search for each agent, default=0.5
            max_local_gens (int): Number of local search agent will be created during local search mechanism, default=10
            bits_per_param (int): Number of bits to decode a real number to 0-1 bitstring, default=4
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pc = self.validator.check_float("pc", pc, (0, 1.0))
        self.pm = self.validator.check_float("pm", pm, (0, 1.0))
        self.p_local = self.validator.check_float("p_local", p_local, (0, 1.0))
        self.max_local_gens = self.validator.check_int("max_local_gens", max_local_gens, [2, int(pop_size/2)])
        self.bits_per_param = self.validator.check_int("bits_per_param", bits_per_param, [2, 32])
        self.set_parameters(["epoch", "pop_size", "pc", "pm", "p_local", "max_local_gens", "bits_per_param"])
        self.sort_flag = True

    def initialize_variables(self):
        self.bits_total = self.problem.n_dims * self.bits_per_param

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        bitstring = ''.join(["1" if self.generator.uniform() < 0.5 else "0" for _ in range(0, self.bits_total)])
        return Agent(solution=solution, bitstring=bitstring)

    def decode__(self, bitstring: str = None) -> np.ndarray:
        """
        Decode the random bitstring into real number

        Args:
            bitstring (str): "11000000100101000101010" - bits_per_param = 16, 32 bit for 2 variable. eg. x1 and x2

        Returns:
            list: list of real number (vector)
        """
        vector = np.ones(self.problem.n_dims)
        for idx in range(0, self.problem.n_dims):
            param = bitstring[idx * self.bits_per_param: (idx + 1) * self.bits_per_param]  # Select 16 bit every time
            vector[idx] = self.problem.lb[idx] + ((self.problem.ub[idx] - self.problem.lb[idx]) / ((2.0 ** self.bits_per_param) - 1)) * int(param, 2)
        return vector

    def crossover__(self, dad=None, mom=None):
        if self.generator.uniform() >= self.pc:
            return dad
        else:
            child = ""
            for idx in range(0, self.bits_total):
                if self.generator.uniform() < 0.5:
                    child += dad[idx]
                else:
                    child += mom[idx]
            return child

    def point_mutation__(self, bitstring=None):
        child = ""
        for bit in bitstring:
            if self.generator.uniform() < self.pc:
                child += "0" if bit == "1" else "1"
            else:
                child += bit
        return child

    def bits_climber__(self, child=None):
        current = child
        list_local = []
        for idx in range(0, self.max_local_gens):
            child = current
            bitstring_new = self.point_mutation__(child.bitstring)
            pos_new = self.decode__(bitstring_new)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            agent.update(solution=pos_new, bitstring=bitstring_new)
            list_local.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                list_local[-1].target = self.get_target(pos_new)
        list_local = self.update_target_for_population(list_local)
        list_local.append(child)
        best = self.get_best_agent(list_local, self.problem.minmax)
        return best

    def create_child__(self, idx, pop_copy):
        ancient = pop_copy[idx + 1] if idx % 2 == 0 else pop_copy[idx - 1]
        if idx == self.pop_size - 1:
            ancient = pop_copy[0]
        bitstring_new = self.crossover__(pop_copy[idx].bitstring, ancient.bitstring)
        bitstring_new = self.point_mutation__(bitstring_new)
        pos_new = self.decode__(bitstring_new)
        pos_new = self.correct_solution(pos_new)
        agent = self.generate_agent(pos_new)
        agent.bitstring = bitstring_new
        return agent

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Binary tournament
        children = []
        for idx in range(0, self.pop_size):
            idx_offspring = self.get_index_kway_tournament_selection(self.pop, k_way=2, output=1)[0]
            children.append(self.pop[idx_offspring].copy())
        pop = []
        for idx in range(0, self.pop_size):
            ancient = children[idx + 1] if idx % 2 == 0 else children[idx - 1]
            if idx == self.pop_size - 1:
                ancient = children[0]
            bitstring_new = self.crossover__(children[idx].bitstring, ancient.bitstring)
            bitstring_new = self.point_mutation__(bitstring_new)
            pos_new = self.decode__(bitstring_new)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            agent.update(bitstring=bitstring_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop[-1].target = self.get_target(pos_new)
        self.pop = self.update_target_for_population(pop)
        # Searching in local
        for idx in range(0, self.pop_size):
            if self.generator.random() < self.p_local:
                self.pop[idx] = self.bits_climber__(pop[idx])



from enumFunctions import Functions
from functions import selectFunction
import numpy as np
from mealpy import FloatVar, SMA
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
parser.add_argument('pc', type=float, help=" pc")
parser.add_argument('pm', type=float, help="pm")
parser.add_argument('pLocal', type=float, help="p local")
parser.add_argument('maxLocalGenes', type=int, help=" maxLocalGenes")
parser.add_argument('bitsPerParam', type=int, help=" bitsPerParam")

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
pc=args.pc
pm = args.pm
pLocal = args.pLocal
maxLocalGenes = args.maxLocalGenes
bitsPerParam= args.bitsPerParam

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
from mealpy import FloatVar, MA

def objective_function(solution):
    return np.sum(solution**2)
bounds = FloatVar(lb=(lb,) * int(nVars), ub=(ub,) * int(nVars), name="delta")
problem_dict = {
    "bounds": bounds,
    "obj_func": custom_objective_func,
    "minmax": minmax,
}

model = MA.OriginalMA(epoch=epoch, pop_size=popSize, pc = pc, pm = pm, p_local = pLocal, max_local_gens = maxLocalGenes, bits_per_param = bitsPerParam)
g_best = model.solve(problem_dict)

