#!/usr/bin/env python
# Created by "Thieu" at 09:33, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseGA(Optimizer):
    """
    The original version of: Genetic Algorithm (GA)

    Links:
        1. https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        2. https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        3. https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.95 +
        + pm (float): [0.01, 0.2], mutation probability, default = 0.025  +
        + selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament" +
        + k_way (float): Optional, set it when use "tournament" selection, default = 0.2 +
        + crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform" +
        + muton_multipoints (bool): Optional, True or False, effect on mutation process, default = True + 
        + mutatioatin (str): Optional, can be ["flip", "swap"] for multipoints and can be ["flip", "swap", "scramble", "inversion"] for one-point

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, GA
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
    >>> model = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    >>>
    >>> model2 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, selection="tournament", k_way=0.4, crossover="multi_points")
    >>>
    >>> model3 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, crossover="one_point", mutation="scramble")
    >>>
    >>> model4 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, crossover="arithmetic", mutation_multipoints=True, mutation="swap")
    >>>
    >>> model5 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, selection="roulette", crossover="multi_points")
    >>>
    >>> model6 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, selection="random", mutation="inversion")
    >>>
    >>> model7 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, crossover="arithmetic", mutation="flip")

    References
    ~~~~~~~~~~
    [1] Whitley, D., 1994. A genetic algorithm tutorial. Statistics and computing, 4(2), pp.65-85.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, pc: float = 0.95, pm: float = 0.025, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            pc: cross-over probability, default = 0.95
            pm: mutation probability, default = 0.025
            selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
            k_way (float): Optional, set it when use "tournament" selection, default = 0.2
            crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
            mutation_multipoints (bool): Optional, True or False, effect on mutation process, default = False
            mutation (str): Optional, can be ["flip", "swap"] for multipoints and can be ["flip", "swap", "scramble", "inversion"] for one-point, default="flip"
        """
        super().__init__(**kwargs)

        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pc = self.validator.check_float("pc", pc, (0, 1.0))
        self.pm = self.validator.check_float("pm", pm, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "pc", "pm"])
        self.sort_flag = False
        self.selection = "tournament"
        self.k_way = 0.2
        self.crossover = "uniform"
        self.mutation = "flip"
        self.mutation_multipoints = True

        if "selection" in kwargs:
            self.selection = self.validator.check_str("selection", kwargs["selection"], ["tournament", "random", "roulette"])
        if "k_way" in kwargs:
            self.k_way = self.validator.check_float("k_way", kwargs["k_way"], (0, 1.0))
        if "crossover" in kwargs:
            self.crossover = self.validator.check_str("crossover", kwargs["crossover"], ["one_point", "multi_points", "uniform", "arithmetic"])
        if "mutation_multipoints" in kwargs:
            self.mutation_multipoints = self.validator.check_bool("mutation_multipoints", kwargs["mutation_multipoints"])
        if self.mutation_multipoints:
            if "mutation" in kwargs:
                self.mutation = self.validator.check_str("mutation", kwargs["mutation"], ["flip", "swap"])
        else:
            if "mutation" in kwargs:
                self.mutation = self.validator.check_str("mutation", kwargs["mutation"], ["flip", "swap", "scramble", "inversion"])

    def selection_process__(self, list_fitness):
        """
        Notes
        ~~~~~
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
        + Default selection strategy is Tournament with k% = 0.2.
        + Other strategy like "roulette" and "random" can be selected via Optional parameter "selection"

        Args:
            list_fitness (np.array): list of fitness values.

        Returns:
            list: The position of dad and mom
        """
        if self.selection == "roulette":
            id_c1 = self.get_index_roulette_wheel_selection(list_fitness)
            id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
            while id_c2 == id_c1:
                id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
        elif self.selection == "random":
            id_c1, id_c2 = self.generator.choice(range(self.pop_size), 2, replace=False)
        else:   ## tournament
            id_c1, id_c2 = self.get_index_kway_tournament_selection(self.pop, k_way=self.k_way, output=2)
        return self.pop[id_c1].solution, self.pop[id_c2].solution

    def selection_process_00__(self, pop_selected):
        """
        Notes
        ~~~~~
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
        + Default selection strategy is Tournament with k% = 0.2.
        + Other strategy like "roulette" and "random" can be selected via Optional parameter "selection"

        Args:
            pop_selected (np.array): a population that will be selected

        Returns:
            list: The position of dad and mom
        """
        if self.selection == "roulette":
            list_fitness = np.array([agent.target.fitness for agent in pop_selected])
            id_c1 = self.get_index_roulette_wheel_selection(list_fitness)
            id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
            while id_c2 == id_c1:
                id_c2 = self.get_index_roulette_wheel_selection(list_fitness)
        elif self.selection == "random":
            id_c1, id_c2 = self.generator.choice(range(len(pop_selected)), 2, replace=False)
        else:   ## tournament
            id_c1, id_c2 = self.get_index_kway_tournament_selection(pop_selected, k_way=self.k_way, output=2)
        return pop_selected[id_c1].solution, pop_selected[id_c2].solution

    def selection_process_01__(self, pop_dad, pop_mom):
        """
        Notes
        ~~~~~
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
        + Default selection strategy is Tournament with k% = 0.2.
        + Other strategy like "roulette" and "random" can be selected via Optional parameter "selection"

        Returns:
            list: The position of dad and mom
        """
        if self.selection == "roulette":
            list_fit_dad = np.array([agent.target.fitness for agent in pop_dad])
            list_fit_mom = np.array([agent.target.fitness for agent in pop_mom])
            id_c1 = self.get_index_roulette_wheel_selection(list_fit_dad)
            id_c2 = self.get_index_roulette_wheel_selection(list_fit_mom)
        elif self.selection == "random":
            id_c1 = self.generator.choice(range(len(pop_dad)))
            id_c2 = self.generator.choice(range(len(pop_mom)))
        else:   ## tournament
            id_c1 = self.get_index_kway_tournament_selection(pop_dad, k_way=self.k_way, output=1)[0]
            id_c2 = self.get_index_kway_tournament_selection(pop_mom, k_way=self.k_way, output=1)[0]
        return pop_dad[id_c1].solution, pop_mom[id_c2].solution

    def crossover_process__(self, dad, mom):
        """
        Notes
        ~~~~~
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm
        + Default crossover strategy is "uniform"
        + Other strategy like "arithmetic", "one_point", "multi_points" can be selected via parameter: crossover

        Args:
            dad (np.array): The position of dad
            mom (np.array): The position of mom

        Returns:
            list: The position of child 1 and child 2
        """
        if self.crossover == "arithmetic":
            w1, w2 = self.crossover_arithmetic(dad, mom)
        elif self.crossover == "one_point":
            cut = self.generator.integers(1, self.problem.n_dims-1)
            w1 = np.concatenate([dad[:cut], mom[cut:]])
            w2 = np.concatenate([mom[:cut], dad[cut:]])
        elif self.crossover == "multi_points":
            idxs = self.generator.choice(range(1, self.problem.n_dims-1), 2, replace=False)
            cut1, cut2 = np.min(idxs), np.max(idxs)
            w1 = np.concatenate([dad[:cut1], mom[cut1:cut2], dad[cut2:]])
            w2 = np.concatenate([mom[:cut1], dad[cut1:cut2], mom[cut2:]])
        else:           # uniform
            flip = self.generator.integers(0, 2, self.problem.n_dims)
            w1 = dad * flip + mom * (1 - flip)
            w2 = mom * flip + dad * (1 - flip)
        return w1, w2

    def mutation_process__(self, child):
        """
        Notes
        ~~~~~
        + https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_mutation.htm
        + There are 2 strategies that effects by the mutation probability: Mutated on single point or the whole vector.
            + Multiple points (whole vector) has 2 strategies selected via parameter: mutation
                + flip --> (default in this case) should set the pm small such as: [0.01 -> 0.2]
                + swap --> should set the pm small such as: [0.01 -> 0.2]
            + Single point has 4 strategies:
                + flip --> should set the pm large such as: [0.5 -> 0.9]
                + swap --> same as flip: pm in range [0.5 -> 0.9]
                + scramble --> should set the pm small enough such as: [0.4 -> 0.6]
                + inversion --> like scramble [0.4 -> 0.6]

        Args:
            child (np.array): The position of the child

        Returns:
            np.array: The mutated vector of the child
        """

        if self.mutation_multipoints:
            if self.mutation == "swap":
                for idx in range(self.problem.n_dims):
                    idx_swap = self.generator.choice(list(set(range(0, self.problem.n_dims)) - {idx}))
                    child[idx], child[idx_swap] = child[idx_swap], child[idx]
                    return child
            else:       # "flip"
                mutation_child = self.problem.generate_solution()
                flag_child = self.generator.uniform(0, 1, self.problem.n_dims) < self.pm
                return np.where(flag_child, mutation_child, child)
        else:
            if self.mutation == "swap":
                idx1, idx2 = self.generator.choice(range(0, self.problem.n_dims), 2, replace=False)
                child[idx1], child[idx2] = child[idx2], child[idx1]
                return child
            elif self.mutation == "inversion":
                cut1, cut2 = self.generator.choice(range(0, self.problem.n_dims), 2, replace=False)
                temp = child[cut1:cut2]
                temp = temp[::-1]
                child[cut1:cut2] = temp
                return child
            elif self.mutation == "scramble":
                cut1, cut2 = self.generator.choice(range(0, self.problem.n_dims), 2, replace=False)
                temp = child[cut1:cut2]
                self.generator.shuffle(temp)
                child[cut1:cut2] = temp
                return child
            else:   # "flip"
                idx = self.generator.integers(0, self.problem.n_dims)
                child[idx] = self.generator.uniform(self.problem.lb[idx], self.problem.ub[idx])
                return child

    def survivor_process__(self, pop, pop_child):
        """
        The current survivor process is select the worst solution out of k-way solutions (tournament selection) and
        compare with child solutions. The better solution will be kept for the next generation.

        Args:
            pop: The old population
            pop_child: The new population

        Returns:
            The new population
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            id_child = self.get_index_kway_tournament_selection(pop, k_way=0.1, output=1, reverse=True)[0]
            pop_new.append(self.get_better_agent(pop_child[idx], pop[id_child], self.problem.minmax))
        return pop_new

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        list_fitness = np.array([agent.target.fitness for agent in self.pop])
        pop_new = []
        for i in range(0, int(self.pop_size/2)):
            ### Selection
            child1, child2 = self.selection_process__(list_fitness)

            ### Crossover
            if self.generator.random() < self.pc:
                child1, child2 = self.crossover_process__(child1, child2)

            ### Mutation
            child1 = self.mutation_process__(child1)
            child2 = self.mutation_process__(child2)

            child1 = self.correct_solution(child1)
            child2 = self.correct_solution(child2)

            agent1 = self.generate_empty_agent(child1)
            agent2 = self.generate_empty_agent(child2)

            pop_new.append(agent1)
            pop_new.append(agent2)

            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-2].target = self.get_target(child1)
                pop_new[-1].target = self.get_target(child2)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
        ### Survivor Selection
        self.pop = self.survivor_process__(self.pop, pop_new)




# aralık
from enumFunctions import Functions
from functions import selectFunction
import numpy as np
from mealpy import FloatVar, GA
import argparse


parser = argparse.ArgumentParser(description='Process CRO arguments')
parser.add_argument('nVars', type=float, help='N Vars')
parser.add_argument('populationSize', type=float, help='Population Size')
parser.add_argument('lb', type=float, help='Lower Bound')
parser.add_argument('ub', type=float, help='Upper Bound')
parser.add_argument('numberOfGenerations', type=float, help='Epoch')
parser.add_argument('mutationProbability', type=float, help='aaaaaaaaaaaaaaaa')
parser.add_argument('crossoverProbability', type=float, help='aaAaAAaaaaaAaaaaaa')
parser.add_argument('crossoverType', type=str, help='AAaAaAAAAAaaaAAa')
parser.add_argument('selectionType', type=str, help='AAaAaAAAAAaaaAAa')
parser.add_argument('mutationMultiPoints', type=bool, help='AAaAaAAAAAaaaAAa')
parser.add_argument('mutationType', type=str, help='AAaAaAAAAAaaaAAa')
parser.add_argument('kWay', type=float, help='AAaAaAAAAAaaaAAa')
parser.add_argument('selectedBenchmarkFunction', type=str, help='AAaAaAAAAAaaaAAa')




# Argümanları ayrıştır
args = parser.parse_args()

# Değişkenlere atama
nVars = args.nVars
populationSize = args.populationSize
lb = args.lb
ub = args.ub
numberOfGenerations = args.numberOfGenerations
mutationProbability = args.mutationProbability
crossoverProbability = args.crossoverProbability
crossoverType = args.crossoverType
mutationMultiPoint = args.mutationMultiPoints
mutationType = args.mutationType
selectionType = args.selectionType
kWay = args.kWay
selectedBenchmarkFunction = args.selectedBenchmarkFunction

if crossoverType == 'multi':
    crossoverType = 'multi_points'
if crossoverType == 'one-point':
    crossoverType = 'one_point'


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
from mealpy import FloatVar, GA

bounds = FloatVar(lb=(lb,) * int(nVars), ub=(ub,) * int(nVars), name="delta")
problem_dict = {
    "bounds": bounds,
    "minmax": 'min',
    "obj_func": custom_objective_func
}


model = GA.BaseGA(epoch=numberOfGenerations, pop_size=populationSize, pc=crossoverProbability, pm=mutationProbability, crossover=crossoverType, selection = selectionType, mutation_multipoints=mutationMultiPoint, mutation=mutationType, k_way=kWay)
g_best = model.solve(problem_dict)