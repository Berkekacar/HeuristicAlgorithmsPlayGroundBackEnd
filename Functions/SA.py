import numpy as np
from mealpy.optimizer import Optimizer
import argparse


class OriginalSA(Optimizer):
    """
    The original version of: Simulated Annealing (OriginalSA)

    Notes:
        + SA is single-based solution, so the pop_size parameter is not matter in this algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + temp_init (float): [1, 10000], initial temperature, default=100
        + step_size (float): the step size of random movement, default=0.1

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy import FloatVar, SA
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
    >>> model = SA.OriginalSA(epoch=1000, pop_size=50, temp_init = 100, step_size = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kirkpatrick, S., Gelatt Jr, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. science, 220(4598), 671-680.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 2, temp_init: float = 100, step_size: float = 0.1, **kwargs: object) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            temp_init (float): initial temperature, default=100
            step_size (float): the step size of random movement, default=0.1
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [2, 10000])
        self.temp_init = self.validator.check_float("temp_init", temp_init, [1, 10000])
        self.step_size = self.validator.check_float("step_size", step_size, (-100., 100.))
        self.set_parameters(["epoch", "temp_init", "step_size"])

    def before_main_loop(self):
        self.agent_current = self.g_best.copy()

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        # Perturb the current solution

        pos_new = self.agent_current.solution + self.generator.standard_normal(self.problem.n_dims) * self.step_size
        agent = self.generate_agent(pos_new)
        # Accept or reject the new solution
        if self.compare_target(agent.target, self.agent_current.target, self.problem.minmax):
            self.agent_current = agent
        else:
            # Calculate the energy difference
            delta_energy = np.abs(self.agent_current.target.fitness - agent.target.fitness)
            # calculate probability acceptance criterion
            p_accept = np.exp(-delta_energy/ (self.temp_init / float(epoch + 1)))
            if self.generator.random() < p_accept:
                self.agent_current = agent
        self.pop = [self.g_best.copy(), self.agent_current.copy()]

from mealpy import FloatVar, SA
import numpy as np
from enumFunctions import Functions
from functions import selectFunction

parser = argparse.ArgumentParser(description='Process SA arguments')
parser.add_argument('nVars', type=float, help='Lower Bound')
parser.add_argument('lb', type=float, help='Lower Bound')
parser.add_argument('ub', type=float, help='Upper Bound')
parser.add_argument('initialTemperature', type=float, help='Initial temperature')
parser.add_argument('coolDownFactor', type=float, help='Cool down factor')
parser.add_argument('temperatureDecreaseType', type=str, help='Temperature decrease type')
parser.add_argument('populationSize', type=float, help='Lower Bound')
parser.add_argument('epoch', type=float, help='Lower Bound')
parser.add_argument('selectedBenchmarkFunction', type=str, help='Selected benchmark function')
parser.add_argument('minmax', type=str, help='aaaaaa')

# Argümanları ayrıştır
args = parser.parse_args()

# Değişkenlere atama
nVars = args.nVars
lb = args.lb
ub = args.ub
initial_temperature = args.initialTemperature
cool_down_factor = args.coolDownFactor
temperature_decrease_type = args.temperatureDecreaseType
populationSize = args.populationSize
epoch = args.epoch
selected_benchmark_function = args.selectedBenchmarkFunction
minmax = args.minmax
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


selected_function_name = selected_benchmark_function
custom_objective_func = select_function_by_name(selected_function_name)

# def objective_function(solution):
#     return np.sum(solution**2)

bounds = FloatVar(lb=(lb,) * int(nVars), ub=(ub,) * int(nVars), name="delta")
problem_dict = {
    "bounds": bounds,
    "minmax": minmax,
    "obj_func": custom_objective_func
}

model = SA.OriginalSA(epoch=epoch, pop_size=populationSize, temp_init=initial_temperature, step_size=cool_down_factor)
g_best = model.solve(problem_dict)







