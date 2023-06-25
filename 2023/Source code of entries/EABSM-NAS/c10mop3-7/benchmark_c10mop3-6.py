import os
import sys
import random
import datetime
import numpy as np
os.chdir('/home/wcx/gitProject/evoxbench/evoxbench_submit/c10mop3-7')
sys.path.insert(0, '/home/wcx/gitProject/evoxbench/evoxbench_submit') # for evoxbench
sys.path.insert(0, '/home/wcx/gitProject/evoxbench/evoxbench_submit/pymoo060') # for pymoo0.6.0


from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair

from evoxbench.test_suites import c10mop
from evoxbench.benchmarks.nb101 import NASBench101Graph

_DEBUG = False  # run in debug mode

class C10MOPProblem(Problem):
    def __init__(self,
                 benchmark,
                 **kwargs):
        super().__init__(n_var=benchmark.search_space.n_var, n_obj=benchmark.evaluator.n_objs,
                         n_constr=0, xl=benchmark.search_space.lb, xu=benchmark.search_space.ub,
                         type_var=np.int64, **kwargs)
        self.benchmark = benchmark
        self.X_history = {}

    def _evaluate(self, x, out, *args, **kwargs):

        F = self.benchmark.evaluate(x, true_eval=True)

        out["F"] = F
    
    def _add_history(self, X):
        self.X_history.update({','.join(str(x) for x in p): 1 for p in X})
        
    def _get_valid_x(self, X):
        return range(len(X))
    
    def _get_fine_x(self, X):
        return range(len(X))
        
def get_genetic_operator(crx_prob=1.0,  # crossover probability
                         crx_eta=30.0,  # SBX crossover eta
                         mut_prob=0.9,  # mutation probability
                         mut_eta=20.0,  # polynomial mutation hyperparameter eta
                         ):
    sampling = IntegerRandomSampling()
    crossover = SBX(prob=crx_prob, eta=crx_eta, repair=RoundingRepair(), vtype=int)
    mutation = PM(prob=mut_prob, eta=mut_eta, repair=RoundingRepair(), vtype=int)
    return sampling, crossover, mutation


def get_benchmark_settings(n_obj):
    n_gen = 100

    if n_obj == 2:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=99)
    elif n_obj == 3:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=13)
    elif n_obj == 4:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=7)
    elif n_obj == 5:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=5)
    elif n_obj == 6:
        ref_dirs = get_reference_directions(
            "multi-layer",
            get_reference_directions("das-dennis", n_obj, n_partitions=4, scaling=1.0),
            get_reference_directions("das-dennis", n_obj, n_partitions=1, scaling=0.5))
    elif n_obj == 8:
        ref_dirs = get_reference_directions(
            "multi-layer",
            get_reference_directions("das-dennis", n_obj, n_partitions=3, scaling=1.0),
            get_reference_directions("das-dennis", n_obj, n_partitions=2, scaling=0.5))
    else:
        raise NotImplementedError

    pop_size = ref_dirs.shape[0]

    return pop_size, n_gen, ref_dirs


def nsga2(pop_size,
          crx_prob=1.0,  # crossover probability
          crx_eta=30.0,  # SBX crossover eta
          mut_prob=0.9,  # mutation probability
          mut_eta=20.0,  # polynomial mutation hyperparameter eta
          ):

    sampling, crossover, mutation = get_genetic_operator(crx_prob, crx_eta, mut_prob, mut_eta)

    return NSGA2(pop_size=pop_size, sampling=sampling, crossover=crossover,
        mutation=mutation, eliminate_duplicates=True)


def moead(ref_dirs,
          crx_prob=1.0,  # crossover probability
          crx_eta=20.0,  # SBX crossover eta
          mut_prob=0.9,  # mutation probability
          mut_eta=20.0,  # polynomial mutation hyperparameter eta
          neighborhood_size=20,  # neighborhood size
          prob_neighbor_mating=0.9,  # neighborhood selection probability
          ):

    sampling, crossover, mutation = get_genetic_operator(crx_prob, crx_eta, mut_prob, mut_eta)
    return MOEAD(ref_dirs=ref_dirs, n_neighbors=neighborhood_size, prob_neighbor_mating=prob_neighbor_mating,
        sampling=sampling, crossover=crossover, mutation=mutation)

def nsga3(pop_size,
          ref_dirs,
          crx_prob=1.0,  # crossover probability
          crx_eta=30.0,  # SBX crossover eta
          mut_prob=0.9,  # mutation probability
          mut_eta=20.0,  # polynomial mutation hyperparameter eta
          ):

    sampling, crossover, mutation = get_genetic_operator(crx_prob, crx_eta, mut_prob, mut_eta)

    return NSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=sampling, crossover=crossover,
        mutation=mutation, eliminate_duplicates=True)
    
if __name__ == '__main__':
    import json
    import argparse

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    parser = argparse.ArgumentParser(description='Benchmark C10/MOP')
    parser.add_argument('--moea', type=str, default='nsga2', help='which MOEA to run')
    parser.add_argument('--runs', type=int, default=31, help='number of runs to repeat')
    args = parser.parse_args()
    
    for pid in range(3, 6):
        experiment_stats = []
        print('start to run problem {}'.format(pid), ' time = ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') )
        for r in range(1, args.runs+1):
            run_stats = {'run': r}

            benchmark = c10mop(pid)
            problem = C10MOPProblem(benchmark)

            pop_size, n_gen, ref_dirs = get_benchmark_settings(problem.n_obj)
            print(pop_size)

            if args.moea == 'nsga2':
                algorithm = nsga2(pop_size)
            elif args.moea == 'moead':
                algorithm = moead(ref_dirs)
            elif args.moea == 'nsga3':
                algorithm = nsga3(pop_size, ref_dirs)
            else:
                raise NotImplementedError

            # train_x and train_y
            train_x = []
            train_y = []

            hv_list = []
            remains = 10000
            
            hashs = []
            fitness = []
            pops = []

            searched_arch = set()
            algorithm.setup(problem=problem, seed=r, verbose=True)
            for i in range(1, n_gen+1):
                pop = algorithm.ask()
                
                if pop is None:
                    res = algorithm.result()
                    break
                
                pop_unique = set([','.join(str(x) for x in p) for p in pop.get('X')])
                searched_arch.update(pop_unique)
                print('searched arch: ', len(searched_arch) )
                
                # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
                algorithm.evaluator.eval(problem, pop)
                remains -= len(pop)

                f = pop.get('F')
                fitness.extend(f)
                per = benchmark.calc_perf_indicator_fitness(np.array(fitness), 'hv')

                # returned the evaluated individuals which have been evaluated or even modified
                algorithm.tell(infills=pop)
                pop_unique = set([','.join(str(x) for x in p) for p in algorithm.pop.get('X')])
                hv_pop = benchmark.calc_perf_indicator(algorithm.pop.get("X"), 'hv')
                hv_list.append(hv_pop)
                print('hv: ', hv_pop)
                
                F = algorithm.pop.get("F")
                
                train_x += [p.X for p in pop ]
                train_y += [p.F for p in pop ]
                
                if i == n_gen:
                    res = algorithm.result()
                    break
        
            F = benchmark.evaluate(res.X, true_eval=True)
            run_stats['F'] = F
            run_stats['X'] = res.X.tolist()

            if pid < 8:
                # We only calculate IGD for C-10/MOP1 - C-10/MOP7 since the true Pareto Fronts are available.
                igd = benchmark.calc_perf_indicator(res.X, 'igd')
                run_stats['IGD'] = igd

            hv = benchmark.calc_perf_indicator(res.X, 'hv')
            run_stats['HV'] = hv
            experiment_stats.append(run_stats)

            print("Run {} finished.".format(r), "\t HV = {}".format(hv))

            if _DEBUG:
                print("Final population objectives:")
                print(F)
                if pid < 8:
                    print("IGD metric = {}".format(igd))
                print("HV metric = {}".format(hv))
                # hv2 = benchmark.calc_perf_indicator(res.X, 'normalized_hv')
                # print("Normalized HV metric = {}".format(hv2))
                from pymoo.visualization.scatter import Scatter

                plot = Scatter()
                if pid < 8:
                    pf = benchmark.pareto_front
                    sort_idx = np.argsort(pf[:, 0])

                    # plot.add(pf[sort_idx], plot_type="line", color="black", alpha=0.7)
                    plot.add(pf[sort_idx], facecolor="none", edgecolor="black", alpha=0.7)

                plot.add(F, facecolor="none", edgecolor="red")
                plot.show()

        with open('c10mop{}_{}_31runs.json'.format(pid, args.moea), 'w') as fp:
            json.dump(experiment_stats, fp, indent=4, cls=NumpyEncoder)
        
        print('end running problem {}'.format(pid), ' time = ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') )
  