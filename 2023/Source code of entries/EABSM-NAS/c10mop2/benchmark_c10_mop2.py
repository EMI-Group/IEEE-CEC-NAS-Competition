import os
import sys
import random
import datetime
import numpy as np
os.chdir('/home/wcx/gitProject/evoxbench/evoxbench_submit/c10mop2')
sys.path.insert(0, '/home/wcx/gitProject/evoxbench/evoxbench_submit/') # for evoxbench
sys.path.insert(0, '/home/wcx/gitProject/evoxbench/evoxbench_submit/pymoo050/') # for pymoo0.5.0

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.factory import get_algorithm, get_sampling, get_crossover, get_mutation, get_reference_directions
from pymoo.config import Config
Config.show_compile_hint = False

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
        self.hash_X_history = {}

    def _evaluate(self, x, out, *args, **kwargs):

        F = self.benchmark.evaluate(x, true_eval=True)

        out["F"] = F

        self._add_history(x)
    
    def _add_history(self, X):
        hash_keys = self._get_hash_x(X)
        for hash_key in hash_keys:
            self.hash_X_history[hash_key] = True
    
    def _get_valid_x(self, X):
        # convert genotype X to architecture phenotype
        archs = self.benchmark.search_space.decode(X)
        valid_idx = []

        for i, arch in enumerate(archs):
            model_spec = NASBench101Graph(matrix=arch['matrix'], ops=arch['ops'])
            if model_spec.is_valid():
                valid_idx.append(i)

        return valid_idx

    # get the vaild archs not in the history by hash
    def _get_fine_x(self, X):
        valid_idx = self._get_valid_x(X)
        
        tmp_hash_history = {}
        
        fine_idx = []
        for i in valid_idx:
            model_key = self._get_hash_x([X[i]])[0]
            if model_key not in tmp_hash_history:
                tmp_hash_history[model_key] = True
            else:
                continue
            if model_key not in self.hash_X_history:
                fine_idx.append(i)
                
        return fine_idx

    def _get_hash_arch(self, arch):
        model_spec = NASBench101Graph(matrix=arch['matrix'], ops=arch['ops'])
        model_key = model_spec.hash_spec(['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'])
        return model_key

    def _get_hash_x(self, X):
        hash_keys = []
        archs = self.benchmark.search_space.decode(X)
        for arch in archs:
            model_key = self._get_hash_arch(arch)
            hash_keys.append(model_key)
        return hash_keys
        
def get_genetic_operator(crx_prob=1.0,  # crossover probability
                         crx_eta=30.0,  # SBX crossover eta
                         mut_prob=None,  # mutation probability
                         mut_eta=20.0,  # polynomial mutation hyperparameter eta
                         ):
    sampling = get_sampling('int_lhs')
    crossover = get_crossover('int_sbx', prob=crx_prob, eta=crx_eta)
    mutation = get_mutation('int_pm', eta=mut_eta, prob=mut_prob)
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
          mut_prob=None,  # mutation probability, i.e., 1/n
          mut_eta=20.0,  # polynomial mutation hyperparameter eta
          ):
    sampling, crossover, mutation = get_genetic_operator(crx_prob, crx_eta, mut_prob, mut_eta)

    return get_algorithm(
        "nsga2", pop_size=pop_size, sampling=sampling, crossover=crossover,
        mutation=mutation, eliminate_duplicates=True)

def moead(ref_dirs,
          crx_prob=1.0,  # crossover probability
          crx_eta=20.0,  # SBX crossover eta
          mut_prob=None,  # mutation probability, i.e., 1/n
          mut_eta=20.0,  # polynomial mutation hyperparameter eta
          neighborhood_size=20,  # neighborhood size
          prob_neighbor_mating=0.9,  # neighborhood selection probability
          ):
    sampling, crossover, mutation = get_genetic_operator(crx_prob, crx_eta, mut_prob, mut_eta)
    return get_algorithm(
        "moead", ref_dirs=ref_dirs, n_neighbors=neighborhood_size, prob_neighbor_mating=prob_neighbor_mating,
        sampling=sampling, crossover=crossover, mutation=mutation)

def nsga3(pop_size,
          ref_dirs,
          crx_prob=1.0,  # crossover probability
          crx_eta=30.0,  # SBX crossover eta
          mut_prob=None,  # mutation probability, i.e., 1/n
          mut_eta=20.0,  # polynomial mutation hyperparameter eta
          ):
    sampling, crossover, mutation = get_genetic_operator(crx_prob, crx_eta, mut_prob, mut_eta)

    return get_algorithm(
        'nsga3', pop_size=pop_size, ref_dirs=ref_dirs, sampling=sampling, crossover=crossover,
        mutation=mutation, eliminate_duplicates=True)


if __name__ == '__main__':
    import pickle
    d = './'
    err_models = pickle.load(open(d+'err_models_2000.pkl', 'rb'))
    pd_91_models = pickle.load(open(d+'pd_91_models_2000.pkl', 'rb'))
    def pd_X_more_than_91(X):
        pred_y_list = []
        for model in pd_91_models:
            pred_y = model.predict(X)
            pred_y_list.append(pred_y)
        return np.around(np.array(pred_y_list).mean(axis=0))
    def get_pred_err(X):
        pred_y_list = []
        for model in err_models:
            pred_y = model.predict(X)
            pred_y_list.append(pred_y)
        return np.array(pred_y_list).mean(axis=0)
    def get_end_err(X):
        pd = pd_X_more_than_91(X)
        pd_err = []
        for f, x in zip(pd, X):
            if f == 1:
                pd_err.append(get_pred_err([x])[0])
            else:
                pd_err.append(max(get_pred_err([x])[0], 0.09) )
        return np.array(pd_err)
    def get_pred_params_direct(X):
        return benchmark.evaluate(X, true_eval=True)[:, 1]
    def get_pred_flops_direct(X):
        return benchmark.evaluate(X, true_eval=True)[:, 2]
    
    import json
    import argparse

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    pid = 2
    parser = argparse.ArgumentParser(description='Benchmark C10/MOP')
    parser.add_argument('--moea', type=str, default='nsga2', help='which MOEA to run')
    parser.add_argument('--runs', type=int, default=31, help='number of runs to repeat')
    args = parser.parse_args()
    
    experiment_stats = []
    print('start to run problem {}'.format(pid), ' time = ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') )
    import tqdm
    for r in tqdm.trange(1, args.runs+1):
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
        
        hv_list = []
        
        hashs = []
        fitness = []
        pops = []

        searched_arch = set()
        algorithm.setup(problem=problem, seed=r, verbose=True)
        n_gen = 80 # using offline model
        aux = True
        for i in range(1, n_gen+1):
            pop = algorithm.ask()
            
            if pop is None:
                res = algorithm.result()
                break      
            
            if aux:
                # get triple individuals
                from pymoo.core.population import Population
                pop = Population.merge(pop, algorithm.ask())
                pop = Population.merge(pop, algorithm.ask())
            
                pred_param = get_pred_params_direct(pop.get('X'))
                pred_err = get_end_err(pop.get('X'))
                pred_flops = get_pred_flops_direct(pop.get('X'))
                
                norm_pred = np.hstack([pred_err.reshape(-1, 1), pred_param.reshape(-1, 1),
                                       pred_flops.reshape(-1, 1)])
                norm_sum = norm_pred.sum(axis=1)
                
                # rank scheme
                from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
                cur_preds = np.array(norm_pred)
                cur_X = np.array(pop.get('X'))
                topk_X, topk_pred = [], []

                while True:
                    nd_front = NonDominatedSorting().do(cur_preds, only_non_dominated_front=True)
                    t_X = cur_X[nd_front]
                    t_pred = cur_preds[nd_front]
                    
                    cur_preds = cur_preds[[j for j in range(len(cur_preds)) if j not in nd_front]]
                    cur_X = cur_X[[j for j in range(len(cur_X)) if j not in nd_front]]
                    print(len(topk_X), '\t', len(nd_front))
                    
                    if len(topk_X) + len(t_X) < pop_size:
                        topk_X.extend(t_X)
                        topk_pred.extend(t_pred)
                    else:
                        rand_idx = np.random.choice(len(t_X), size=pop_size-len(topk_X), replace=False)
                        topk_X.extend(t_X[rand_idx])
                        topk_pred.extend(t_pred[rand_idx])
                        break
                
                # topk = sorted(zip(norm_sum, range(len(pop))), key=lambda x:x[0])[:pop_size]
                topk_X, topk_pred = topk_X[:pop_size], topk_pred[:pop_size]
                topk_tuple = set([tuple(p) for p in topk_X])
                cur_X = pop.get('X')
                need_X = [idx for idx,x in enumerate(cur_X) if tuple(x) in topk_tuple]
                pop = pop[need_X]
            
            # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            
            algorithm.evaluator.eval(problem, pop)

            f = pop.get('F')
            fitness.extend(f)

            algorithm.tell(infills=pop)
            hv_pop = benchmark.calc_perf_indicator050(algorithm.pop.get("X"), 'hv')
            hv_list.append(hv_pop)
            print('hv: ', hv_pop)
            
            F = algorithm.pop.get("F")
            
            if (i == n_gen):
                res = algorithm.result()
                break
    
        F = benchmark.evaluate(res.X, true_eval=True)
        run_stats['F'] = F

        if pid < 8:
            # We only calculate IGD for C-10/MOP1 - C-10/MOP7 since the true Pareto Fronts are available.
            igd = benchmark.calc_perf_indicator050(res.X, 'igd')
            run_stats['IGD'] = igd

        hv = benchmark.calc_perf_indicator050(res.X, 'hv')
        run_stats['HV'] = hv
        run_stats['X'] = res.X
        experiment_stats.append(run_stats)

        print("Run {} finished.".format(r), "\t HV = {}".format(hv))
    
    d = './'
    with open(d + '{}_{}.json'.format(args.moea, args.runs), 'w') as f:
        json.dump(experiment_stats, f, cls=NumpyEncoder)
    
    print('end running problem {}'.format(pid), ' time = ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') )
