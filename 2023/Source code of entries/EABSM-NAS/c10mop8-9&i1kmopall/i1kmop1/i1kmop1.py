import os
import sys
import random
import datetime
import numpy as np
os.chdir('/home/wcx/gitProject/evoxbench/evoxbench_submit/c10mop8-9&i1k/i1kmop1')
sys.path.insert(0, '/home/wcx/gitProject/evoxbench/evoxbench_submit/') # for evoxbench
sys.path.insert(0, '/home/wcx/gitProject/evoxbench/evoxbench_submit/pymoo050/') # for pymoo0.5.0

import pickle
import datetime
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.factory import get_algorithm, get_sampling, get_crossover, get_mutation, get_reference_directions
from pymoo.config import Config
from pymoo.core.population import Population
Config.show_compile_hint = False

from evoxbench.test_suites import in1kmop

class In1KMOPProblem(Problem):
    def __init__(self,
                 benchmark,
                 pid,
                 **kwargs):
        super().__init__(n_var=benchmark.search_space.n_var, n_obj=benchmark.evaluator.n_objs,
                         n_constr=0, xl=benchmark.search_space.lb, xu=benchmark.search_space.ub,
                         type_var=np.int64, **kwargs)
        self.pid = pid
        self.benchmark = benchmark
        self.X_history = {}
        self.hash_X_history = {}

    def _evaluate(self, x, out, *args, **kwargs):

        F = self.benchmark.evaluate(x, true_eval=True)

        out["F"] = F

    def _add_history(self, X):
        for x in X:
            model_key = self._get_hash_x([x])[0]
            if model_key not in self.hash_X_history:
                self.hash_X_history[model_key] = True

    def _get_valid_x(self, X):
        return range(len(X))
        # convert genotype X to architecture phenotype
        archs = self.benchmark.search_space.decode(X)
        valid_idx = []

        for i, arch in enumerate(archs):
            model_spec = NASBench101Graph(matrix=arch['matrix'], ops=arch['ops'])
            if model_spec.is_valid():
                valid_idx.append(i)

        return valid_idx
    
    def _get_fine_x(self, X):
        valid_idx = self._get_valid_x(X)
        
        tmp_X_history = {}
        
        fine_idx = []
        for i in valid_idx:
            model_key = self._get_hash_x([X[i]])[0]
            if model_key not in tmp_X_history:
                tmp_X_history[model_key] = True
            else:
                continue
            if model_key not in self.hash_X_history:
                fine_idx.append(i)
                
        return fine_idx

    def _get_hash_x(self, X):
        return [tuple(i) for i in X]
    
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
    import json
    import argparse

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    pid = 1
    parser = argparse.ArgumentParser(description='Benchmark IN1K/MOP')
    parser.add_argument('--moea', type=str, default='nsga2', help='which MOEA to run')
    parser.add_argument('--runs', type=int, default=31, help='number of runs to repeat')
    args = parser.parse_args()

    d = './'
    err_models = pickle.load(open(d + 'err_models_2000.pkl', 'rb'))
    def get_pred_err(X):
        pred_y_list = []
        for model in err_models:
            pred_y = model.predict(X)
            pred_y_list.append(pred_y)
        return np.array(pred_y_list).mean(axis=0)
    def get_pred_params_direct(X):
        return benchmark.evaluate(X)[:, 1]
    def get_pred_flops_direct(X):
        return benchmark.evaluate(X)[:, 2]

    experiment_stats = []

    print('start to run problem {}'.format(pid), ' time = ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') )
    import tqdm
    for r in tqdm.trange(1, args.runs+1):
        run_stats = {'run': r}

        benchmark = in1kmop(pid)
        problem = In1KMOPProblem(benchmark, pid)

        pop_size, n_gen, ref_dirs = get_benchmark_settings(problem.n_obj)

        if args.moea == 'nsga2':
            algorithm = nsga2(pop_size)
        elif args.moea == 'moead':
            algorithm = moead(ref_dirs)
        elif args.moea == 'nsga3':
            algorithm = nsga3(pop_size, ref_dirs)
        else:
            raise NotImplementedError

        hv_list = []
        fitness = []

        # res = minimize(problem, algorithm, ('n_gen', n_gen), verbose=True)
        searched_arch = set()
        algorithm.setup(problem=problem, seed=r, verbose=True)
        aux = True
        n_gen = 80
        for i in range(1, n_gen+1):
            
            pop = algorithm.ask()
            if pop is None:
                res = algorithm.result()
                break
             
            if aux:
                pop = Population.merge(pop, algorithm.ask())
                pop = Population.merge(pop, algorithm.ask())
            
                pred_param = get_pred_params_direct(pop.get('X'))
                pred_err = get_pred_err(pop.get('X'))
                # pred_flops = get_pred_flops_direct(pop.get('X'))
                
                norm_pred = benchmark.normalize(np.hstack([pred_err.reshape(-1, 1), pred_param.reshape(-1, 1)] ))
                norm_sum = norm_pred.sum(axis=1)
                
                # rank scheme: non_dominated_sorting
                from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
                cur_preds = np.array(norm_pred)
                cur_X = np.array(pop.get('X'))
                topk_X, topk_pred = [], []

                while True:
                    nd_front = NonDominatedSorting().do(cur_preds, only_non_dominated_front=True)
                    t_X = cur_X[nd_front]
                    t_pred = cur_preds[nd_front]
                    
                    cur_preds = cur_preds[[i for i in range(len(cur_preds)) if i not in nd_front]]
                    cur_X = cur_X[[i for i in range(len(cur_X)) if i not in nd_front]]
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
            
            algorithm.evaluator.eval(problem, pop)
            f = pop.get('F')
            fitness.extend(f)
            
            algorithm.tell(infills=pop)
            
            hv_pop = benchmark.calc_perf_indicator050(algorithm.pop.get("X"), 'hv')
            hv_list.append(hv_pop)
            print('hv: ', hv_pop)
            
            if i == n_gen:
                res = algorithm.result()
                break
            
        remains = 10000 - len(searched_arch)
        
        F = benchmark.evaluate(res.X, true_eval=True)
        run_stats['F'] = F

        hv = benchmark.calc_perf_indicator050(res.X, 'hv') # EP
        run_stats['HV'] = hv
        run_stats['X'] = res.X
        experiment_stats.append(run_stats)
    
    d = './'
    with open(d + 'c10mop_{}_online_ea.json'.format(args.runs), 'w') as fp:
        json.dump(experiment_stats, fp, indent=4, cls=NumpyEncoder)
    print('end running problem {}'.format(pid), ' time = ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') )