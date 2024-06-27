from evoxbench.database.init import config
from evoxbench.test_suites import citysegmop

import os
import sys
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm

from core import Evaluator
from utilities import Debugger

from ims_lomonas import IMS_LOMONAS
from pymoo.core.problem import Problem


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class CitySegMOPProblem(Problem):
    def __init__(self, benchmark, **kwargs):
        super().__init__(
            n_var=benchmark.search_space.n_var,
            n_obj=benchmark.evaluator.n_objs,
            n_constr=0,
            xl=benchmark.search_space.lb,
            xu=benchmark.search_space.ub,
            type_var=np.int64,
            **kwargs)

        self.benchmark = benchmark

    def _evaluate(self, x, out, *args, **kwargs):
        F = self.benchmark.evaluate(x, true_eval=False)
        out["F"] = F

    def sample(self):
        raw_genotype = self.benchmark.search_space.sample(1)
        genotype = self.benchmark.search_space.encode(raw_genotype)[-1]
        return genotype


def main(kwargs):
    database_path, data_path = kwargs.database_path, kwargs.data_path
    config(database_path, data_path)

    test_suite = 'cityseg'
    pid = kwargs.pid
    benchmark = citysegmop(pid)
    problem = CitySegMOPProblem(benchmark=benchmark)

    max_eval = kwargs.max_eval
    res_path = kwargs.res_path
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    verbose = bool(kwargs.verbose)

    list_hv = []

    debugger = Debugger(verbose=bool(kwargs.verbose))

    configurations = {'problem': {'test_suite': test_suite, 'pid': pid,
                                  'search_space': problem.benchmark.name,
                                  'objectives': list(problem.benchmark.evaluator.objs.split('&'))},
                      'environment': {'max_eval': max_eval}}

    experiment_stats = []
    seeds = json.load(open('seed.json'))[f'{pid}']
    for r in tqdm(range(1, kwargs.runs + 1)):
        seed = seeds[r-1]
        logging.info(f'Run: {r}')
        logging.info(f'Seed: {seed}')
        evaluator = Evaluator(problem)
        algo = IMS_LOMONAS(base=kwargs.base,
                           neighborhood_check_on_potential_sols=True,
                           check_limited_neighbors=True,
                           alpha=kwargs.alpha, evaluator=evaluator, debugger=debugger)

        run_stats = {'run': r, 'seed': seed, 'max_eval': max_eval}

        configurations['optimizer'] = algo.hyperparameters
        if r == 1:
            logging.info('Configurations:')
            print(json.dumps(configurations, indent=1))

        res = algo.solve(problem=problem, seed=seed, max_eval=max_eval, test_suite=test_suite, verbose=verbose)

        X = res.genotype_list
        F = benchmark.evaluate(X, true_eval=True)

        hv = benchmark.calc_perf_indicator(X, "hv")
        list_hv.append(hv)

        logging.info(f'HV: {hv}')
        print("-" * 196)

        run_stats["X"] = X
        run_stats["F"] = F
        run_stats["HV"] = hv
        experiment_stats.append(run_stats)

        ################################################## Log results #################################################
        with open(f'{res_path}/{test_suite}mop{pid}_ims-lomonas.json', 'w') as fp:
            json.dump(experiment_stats, fp, indent=4, cls=NumpyEncoder)

    logging.info(f'Average HV: {np.round(np.mean(list_hv), 4)} ({np.round(np.std(list_hv), 4)})')
    print("-" * 196)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--pid', type=int, default=1)
    parser.add_argument('--max_eval', type=int, default=10000)

    ''' ALGORITHM '''
    parser.add_argument('--base', type=int, default=8)
    parser.add_argument('--alpha', type=int, default=210)

    parser.add_argument('--verbose', action='store_true')

    ''' ENVIRONMENT '''
    parser.add_argument('--runs', type=int, default=31)
    parser.add_argument('--database_path', type=str, default='./database', help='path for loading api benchmark (database CEC)')
    parser.add_argument('--data_path', type=str, default='./data', help='path for loading api benchmark (data CEC)')
    parser.add_argument('--res_path', type=str, default='./exp', help='path for saving results')
    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    main(args)