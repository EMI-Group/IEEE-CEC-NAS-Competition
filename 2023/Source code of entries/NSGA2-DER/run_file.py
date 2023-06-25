import json
import os.path
import shutil

from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.optimize import minimize

from utils import folder_create, NumpyEncoder, Record, md5

method_list = [
    'MOEAD',
    'NSGA3',  # ERROR for C10_MOP1 and C10_MOP2
    'NSGA2',
    'RVEA',  # ERROR for C10_MOP1 and C10_MOP2
    'AGEMOEA',
    'AGEMOEA2',
    'CTAEA',
    'SMSEMOA',
    'SPEA2',
    'DNSGA2',
    'UNSGA3',  # ERROR for C10_MOP1 and C10_MOP2
]


# problem_list = [f'C10_MOP{i}' for i in range(1, 10)] + [f'IN1K_MOP{i}' for i in range(1, 10)]


def main(runs, problem, method):
    assert method in method_list
    records = []
    folder_create(f'EXP-{method}/')
    output_folder = folder_create(f'EXP-{method}/' + problem.name)
    for r in range(1, runs + 1):
        run_stats = {'run': r}
        benchmark = problem.benchmark
        ref_dirs = problem.ref_dirs
        if method == 'MOEAD':
            algorithm = MOEAD(ref_dirs=ref_dirs, sampling=problem.sampling, crossover=problem.crossover,
                              mutation=problem.mutation)
        elif method == 'NSGA3':
            algorithm = NSGA3(ref_dirs=ref_dirs, sampling=problem.sampling, crossover=problem.crossover,
                              mutation=problem.mutation)
        elif method == 'NSGA2':
            algorithm = NSGA2(pop_size=problem.pop_size, sampling=problem.sampling, crossover=problem.crossover,
                              mutation=problem.mutation)
        elif method == 'RVEA':
            algorithm = RVEA(ref_dirs=ref_dirs, sampling=problem.sampling, crossover=problem.crossover,
                             mutation=problem.mutation)
        elif method == 'AGEMOEA':
            algorithm = AGEMOEA(pop_size=problem.pop_size, sampling=problem.sampling, crossover=problem.crossover,
                                mutation=problem.mutation)
        elif method == 'AGEMOEA2':
            algorithm = AGEMOEA2(pop_size=problem.pop_size, sampling=problem.sampling, crossover=problem.crossover,
                                 mutation=problem.mutation)
        elif method == 'CTAEA':
            algorithm = CTAEA(ref_dirs=ref_dirs, sampling=problem.sampling, crossover=problem.crossover,
                              mutation=problem.mutation)
        elif method == 'SMSEMOA':
            algorithm = SMSEMOA(pop_size=problem.pop_size, sampling=problem.sampling, crossover=problem.crossover,
                                mutation=problem.mutation)
        elif method == 'SPEA2':
            algorithm = SPEA2(pop_size=problem.pop_size, sampling=problem.sampling, crossover=problem.crossover,
                              mutation=problem.mutation)
        elif method == 'DNSGA2':
            algorithm = DNSGA2(pop_size=problem.pop_size, sampling=problem.sampling, crossover=problem.crossover,
                               mutation=problem.mutation)
        elif method == 'UNSGA3':
            algorithm = UNSGA3(ref_dirs=ref_dirs, sampling=problem.sampling, crossover=problem.crossover,
                               mutation=problem.mutation)
        else:
            raise Exception()

        res = minimize(problem, algorithm, ('n_gen', problem.n_gen), verbose=True)
        res.X = res.X.astype(int)
        F = benchmark.evaluate(res.X, true_eval=True)
        run_stats['F'] = F
        run_stats['X'] = res.X

        # only PID<8
        if problem.pid <= 7 and problem.name.startswith('C10'):
            igd = benchmark.calc_perf_indicator(res.X, 'igd')
        else:
            igd = None
        run_stats['IGD'] = igd
        hv = benchmark.calc_perf_indicator(res.X, 'hv')
        run_stats['HV'] = hv
        # with open(os.path.join(output_folder, f'run_{r}.json'), 'w') as fp:
        with open(os.path.join(output_folder, f'run_{md5(str(run_stats).encode())}.json'), 'w') as fp:
            json.dump(run_stats, fp, indent=4, cls=NumpyEncoder)
        # m = md5('template.sh')
        # shutil.move(os.path.join(output_folder, f'run_{r}.json'), os.path.join(output_folder, f'{m}.json'))
        records.append(Record(r, F, hv, igd))
    with open(os.path.join(output_folder, f'info.json'), 'w') as fp:
        json.dump({
            'runs': runs,
            'database': problem.database,
            'name': problem.name,
            'n_var': problem.n_var,
            'objs': problem.objs,
            'pop_size': problem.pop_size,
            'off_spring': problem.off_spring,
            'n_gen': problem.n_gen
        }, fp, indent=4)
    df = Record.to_pd(records)
    df.to_csv(os.path.join(output_folder, f'results.csv'))
    # print(df.mean(), df.std())


if __name__ == '__main__':
    from competition.C10_MOP1 import C10_MOP1
    from competition.C10_MOP2 import C10_MOP2
    from competition.C10_MOP3 import C10_MOP3
    from competition.C10_MOP4 import C10_MOP4
    from competition.C10_MOP5 import C10_MOP5
    from competition.C10_MOP6 import C10_MOP6
    from competition.C10_MOP7 import C10_MOP7
    from competition.C10_MOP8 import C10_MOP8
    from competition.C10_MOP9 import C10_MOP9
    from competition.IN1K_MOP1 import IN1K_MOP1
    from competition.IN1K_MOP2 import IN1K_MOP2
    from competition.IN1K_MOP3 import IN1K_MOP3
    from competition.IN1K_MOP4 import IN1K_MOP4
    from competition.IN1K_MOP5 import IN1K_MOP5
    from competition.IN1K_MOP6 import IN1K_MOP6
    from competition.IN1K_MOP7 import IN1K_MOP7
    from competition.IN1K_MOP8 import IN1K_MOP8
    from competition.IN1K_MOP9 import IN1K_MOP9

    problem_list = [
        C10_MOP1(),
        # C10_MOP2(),
        # C10_MOP3(),
        # C10_MOP4(),
        # C10_MOP5(),
        # C10_MOP6(),
        # C10_MOP7(),
        # C10_MOP8(),
        # C10_MOP9(),
        # IN1K_MOP1(),
        # IN1K_MOP2(),
        # IN1K_MOP3(),
        # IN1K_MOP4(),
        # IN1K_MOP5(),
        # IN1K_MOP6(),
        # IN1K_MOP7(),
        # IN1K_MOP8(),
        # IN1K_MOP9(),
    ]
    for i in range(len(problem_list)):
        for method in method_list:
            try:
                print(i, method, '######################')
                problem_list[i].n_gen = 3
                main(1, problem_list[i], method)
            except Exception:
                print('ERROR', method, problem_list[i].name)
