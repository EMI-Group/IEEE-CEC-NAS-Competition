"""
Source code for Local-search algorithm for Multi-Objective Neural Architecture Search (LOMONAS)
Authors: Quan Minh Phan, Ngoc Hoang Luong
doi: 10.1145/3583131.3590395
"""
import random
import numpy as np
from copy import deepcopy
import itertools
from functions import not_existed
from utilities import Result, Footprint
from core import ElitistArchive, Solution

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.misc import find_duplicates

sorter = NonDominatedSorting()
#################################################### LOMONAS #######################################################
class LOMONAS:
    def __init__(self, name='LOMONAS',
                 k=3, check_limited_neighbors=True, neighborhood_check_on_potential_sols=True, alpha=210,
                 archive=None, footprint=None, debugger=None, res_logged=None,
                 evaluator=None, **kwargs):
        """
        - name (str) -> the algorithm name (i.e., LOMONAS)
        - k (int) -> number of kept front for neighborhood checking
        - check_limited_neighbors (bool) -> checking a limited neighbors when local search?
        - neighborhood_check_on_potential_sols (bool) -> local search on potential or all solutions?
        - alpha (int, [0, 360]) -> angle for checking knee solution or not
        """
        self.name = name

        self.k = k
        self.check_limited_neighbors = check_limited_neighbors
        self.neighborhood_check_on_potential_sols = neighborhood_check_on_potential_sols
        self.alpha = alpha

        self.debugger = debugger
        self.footprint = Footprint() if footprint is None else footprint
        self.res_logged = [] if res_logged is None else res_logged

        self.evaluator = evaluator
        self.local_archive = ElitistArchive()
        self.archive = ElitistArchive() if archive is None else archive
        self.last_archive = None

        self.S, self.Q = [], []

        self.problem, self.max_eval = None, None

        self.solutions_collector = None
        self.neighbors_collector = None

        self.n_eval = 0

        self.last_S_fid, self.last_Q = [], []

    @property
    def hyperparameters(self):
        return {
            'optimizer': self.name,
            'k': self.k,
            'check_limited_neighbors': self.check_limited_neighbors,
            'neighborhood_check_on_potential_sols': self.neighborhood_check_on_potential_sols,
            'alpha': self.alpha,
        }

    def set(self, key_value):
        for key, value in key_value.items():
            if key in self.__dict__:
                self.__dict__[key] = value

    """-------------------------------------------------- SETUP -----------------------------------------------"""
    def setup(self, problem, max_eval, **kwargs):
        self.problem, self.max_eval = problem, max_eval
        self._setup(**kwargs)

    def _setup(self, **kwargs):
        if self.neighborhood_check_on_potential_sols:  # Only performing neighborhood check on knee and extreme ones
            self.solutions_collector = get_potential_solutions
        else:
            self.solutions_collector = get_all_solutions

        if self.check_limited_neighbors:
            self.neighbors_collector = get_some_neighbors
        else:
            self.neighbors_collector = get_all_neighbors

    """------------------------------------------------- EVALUATE --------------------------------------------"""
    def evaluate(self, solution):
        self.n_eval += 1
        self.evaluator(solution)

        if self.evaluator.n_eval % 100 == 0:
            self.debugger(algorithm=self)

    """-------------------------------------------------- SOLVE -----------------------------------------------"""
    def solve(self, problem, max_eval, seed, **kwargs):
        random.seed(seed)
        np.random.seed(seed)

        self.setup(problem, max_eval)  # Setup (general)

        self._solve(**kwargs)
        genotype_archive = [elitist.X for elitist in self.archive.archive]
        res = Result(genotype_archive)
        return res

    def _solve(self, **kwargs):
        first = True
        while not self.isTerminated():  # line 5 - 27
            self.initialize(first)  # Sample new starting solution for the next local search
            first = False
            isContinued = True
            while isContinued:
                isContinued = self.neighborhood_checking()

    """-------------------------------------------------- UTILITIES -----------------------------------------------"""
    def isTerminated(self):
        if self.evaluator.n_eval >= self.max_eval:
            return True
        return False

    def update_archive(self, solution):
        self.local_archive.update(solution)
        self.archive.update(solution)

    def initialize(self, first=True):
        start_solution = self.sample_starting_solution(first=first)  # Random a starting solution (line 3)

        # lines 6, 7
        self.S, self.Q = [start_solution], [start_solution]  # approximation set (S) and queue for neighborhood check (Q)
        self.last_archive = deepcopy(self.local_archive)

    def sample_starting_solution(self, first=False):
        if first:
            start_solution = sample_solution(self.footprint.data, self.problem)
            start_solution.set('owner', self.name)
        else:
            # lines 16 - 21
            N = []

            ## Choose one elitist in the archive
            available_idx = list(range(len(self.archive.archive)))
            found_new_start = False
            while len(available_idx) != 0:
                idx = np.random.choice(available_idx)
                available_idx.remove(idx)
                selected_solution = deepcopy(self.archive.archive[idx])
                tmp_N, _ = get_all_neighbors(solution=selected_solution, H={}, problem=self.problem)
                N = [neighbor for neighbor in tmp_N if neighbor.genotypeHash not in self.footprint.data]

                if len(N) != 0:  # If all neighbors of chosen elitist are not visited, choose a random neighbor as new starting solution.
                    found_new_start = True
                    break
            if not found_new_start:  # If not, randomly sampling from the search space.
                start_solution = sample_solution(self.footprint.data, self.problem)
            else:
                idx_selected_neighbor = np.random.choice(len(N))
                start_solution = N[idx_selected_neighbor]
            start_solution.set('owner', self.name)

        self.evaluate(start_solution)
        self.update_archive(start_solution)
        return start_solution

    def neighborhood_checking(self):
        N = self.get_neighbors()  # N: neighboring set, line 9

        # lines 10 - 22
        if len(N) == 0:
            # lines 11 - 15
            for fid in range(1, self.k):
                self.Q = self.create_Q(fid=fid)

                N = self.get_neighbors()
                if len(N) != 0:
                    break

            if len(N) == 0:
                return False

        # line 23
        for neighbor in N:
            self.evaluate(neighbor)
            self.update_archive(neighbor)
            if self.isTerminated():
                return False

        # lines 24, 25
        self.create_S(N)

        self.Q = self.create_Q(fid=0)
        return True

    def create_S(self, N):
        P = self.S + N
        F_P = [s.F for s in P]
        idx_fronts = sorter.do(np.array(F_P))
        idx_selected = np.zeros(len(F_P), dtype=bool)
        k = min(len(idx_fronts), self.k)
        for fid in range(k):
            idx_selected[idx_fronts[fid]] = True
            for idx in idx_fronts[fid]:
                P[idx].set('rank', fid)
        self.S = np.array(P)[idx_selected].tolist()
        self.S = remove_duplicate(self.S)

    def create_Q(self, fid):
        Q, last_S_fid, duplicated = self.solutions_collector(S=self.S, fid=fid, alpha=self.alpha,
                                                             last_S_fid=self.last_S_fid, last_Q=self.last_Q)
        if not duplicated:
            self.last_Q, self.last_S_fid = deepcopy(Q), last_S_fid.copy()

        return Q

    def get_neighbors(self):
        """ Get neighbors of all solutions in queue Q, but discard solutions that has been already in H """
        _H = self.footprint.data
        N = []
        for solution in self.Q:
            tmp_N, _H = self.neighbors_collector(solution, _H, self.problem)

            # Remove duplication
            genotypeHash_S = [s.genotypeHash for s in self.S]
            genotypeHash_N = [s.genotypeHash for s in N]
            for neighbor in tmp_N:
                if not_existed(neighbor.genotypeHash, S=genotypeHash_S, N=genotypeHash_N):
                    N.append(neighbor)
                    genotypeHash_N.append(neighbor.genotypeHash)
        self.footprint.data = _H
        return N

#####################################################################################
def seeking(list_sol, alpha):
    list_sol = np.array(list_sol)
    non_dominated_front = np.array([solution.F for solution in list_sol])

    ids = range(non_dominated_front.shape[-1])
    info_potential_sols_all = []
    for f_ids in itertools.combinations(ids, 2):
        f_ids = np.array(f_ids)
        obj_1, obj_2 = f'{f_ids[0]}', f'{f_ids[1]}'

        _non_dominated_front = non_dominated_front[:, f_ids].copy()

        ids_sol = np.array(list(range(len(list_sol))))
        ids_fr0 = sorter.do(_non_dominated_front, only_non_dominated_front=True)

        ids_sol = ids_sol[ids_fr0]
        _non_dominated_front = _non_dominated_front[ids_fr0]

        sorted_idx = np.argsort(_non_dominated_front[:, 0])

        ids_sol = ids_sol[sorted_idx]
        _non_dominated_front = _non_dominated_front[sorted_idx]

        min_values, max_values = np.min(_non_dominated_front, axis=0), np.max(_non_dominated_front, axis=0)
        _non_dominated_front_norm = (_non_dominated_front - min_values) / (max_values - min_values)

        info_potential_sols = [
            [0, list_sol[ids_sol[0]], f'best_f{obj_1}']  # (idx (in full set), property)
        ]

        l_non_front = len(_non_dominated_front)
        for i in range(l_non_front - 1):
            if np.sum(np.abs(_non_dominated_front[i] - _non_dominated_front[i + 1])) != 0:
                break
            else:
                info_potential_sols.append([i + 1, list_sol[ids_sol[i + 1]], f'best_f{obj_1}'])

        for i in range(l_non_front - 1, -1, -1):
            if np.sum(np.abs(_non_dominated_front[i] - _non_dominated_front[i - 1])) != 0:
                break
            else:
                info_potential_sols.append([i - 1, list_sol[ids_sol[i - 1]], f'best_f{obj_2}'])
        info_potential_sols.append([l_non_front - 1, list_sol[ids_sol[l_non_front - 1]], f'best_f{obj_2}'])

        ## find the knee solutions
        start_idx, end_idx = 0, l_non_front - 1

        for i in range(len(info_potential_sols)):
            if info_potential_sols[i + 1][-1] == f'best_f{obj_2}':
                break
            else:
                start_idx = info_potential_sols[i][0] + 1

        for i in range(len(info_potential_sols) - 1, -1, -1):
            if info_potential_sols[i - 1][-1] == f'best_f{obj_1}':
                break
            else:
                end_idx = info_potential_sols[i][0] - 1

        for i in range(start_idx, end_idx + 1):
            l = None
            h = None
            for m in range(i - 1, -1, -1):
                if np.sum(np.abs(_non_dominated_front[m] - _non_dominated_front[i])) != 0:
                    l = m
                    break
            for m in range(i + 1, l_non_front, 1):
                if np.sum(np.abs(_non_dominated_front[m] - _non_dominated_front[i])) != 0:
                    h = m
                    break

            if (h is not None) and (l is not None):
                position = above_or_below(considering_pt=_non_dominated_front[i],
                                          remaining_pt_1=_non_dominated_front[l],
                                          remaining_pt_2=_non_dominated_front[h])
                if position == -1:
                    angle_measure = calc_angle_measure(considering_pt=_non_dominated_front_norm[i],
                                                            neighbor_1=_non_dominated_front_norm[l],
                                                            neighbor_2=_non_dominated_front_norm[h])
                    if angle_measure > alpha:
                        info_potential_sols.append([i, list_sol[ids_sol[i]], 'knee'])
        info_potential_sols_all += info_potential_sols
    return info_potential_sols_all


def above_or_below(considering_pt, remaining_pt_1, remaining_pt_2):
    """
    This function is used to check if the considering point is above or below
    the line connecting two remaining points.\n
    1: above\n
    -1: below
    """
    orthogonal_vector = remaining_pt_2 - remaining_pt_1
    line_connecting_pt1_and_pt2 = -orthogonal_vector[1] * (considering_pt[0] - remaining_pt_1[0]) \
                                  + orthogonal_vector[0] * (considering_pt[1] - remaining_pt_1[1])
    if line_connecting_pt1_and_pt2 > 0:
        return 1
    return -1


def calc_angle_measure(considering_pt, neighbor_1, neighbor_2):
    """
    This function is used to calculate the angle measure is created by the considering point
    and two its nearest neighbors
    """
    line_1 = neighbor_1 - considering_pt
    line_2 = neighbor_2 - considering_pt
    cosine_angle = (line_1[0] * line_2[0] + line_1[1] * line_2[1]) \
                   / (np.sqrt(np.sum(line_1 ** 2)) * np.sqrt(np.sum(line_2 ** 2)))
    if cosine_angle < -1:
        cosine_angle = -1
    if cosine_angle > 1:
        cosine_angle = 1
    angle = np.arccos(cosine_angle)
    return 360 - np.degrees(angle)


def sample_solution(footprint_data, problem):
    while True:
        solution = Solution(X=problem.sample())
        if solution.genotypeHash not in footprint_data:
            return solution


def get_all_solutions(S, fid, **kwargs):
    Q = []
    Q_genotypeHash = []
    rank_S = np.array([s.get('rank') for s in S])
    S_front_i = np.array(S)[rank_S == fid]

    list_genotypeHash = [s.genotypeHash for s in S_front_i]
    if is_duplicated(list_genotypeHash, kwargs['last_S_fid']):
        return kwargs['last_Q'], list_genotypeHash, True

    for sol in S_front_i:
        if sol.genotypeHash not in Q_genotypeHash:
            Q_genotypeHash.append(sol.genotypeHash)
            Q.append(sol)
    return Q, list_genotypeHash, False


def get_potential_solutions(S, fid, **kwargs):
    alpha = kwargs['alpha']
    Q = []
    Q_genotypeHash = []
    rank_S = np.array([s.get('rank') for s in S])
    S_front_i = np.array(S)[rank_S == fid]

    list_genotypeHash = [s.genotypeHash for s in S_front_i]
    if is_duplicated(list_genotypeHash, kwargs['last_S_fid']):
        return kwargs['last_Q'], list_genotypeHash, True

    info_potential_sols = seeking(S_front_i, alpha)
    potential_sols = [info[1] for info in info_potential_sols]
    for i, sol in enumerate(potential_sols):
        if sol.genotypeHash not in Q_genotypeHash:
            Q_genotypeHash.append(sol.genotypeHash)
            Q.append(sol)
    return Q, list_genotypeHash, False


## Get neighboring architectures
def get_some_neighbors(solution, H, problem):
    X, genotypeHash = solution.X, solution.genotypeHash
    N = []

    if genotypeHash in H:
        if len(H[genotypeHash]) == 0:
            return [], H
        available_idx = H[genotypeHash]
        idx_replace = np.random.choice(available_idx)
        H[genotypeHash].remove(idx_replace)
    else:
        available_idx = list(range(len(X)))
        H[genotypeHash] = available_idx
        idx_replace = np.random.choice(H[genotypeHash])
        H[genotypeHash].remove(idx_replace)

    available_ops = problem.benchmark.search_space.categories[idx_replace]
    available_ops_at_idx_replace = available_ops.copy()
    available_ops_at_idx_replace.remove(X[idx_replace])
    for op in available_ops_at_idx_replace:
        X = solution.X.copy()
        X[idx_replace] = op
        neighbor = Solution(X=X)
        N.append(neighbor)
    return N, H


def get_all_neighbors(solution, H, problem):
    X, genotypeHash = solution.X, solution.genotypeHash
    if genotypeHash in H:
        return [], H
    else:
        H[genotypeHash] = []
    N = []

    available_idx = list(range(len(X)))

    for idx_replace in available_idx:
        available_ops = problem.benchmark.search_space.categories[idx_replace]

        available_ops_at_idx_replace = available_ops.copy()
        available_ops_at_idx_replace.remove(X[idx_replace])
        for op in available_ops_at_idx_replace:
            X = solution.X.copy()
            X[idx_replace] = op
            neighbor = Solution(X=X)
            N.append(neighbor)
    return N, H

def is_duplicated(genotypeHash_list1, genotypeHash_list2):
    if len(genotypeHash_list1) != len(genotypeHash_list1):
        return False
    for genotypeHash in genotypeHash_list1:
        if genotypeHash not in genotypeHash_list2:
            return False
    return True

########################################################################################################################
def remove_duplicate(pop):
    F = np.array([idv.F for idv in pop])
    is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-8)))[0]
    return np.array(pop)[is_unique].tolist()
