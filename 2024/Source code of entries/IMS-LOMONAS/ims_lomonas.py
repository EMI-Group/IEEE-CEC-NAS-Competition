"""
Source code for Interleaved Multi-start Scheme LOMONAS (IMS-LOMONAS)
Authors: Quan Minh Phan, Ngoc Hoang Luong
doi: 10.1016/J.SWEVO.2024.101573
"""
import random
import numpy as np
from typing import Tuple
from lomonas import LOMONAS
from utilities import Result, Footprint
from core import ElitistArchive
import math

class IMS_LOMONAS:
    def __init__(self,
                 base=2,
                 check_limited_neighbors=False, neighborhood_check_on_potential_sols=False, alpha=210,
                 perform_termination=True,
                 evaluator=None,
                 debugger=None):
        self.name = f'IMS-LOMONAS-base{base}'
        self.check_limited_neighbors = check_limited_neighbors
        self.neighborhood_check_on_potential_sols = neighborhood_check_on_potential_sols
        self.alpha = alpha
        self.perform_termination = perform_termination
        self.base = base

        self.evaluator = evaluator
        self.debugger = debugger

    @property
    def hyperparameters(self):
        return {
            'optimizer': self.name,
            'base': self.base,
            'check_limited_neighbors': self.check_limited_neighbors,
            'neighborhood_check_on_potential_sols': self.neighborhood_check_on_potential_sols,
            'alpha': self.alpha,
            'perform_termination': self.perform_termination
        }

    def set(self, key_value):
        for key, value in key_value.items():
            if key in self.__dict__:
                self.__dict__[key] = value

    ######################################################## Main ######################################################
    def solve(self, problem, seed, max_eval, **kwargs):
        random.seed(seed)
        np.random.seed(seed)
        self._solve(problem, max_eval, **kwargs)

        genotype_archive = [elitist.X for elitist in self.archive.archive]
        res = Result(genotype_archive)
        return res

    def _solve(self, problem, max_eval, **kwargs):
        LOP = []  # list of processors
        self.archive = ElitistArchive()
        self.footprint = Footprint()

        self.problem = problem
        self.max_eval = max_eval

        self.visited = []
        self.res_logged = []

        hat = 1
        n_iter, id_candidate = 0, 1
        nF = 1

        new_processor = self.initialize_new_processor(name=f'LOMONAS #{id_candidate - 1}', nF=nF)  # Initialize a new processor
        new_processor.initialize()
        LOP.append(new_processor)  # Add the new processor to the last position of LOP
        isKilled = [False]

        while True:
            n_iter += 1
            if n_iter == self.base ** hat or len(LOP) == 0:
                nF += 1
                id_candidate += 1
                new_processor = self.initialize_new_processor(name=f'LOMONAS #{id_candidate - 1}', nF=nF)  # Initialize a new processor
                new_processor.initialize()

                LOP.append(new_processor)  # Add the new processor to the last position of LOP
                isKilled.append(False)

                hat += 1

            for i, processor in enumerate(LOP):
                if n_iter % (self.base ** i) == 0:
                    if i == 0:
                        isContinued = processor.neighborhood_checking()
                        if not isContinued:
                            processor.initialize(False)
                    else:
                        while processor.n_eval < LOP[i-1].n_eval/math.log2(self.base):
                            isContinued = processor.neighborhood_checking()
                            if not isContinued:
                                processor.initialize(False)
            if self.isTerminated():
                print('#Evals:', self.evaluator.n_eval)
                return

            if self.perform_termination:
                isKilled, LOP = self.selection(isKilled, LOP)

                # Reset the counter
                isKilled = np.array(isKilled)
                LOP = np.array(LOP)[~isKilled].tolist()
                isKilled = isKilled[~isKilled].tolist()

    ##################################################### Utilities ####################################################
    def isTerminated(self):
        return self.evaluator.n_eval >= self.max_eval

    def initialize_new_processor(self, **kwargs):
        new_processor = LOMONAS(
            name=kwargs['name'],
            nF=kwargs['nF'], neighborhood_check_on_potential_sols=self.neighborhood_check_on_potential_sols,
            check_limited_neighbors=self.check_limited_neighbors, alpha=self.alpha,
            debugger=self.debugger,
            evaluator=self.evaluator,
            res_logged=self.res_logged,
            archive=self.archive,
            footprint=self.footprint
        )
        new_processor.setup(self.problem, self.max_eval)
        return new_processor

    @staticmethod
    def selection(isKilled, LOP) -> Tuple:
        # Kill processors that are dominated by one of later processors (using archive)
        for i in range(len(LOP)):
            for j in range(i + 1, len(LOP)):
                if LOP[i].local_archive.isDominated(LOP[j].local_archive):
                    isKilled[i] = True
                    break
        return isKilled, LOP