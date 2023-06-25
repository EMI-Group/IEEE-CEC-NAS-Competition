import numpy as np
from individual import Individual
import ast
import numpy as np
from operator import itemgetter, attrgetter
from mlp_NB101 import Net
import torch
class Result:
    def __init__(self,length):
        self.X = []



def one_hot_encode(input_arr, low_arr, up_arr):
    num_values = [up_arr[i] - low_arr[i]  for i in range(len(up_arr))]
    # print(num_values)
    encoded_arr = np.zeros((input_arr.shape[0],sum(num_values)))
    # print(encoded_arr)
    for i in range(len(encoded_arr)):
        # print("i",i)
        idx = 0
        for j in range(len(input_arr[0])):
            # print(i,idx + input_arr[i][j]-low_arr[j])
            encoded_arr[i][idx + input_arr[i][j]-low_arr[j]] = 1
            idx += up_arr[j]-low_arr[j]
    return encoded_arr


class NSGA_2:
    def __init__(self,pop_size, sampling, crossover,
        mutation, eliminate_duplicates):
        self.pop_size = pop_size
        self.pm = 0.7
        self.max_iter = 33
        self.eval_num = 0
        self.cnt = 0
        
        self.max_eval_num = 10000
        return
    


    def set_up(self,problem,benchmark):
        with open('data.txt', 'r') as f:
            contents = f.read()
        data_dict = ast.literal_eval(contents)
        self.lower = np.array(data_dict['IN4']['lower'])
        self.upper = np.array(data_dict['IN4']['upper'])

        self.problem = problem
        self.benchmark = benchmark
        self.len_dna = problem.n_var
        self.archive = np.empty(0,dtype=object)
        self.dna_archive = np.array((0,self.len_dna))
        self.n_objs = problem.n_obj
        print(self.problem)

    def init_pop(self):
        P = np.empty(self.pop_size*8,dtype=object)
        idx = 0
        alpha = 0
        while idx < self.pop_size*8:
            
            individual = Individual(True,self.lower,self.upper,self.n_objs,None)
            f = self.benchmark.normalize(self.benchmark.evaluate(np.reshape((individual.dna),(1,-1))))
            # d = np.reshape(individual.dna,(1,-1))
            # print(d)
            # f = self.benchmark.evaluate(d)
            # f = (self.benchmark.evaluate(np.reshape((individual.dna),(1,-1))))
            individual.f = f[0]
            
            self.dna_archive = np.concatenate((self.dna_archive,individual.dna))
            individual.parameters = individual.f[1]
            individual.accuracy = individual.f[0]
            individual.f1 = (1-alpha)*individual.f+alpha *np.sum(individual.f)/len(individual.f)
            P[idx] = individual   
            idx += 1 


        F = self.fast_non_dominated_sort(P)
        self.crowding_distance_pop(F)
        P = self.select(P,int(self.pop_size ),0)
        return P
    def cross_over(self,p1,p2):
        p1_dna = p1.dna.copy()
        p2_dna = p2.dna.copy()
        r = np.random.random()
        if r <0.8:
            idx = np.random.randint(1,len(p1_dna)-1)

            temp = p1_dna[idx:].copy()
            p1_dna[idx:] = p2_dna[idx:]
            p2_dna[idx:] = temp
        else:
            idx0 = np.random.randint(1,len(p1_dna)* 0.6)
            idx1 = idx0 + np.random.randint(1,len(p1_dna-idx0)-1)
            temp = p1_dna.copy()
            p1_dna[idx0:idx1] = p2_dna[idx0:idx1]
            p2_dna[idx0:idx1] = temp[idx0:idx1]
        p1_new = Individual(False, self.lower,self.upper,p1_dna,self.n_objs)
        p2_new = Individual(False, self.lower,self.upper,p2_dna,self.n_objs)
        return p1_new, p2_new
    def pop_cross(self,P,num):
        new_Q = []
        i = 0
        while i < num-1:
            p1 = P[(int)(np.random.rand()*len(P))]
            p2 = P[(int)(np.random.rand()*len(P))]
            q1,q2 = self.cross_over(p1,p2)
            new_Q.append(q1)
            new_Q.append(q2)
            i += 2
        for q in new_Q:
            
            if np.random.rand() < self.pm:
                while True:
                    q.mutation(self.lower,self.upper)
                    if q.dna not in self.dna_archive:
                        self.dna_archive = np.concatenate((self.dna_archive,q.dna))
                        break
        return new_Q
    
    def pop_mutation(self,Q):
        for q in Q:
            
            if np.random.rand() < self.pm:
                q.mutation(self.lower,self.upper)

        return Q
    
    def get_unique_pop(self, P):
        record = []
        new_P = []

        for indiv in P:
            if not any((indiv.f == r).all() for r in record):
                record.append(indiv.f)
                new_P.append(indiv)

        return new_P

    def select(self,R,num ,stage = 0):
        R = self.get_unique_pop(R)
        if stage == 0:
            R = sorted(R)
            return R[0:num]
    
        if stage == 1:                  
            R = sorted(R,key=attrgetter('parameters'))
            R1 = R[0:int(2/3*len(R))]
            R1 = sorted(R1)
            res1 = R1[:min((int(num/2),len(R1)))]
            R2 = R[int(2/3*len(R)):]
            R2 = sorted(R2)
            res2 = R2[:int(num/2)]
            return res1 + res2
        if stage == 2:
            R = sorted(R,key=attrgetter('parameters'))
            R1 = R[0:int(1/3*len(R))]
            R1 = sorted(R1)
            res1 = R1[:min((int(num/2),len(R1)))]
            R2 = R[int(1/3*len(R)):]
            R2 = sorted(R2)
            res2 = R2[:int(num/2)]
            return res1+res2
    
    def calculate_f(self,population,n_times,alpha,stage):
        if alpha <0:
            alpha = 0
        for idx in range(len(population)):

            dna = np.reshape(population[idx].dna,(1,-1))
            # print(dna)
            f_sum = 0
            for i in range(n_times):
                f_sum += self.benchmark.normalize(self.benchmark.evaluate(dna)[0])
                # f_sum += (self.benchmark.evaluate(dna)[0])
            # print(idx,f)
            population[idx].f = f_sum/n_times
            population[idx].parameters = population[idx].f[1]
            population[idx].accuracy = population[idx].f[0]
            population[idx].f1 = population[idx].f
            if stage == 0:
                population[idx].f1 = (1-alpha)*population[idx].f+alpha *np.sum(population[idx].f)/len(population[idx].f)
            if stage == 1:
                # population[idx].f1[1] =  0.8*population[idx].f[0] + 0.2*population[idx].f[1]
                population[idx].f1 = (1-alpha)*population[idx].f + alpha*population[idx].f[0]

            if stage == 2:
                population[idx].f1 = (1-alpha)*population[idx].f + alpha*population[idx].f[0]
 
            self.eval_num += n_times
    def fast_non_dominated_sort(self,population):
        for indiv in population:
            indiv.rank = -1
            self.crowding_dist = -1
        n = np.zeros(len(population))  
        S = []  
        F = [[]]  
        for i in range(len(population)):
            Sp = []
            n[i] = 0
            for j in range(len(population)):
                if i == j:
                    continue
                if population[i].dominate(population[j]):
                    Sp.append(j)
                elif population[j].dominate(population[i]):
                    n[i] += 1
            S.append(Sp)
            if  n[i] == 0:
                population[i].rank = 0
                F[0].append(i)

        i = 0
        while len(F[i]) != 0:
            Q = []  
            for p in F[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        population[q].rank = i + 1
                        Q.append(q)
            i += 1
            F.append(Q)
        F.pop()


        F_obj = []
        for i in range(len(F)):
            l = []

            for j in range(len(F[i])):
                l.append(population[F[i][j]])
            F_obj.append(l)

        return F_obj  
    
    def crowding_distance_pop(self,F):
        for rank in F:
            self.crowding_distance_assignment(rank)

        return
    
    def crowding_distance_assignment(self,rank):
        
        pop_fitness = np.zeros((len(rank),self.n_objs))

        for idx in range(len(rank)):
            rank[idx].crowding_dist = 0
            # print(rank[idx].f)
            pop_fitness[idx] = rank[idx].f

        sorted_idx = np.argsort(pop_fitness,axis=0)

        for i in range(self.n_objs):
            fi_sort_idx = sorted_idx[:,i]
            fi_sort_p = [rank[i] for i in fi_sort_idx]
            inf = (1e15)
            fi_sort_p[0].crowding_dist = inf
            fi_sort_p[-1].crowding_dist = inf
            for j in range(1,len(fi_sort_p)-1):
                fi_sort_p[j].crowding_dist += (fi_sort_p[j + 1].f[i] - fi_sort_p[j - 1].f[i])/(fi_sort_p[-1].f[i] - fi_sort_p[0].f[i])
            
        return rank


    def greedy_sort_archive(self):
        import numpy as np
        import hvwfg
        ref = np.ones(self.n_objs)
        print("length of archieve" ,len(self.archive))
        F = np.ones(self.n_objs)
        elite = []

        for i in range(self.pop_size):
            print(i,"-------------------------")
            hv_max = 0
            # print(F)

            for indiv in self.archive:

                objs = np.row_stack((F,indiv.f))

                hv_temp = hvwfg.wfg(objs,ref)
                if hv_temp > hv_max:
                    indivi_max = indiv
                    hv_max = hv_temp
            indivi_max.print_info()
            print(hv_max)
            F = np.row_stack((F,indivi_max.f))
            elite.append(indivi_max)
        return elite
    def plot_pop(self,elite):
        
        res = Result(len(elite))
        for i in elite:
            res.X.append(i.dna)
        X = np.array(res.X)
        X = X.astype('int')

        F = self.benchmark.evaluate(X,true_eval = True)
        mask = np.isfinite(F).all(axis=1)
        filtered_array = F[mask]
        import matplotlib.pyplot as plt
        x1 = filtered_array[:, 0] # First column
        x2 = filtered_array[:, 1] # Second column
        ps = self.benchmark.pareto_front
        ps = self.benchmark.normalize(ps)
        y1 = ps[:,0]
        y2 = ps[:,1]
        plt.scatter(x1,x2)
        plt.scatter(y1,y2,alpha=0.5)
        plt.show()
    def run(self,max_iteration):
        #stage 1 init
        P = self.init_pop()
        self.archive = np.concatenate((self.archive,P))
        self.eval_num += 8*self.pop_size
        res = np.zeros((len(P),(self.len_dna)))
        print("eval_num after init",self.eval_num)

  
        R = P
        
        for idx in range(len(res)):
            res[idx] = R[idx].dna
        res = res.astype('int')
        hv = self.benchmark.calc_perf_indicator(res, 'hv')

        print(hv)
        print("------------------------------------------------------")
        l = []
        maxnum = 0
        alpha = 0
        P = self.select(P,(int)(len(P)*0.8),0)
        # for indiv in P:
            # indiv.print_info()
        if self.n_objs >= 0:
            alpha = 0.4
        # pop init by self.init_pop did not have f1 inited.
        for indiv in P:
            indiv.f1 = (1-alpha)*indiv.f+alpha *np.sum(indiv.f)/len(indiv.f)
        while True:
            print(self.eval_num)
            if self.eval_num + 2 * self.pop_size > 4500:
                break
            Q = self.pop_cross(P,len(P))
            # Q = self.pop_mutation(Q)
            R = np.concatenate((P,Q))
            self.archive = np.concatenate((self.archive,Q))
            
            for individual in R:
                individual.rank = -1
                individual.crowding_dist = -1

            self.calculate_f(Q,2,alpha,0)

            F = self.fast_non_dominated_sort(R)
            self.crowding_distance_pop(F) 
            P = self.select(R,len(P),0)
            # print(len(P))
            # res = np.zeros((len(P),len(P[0].dna)))
            # for idx in range(len(P)):
                # res[idx] = P[idx].dna

            # res = res.astype('int')
            # print(res)
            # hv = self.benchmark.calc_perf_indicator(res, 'hv')
            # print(hv,"hv of all")
        
        param_record = []
        P = [item for item in P if item.f[0] < 1]
        P = [item for item in P if item.f[1] < 1]
        for indiv in P:
            # indiv.print_info()
            param_record.append(indiv.f[0])     


        # Calculate the mean and standard deviation using NumPy
        param_record = np.array(param_record)
        mean = np.mean(param_record)
        std_dev = np.std(param_record)

        # Calculate the mean deviation
        param_record = sorted(param_record)
  
        
        # res = np.zeros((len(self.archive),len(P[0].dna)))
        # for idx in range(len(self.archive)):
            # res[idx] = self.archive[idx].dna

        # res = res.astype('int')

        # hv = self.benchmark.calc_perf_indicator(res, 'hv')
        # print("hv of archive",hv)


        for indiv in self.archive:
            indiv.accuracy = indiv.f[0]
        archive = sorted(self.archive,key=attrgetter('accuracy'))
        for indiv in archive:
            indiv.f1 = indiv.f

        self.archive = P
        alpha = 0
        p1 = [item for item in archive if item.accuracy < mean-0.6*std_dev]

        p2 = self.select(P,min(len(P),int(self.pop_size*0.4)),0)
        p3 = [item for item in archive if item.accuracy > mean+0.6*std_dev]

        for idx in range(len(p1)):
            p1[idx].f1 = (1-alpha)*p1[idx].f + alpha*p1[idx].f[1]
        for idx in range(len(p3)):
            p3[idx].f1 = (1-alpha)*p3[idx].f + alpha*p3[idx].f[0]
        F = self.fast_non_dominated_sort(p1)
        self.crowding_distance_pop(F)
        p1 = self.select(p1,min(len(p1),int(self.pop_size*0.6)),0)
        
        F = self.fast_non_dominated_sort(p3)
        self.crowding_distance_pop(F)
        p3 = self.select(p3,min(len(p3),int(self.pop_size*0.6)),0)
        self.archive = np.concatenate((self.archive,p1))
        self.archive = np.concatenate((self.archive,p3))   
        # print("p1")
        # self.plot_pop(p1)
        # print("p3")
        # self.plot_pop(p3)    
        elite = p3+p1+p2
        # self.plot_pop(elite)
        # print(ps)
        print("len P",len(P))
        res = np.zeros((len(self.archive),len(P[0].dna)))
        for idx in range(len(self.archive)):
            res[idx] = self.archive[idx].dna

        res = res.astype('int')


        P = p1 + p2
        elite = P
        
        
        if self.n_objs >= 0:
            alpha = 0.4
        # print(len(self.archive))
        # pop init by self.init_pop did not have f1 inited.
        while True:
            print(self.eval_num)
            if self.eval_num + 2 * self.pop_size > 6800:
                break
            Q = self.pop_cross(P,len(P))
            # Q = self.pop_mutation(Q)
            R = np.concatenate((P,Q))
            self.archive = np.concatenate((self.archive,Q))
            
            for individual in R:
                individual.rank = -1
                individual.crowding_dist = -1

            self.calculate_f(Q,2,alpha,1)

            F = self.fast_non_dominated_sort(R)
            self.crowding_distance_pop(F) 
            P = self.select(R,self.pop_size,1)

        res = res.astype('int')


        if self.n_objs >= 0:
            alpha = 0.2
        # print(len(self.archive))
        # pop init by self.init_pop did not have f1 inited.
        # self.plot_pop(self.archive)

        P = p2+p3
        while True:
            print(self.eval_num)
            if self.eval_num + 2 * self.pop_size >= 8000:
                break
            Q = self.pop_cross(P,len(P))
            # Q = self.pop_mutation(Q)
            R = np.concatenate((P,Q))
            self.archive = np.concatenate((self.archive,Q))
            
            for individual in R:
                individual.rank = -1
                individual.crowding_dist = -1

            self.calculate_f(Q,2,alpha,0)

            F = self.fast_non_dominated_sort(R)
            self.crowding_distance_pop(F) 
            P = self.select(R,self.pop_size,0)

        p1 = P
        # self.plot_pop(self.archive)
        





        res = res.astype('int')
        if self.n_objs >= 0:
            alpha = 0.4
        # pop init by self.init_pop did not have f1 inited.
        while True:
            print(self.eval_num)
            if self.eval_num + 2 * self.pop_size > 9000:
                break
            Q = self.pop_cross(P,len(P))
            # Q = self.pop_mutation(Q)
            R = np.concatenate((P,Q))
            self.archive = np.concatenate((self.archive,Q))
            
            for individual in R:
                individual.rank = -1
                individual.crowding_dist = -1

            self.calculate_f(Q,2,alpha,2)

            F = self.fast_non_dominated_sort(R)
            self.crowding_distance_pop(F) 
            P = self.select(R,self.pop_size,2)

        # self.plot_pop(self.archive)
        res = res.astype('int')
        if self.n_objs >= 0:
            alpha = 0.2
        # pop init by self.init_pop did not have f1 inited.
        while True:
            print(self.eval_num)
            if self.eval_num + 2 * self.pop_size >= 10000:
                break
            Q = self.pop_cross(P,len(P))
            # Q = self.pop_mutation(Q)
            R = np.concatenate((P,Q))
            self.archive = np.concatenate((self.archive,Q))
            
            for individual in R:
                individual.rank = -1
                individual.crowding_dist = -1

            self.calculate_f(Q,2,alpha,2)

            F = self.fast_non_dominated_sort(R)
            self.crowding_distance_pop(F) 
            P = self.select(R,self.pop_size,0)
            res = np.zeros((len(P),len(P[0].dna)))
            for idx in range(len(P)):
                res[idx] = P[idx].dna

            res = res.astype('int')
            # print(res)
            hv = self.benchmark.calc_perf_indicator(res, 'hv')
        
        # self.plot_pop(self.archive)
        # p3 = P

        for indiv in self.archive:
            indiv.accuracy = indiv.f[0]
        archive = sorted(self.archive,key=attrgetter('accuracy'))
        for indiv in archive:
            indiv.f1 = indiv.f

        self.archive = P
        alpha = 0
        p1 = [item for item in archive if item.accuracy < mean-0.8*std_dev]
        # p2 = [item for item in archive if (item.accuracy > mean-0.6*std_dev and item.accuracy < mean+0.6*std_dev)]
        p3 = [item for item in archive if item.accuracy > mean+0.8*std_dev]
        print(len(p1))
        print(len(p2))
        print(len(p3))
        alpha = 0

        for idx in range(len(p1)):
            p1[idx].f1 = (1-alpha)*p1[idx].f + alpha*p1[idx].f[1]
        for idx in range(len(p3)):
            p3[idx].f1 = (1-alpha)*p3[idx].f + alpha*p3[idx].f[0]
        F = self.fast_non_dominated_sort(p1)
        self.crowding_distance_pop(F)
        # self.plot_pop(p1)
        p1 = self.select(p1,min(len(p1),int(self.pop_size*0.4)),0)
        # print("selected p1")
        # self.plot_pop(p1)
        p2 = self.select(p2,min(len(p2),int(self.pop_size*0.3)),0)
        # print("selected p2")
        # self.plot_pop(p2)
        F = self.fast_non_dominated_sort(p3)
        self.crowding_distance_pop(F)
        p3 = self.select(p3,min(len(p3),int(self.pop_size*0.3)),0)
        # print("selected p3")
        # self.plot_pop(p3)
        P = self.select(p1+p2+p3,self.pop_size,0)
        # self.plot_pop(P)  
        
        res = np.zeros((len(P),len(P[0].dna)))
        for idx in range(len(P)):
            res[idx] = P[idx].dna

        res = res.astype('int')
        hv = self.benchmark.calc_perf_indicator(res, 'hv')


        print(hv,"hv of final solution")

        return res
    



