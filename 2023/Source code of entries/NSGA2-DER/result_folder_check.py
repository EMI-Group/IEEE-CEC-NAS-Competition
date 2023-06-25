import re

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


problem_list = [f'C10_MOP{i}' for i in range(1, 10)] + [f'IN1K_MOP{i}' for i in range(1, 10)]

need=[f'run_{i}.json' for i in range(1,51)]
import os
for i in range(len(problem_list)):
    for method in method_list:
        if problem_list[i] in ['C10_MOP1', 'C10_MOP2'] and method in ['NSGA3', 'RVEA', 'UNSGA3']:
            continue
        file_list=os.listdir(os.path.join(f'C:\\Users\\jpc\\Desktop\\2\\EXP-{method}',f'{problem_list[i]}'))
        if (all([(x in file_list) for x in need]))==False:
            print(i,method,problem_list[i],len(file_list))