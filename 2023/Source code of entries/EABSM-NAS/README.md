# evoxbench_submit

Considering the size of the search space and the number of search objectives, we use an online or offline surrogate model strategy

## Offline surrogate model (for c10mop1)

about 420000 archs.

1. We first use an evolutionary algorithm to evolve and terminate using an early stop strategy.

2. Then we randomly sample architectures from the search space (or jointly evolve the searched architectures) to train the offline surrogate model.

3. Finally, we traverse the entire search space. The other architectures are evaluated using the surrogate model, and a simple sorting strategy is used to rank and get topk architectures for further evaluation.

## Online surrogate model (for c10mop2, c10mop8-9&i1kmopall)

More search space and more Pareto frontiers.

1. We randomly sample sone architectures to train the surrogate model to predict the accuracy, latency and so on.

2. During the evolutionary process, more individuals are generated using the evolutionary algorithm (preset to triple). The trained surrogate model is used to evaluate the architecture and the topk individuals are taken using the non-dominated sorting strategy.

## Baseline (for c10mop3-7)
We have tried a novel evolutionary strategy based on an online incremental agent model. It is time consuming and the results are insignificant on the NAS problems. Baseline is used with fewer modifications and later the algorithm is optimised and we will release and update it.

## Extra tricks

- Duplicates: Duplicates are determined by Hash or encoding X so that the evolutionary algorithm will not search the same individuals. (Extremely effective for small search spaces)

- Production: For small search spaces, the evolutionary algorithm converges faster in the evolutionary process. At this point it is difficult to generate new individuals, and we modify the default processing of Pymoo so that we tend to generate more individuals when a small number of individuals need to be generated, delaying the convergence time. Details can be found in c10mop1/README.md. 

- Surrogate Models: We evaluated the model using the five-fold cross-validation and using bagging scheme to ensemble the five models. Sampling are not effective in large search spaces and it is worth exploring further how to train generative surrogate models.


## Results
| c10 mop | performance (avg&std)                    |
| ------- | ---------------------------------------- |
| 1       | HV: 0.9438 (0.0023) IGD: 0.0206 (0.0027) |
| 2       | HV: 0.9235 (0.0012) IGD: 0.0114 (0.0075) |
| 3       | HV: 0.8235 (0.0020) IGD: 0.0275 (0.0030) |
| 4       | HV: 0.7740 (0.0032) IGD: 0.0589 (0.0019) |
| 5       | HV: 0.7126 (0.0000) IGD: 0.0034 (0.0021) |
| 6       | HV: 0.7370 (0.0026) IGD: 0.0325 (0.0101) |
| 7       | HV: 0.5713 (0.0121) IGD: 0.0425 (0.0034) |
| 8       | HV: 0.9756 (0.0033)                      |
| 9       | HV: 0.9607 (0.0046)                      |

| i1k mop |  performance (avg&std)                   |
| ------- | ---------------------------------------- |
| 1       | HV: 0.9326 (0.0030)                      |
| 2       | HV: 0.8867 (0.0007)                      |
| 3       | HV: 0.8050 (0.0031)                      |
| 4       | HV: 0.9947 (0.0035)                      |
| 5       | HV: 0.9977 (0.0045)                      |
| 6       | HV: 0.9867 (0.0031)                      |
| 7       | HV: 0.9198 (0.0018)                      |
| 8       | HV: 0.6914 (0.0054)                      |

# Participants
- Chixin Wang, South China University of Technology, 202121047134@mail.scut.edu.cn
- Zhe Wen, Guangzhou University, 978810732@qq.com
- Jiajun Chen,South China University of Technology, 202221046148@mail.scut.edu.cn
- Zhen Cui，Guangzhou university, z.tsui.work@gmail.com
- Boyi Xiao, Guangzhou University, 1047336005@qq.com
- Advisor Weiqin Ying, South China University of Technology, yingweiqin@scut.edu.cn
- Yu Wu, Guangzhou University, wuyu@gzhu.edu.cn