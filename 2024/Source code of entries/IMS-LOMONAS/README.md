# IMS-LOMONAS: Parameter-less Pareto Local Search for Multi-objective Neural Architecture Search with the Interleaved Multi-start Scheme
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Quan Minh Phan, Ngoc Hoang Luong

## Setup
- Clone this repo
- Create virtual environment
```
$ conda create -n cec_2024 python=3.9.0
$ conda activate cec_2024
```
- Run `install.sh` to install requisite files
```
$ bash install.sh
```
## Reproducing the results
```shell
$ python run_ims_lomonas.py --pid <problem_id>[from 1 to 16]
                            --base <counter_base> 8
                            --alpha <angle_for_selecting_knee_solutions> 210
```

## Run other algorithms
### LOMONAS
```shell
$ python run_lomonas.py --pid <problem_id>[from 1 to 16]
                        --base <counter_base> 8
                        --k <number_of_selected_fronts> 3
                        --alpha <angle_for_selecting_knee_solutions> 210
```
### Baselines (NSGA2, NSGA3, MOEA/D, RVEA, SMS, SPEA2)
```shell
$ python run_baseline.py --pid <problem_id>[from 1 to 16]
                         --moea nsga2 [nsga2, nsga3, moead, rvea, sms, spea2]
```