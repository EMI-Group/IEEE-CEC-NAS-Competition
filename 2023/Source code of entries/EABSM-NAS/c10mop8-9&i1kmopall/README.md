# run the c10mop8-9 and i1kmopall experiments
We give three examples of c10mop8, i1kmop1, and i1kmop9. The other problems are similar.
```
# train the offline models to predict latency, accuracy, energy and so on
run the predictor.ipynb in each directory
# it will generate the model in corresponding directory

# use the offline models to guide the evolutionary algorithms
run the mopx.py in each directory
# generate more individuals and use offline models to rank and select them

```

# requirements
```
pymoo==0.5.0 # for c10 pid 8,9 and i1k pid 1,2,3,4,5,6
pymoo==0.6.0 # for i1k pid 7,8,9
```

# supplement
- different functions to calculate the hypervolume for different pymoo versions
  - benchmark.calc_perf_indicator(algorithm.pop.get("X"), 'hv') # for pymoo 0.6.0
  - benchmark.calc_perf_indicator050(algorithm.pop.get("X"), 'hv') # for pymoo 0.5.0
  - it is implemented in evobench/modules/benchmark.py