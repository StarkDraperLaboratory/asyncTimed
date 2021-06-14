# Comparing Async-Timed with Async-FMB algorithm on TensorFlow

This repository contains the codebase necessary to recreate simulation results presented in [JSAIT'21](https://www.itsoc.org/jsait) publication ["Asynchronous delayed optimization with time-varying minibatches" H. Al-Lawati, T. Adikari, S. C. Draper](https://ieeexplore.ieee.org/document/9429693). 

The repository includes the implementations of two algorithms Async-Timed and Async-FMB. The codebase uses [MPI](https://mpi4py.readthedocs.io/en) for process management (e.g. spawn master and workers in master-worker system) and inter-process communication (e.g. communication between master and workers).


## Implementaion details

### Package requirements

* The code is tested and works on two systems that have following software versions.
```
python: 3.7.0
numpy: 1.16.3
mpi4py: 3.0.0
tensorflow: 1.13.1
```

### Tutorial/sample code for using Async-Timed implementation
* One part of the Async-Timed algorithm is the time-limited computation of a minibatch at workers. This is achieved with the `tf.while_loop` function in TensorFlow.
* The input minibatch is partitioned into `amb_num_partitions` 'micro' batches, each of size `batch_size/amb_num_partitions`. The gradients of partitions are then calculated in a loop, starting from the first while the elapsed time>`amb_time_limit`. When the condition fails the worker sends the gradients (summed across the processed partitions) to master.
* [`run_sample_amb.py`](src/run_sample_amb.py) includes a sample implementation of how `tf.while_loop` is used for time-limited computation within one worker. See the comments in the script and execute the script with command `python src/run_sample_amb.py`.
* [`run_sample_code.py`](src/run_sample_code.py) includes a sample implementation of the same that can be executated with multiple workers. See the comments therewithin. Run the code with `mpirun -n 3 python -u src/run_sample_code.py` (master and two workers). Toggle `is_distributed` boolean to run in distributed or non-distributed manner.



### Recreating ImageNet experiments in [JSAIT'21](https://ieeexplore.ieee.org/document/9429693)

#### Generating data for Fig. 3

Run following commands to generate data

* Async-FMB: `mpirun -n 4 python -u src/run_perf_amb.py imagenet32 fmb sgd 1024 --test_size 128 --last_step 400000 --learning_rate 0.1 --dist_sgy async --async_master batch --async_master_batch_min 3 --cuda gpu_all --gpus_per_node 4`

* Async-Timed: `mpirun -n 4 python -u src/run_perf_amb.py imagenet32 amb sgd 1024 --test_size 128 --last_step 400000 --learning_rate 0.1 --dist_sgy async --async_master time --async_master_time 0.2 --cuda gpu_all --gpus_per_node 4 --amb_time_limit 0.290 --amb_num_partitions 2`

* Simulation output written to `~/scratch/distributed` by default.
* See the output of `python src/run_perf_amb.py --help` for details of arguments.

#### Generating data for Fig. 4

Run following commands to generate data

* Async-FMB: `mpirun -n 4 python -u src/run_perf_amb.py imagenet32 fmb sgd 1024 --test_size 128 --last_step 400000 --learning_rate 0.1 --dist_sgy async --async_master batch --async_master_batch_min 2 --async_delay_std 0.05 --cuda gpu_all --gpus_per_node 4 --induce`

* Async-Timed: `mpirun -n 4 python -u src/run_perf_amb.py imagenet32 amb sgd 1024 --test_size 128 --last_step 400000 --learning_rate 0.1 --dist_sgy async --async_master time --async_master_time 0.2 --cuda gpu_all --gpus_per_node 4 --induce --async_delay_std 0.05 --amb_time_limit 0.320 --amb_num_partitions 8`

#### Collecting data

* Simulation output for all commands is written to `~/scratch/distributed`.
* See the output of `python src/run_perf_amb.py --help` for details of arguments.
