# Parallel Image Editing (MPI + OpenMP)

## Overview

This project implements a parallel image editor capable of applying convolutions on 24-bit BMP images. The editor supports multiple convolution kernels and is designed to run efficiently on a cluster of workstations.
Both serial and multiple parallel versions are provided. The parallel versions are implemented to support environments with and without a shared file tree (SFT), including a Producer/Worker model.

## Supported Operations

The following operations are all implemented based on convolution:
- RIDGE
- EDGE
- SHARPEN
- BOXBLUR
- GAUSSBLUR3
- GAUSSBLUR5
- UNSHARP5

## Program Interface

> [!IMPORTANT]
> ### To compile and run any of these programs, you need to install gcc, MSMPI TOOLS and MSMPI SDK. MSMPI TOOLS and MSMPI SDK can be found [here](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi).

The provided `compile_and_run.bat` compiles all the necessary files into 2 executables: `feature_testing.exe` and `experiments.exe`.

<br/>

- `feature_testing.exe` --> this executable is used to apply a convolution to a BMP image using `N` processes and the specified version.

To run `feature_testing.exe`, use the following command:
```
mpiexec -n N feature_testing.exe VERSION FILE_PATH_IN FILE_PATH_OUT OPERATION SFT
```
N = number of processes
VERSION = {`serial`, `parallel`, `master`}  
FILE_PATH_IN = path at which the image resides (can be relative or absolute)  
FILE_PATH_OUT = path at which the edited image is to be saved (can be relative or absolute)  
OPERATION = one of the supported operations stated above  
SFT = {`1`, `0`}, needed only for the `parallel` versions, tells the program if the system has a SFT or not

<br/>

- `experiments.exe` --> this executable is used to apply a convolution to BMP images using `N` processes and all the implemented versions. The BMP images paths to which to apply a convolution need to be edited inside `experiments.c`.

To run `experiments.exe`, use the following command:
```
mpiexec -n N experiments.exe FILE_PATH_OUT
```
N = number of processes  
FILE_PATH_OUT = path at which to save the measurements

## Implementation Details

### Serial Version

The serial version uses one single MPI process and one single OpenMP thread to process the image. In the following versions, the serial version is used as the ground truth when comparing the results.

### Parallel with SFT Version

The parallel version made for machines with a SFT uses `N` MPI processes, each of these processes running on workstations having at least `C` cores. All the processes have access to the same filesystem, including the image to be edited. Each process reads only its assigned image region. The edited region is then centralized by process 0, assembled and saved.

### Parallel with no SFT Version

The parallel version made for machines without a SFT uses `N` MPI processes, each of these processes running on workstations having at least `C` cores. Only process 0 reads the image. It then distributes approximately equal chunks to all the processes. After processing their respective chunk, all the processes send their edited chunk back to process 0 for it to assemble and save the whole edited image.

### Producer/Worker Version

The master/worker version uses `N` MPI processes, each of these processes running on workstations having at least `C` cores. Only process 0 reads the image. It reads small chunks of the image, which it immediately sends to a free worker process. The worker process then edits their chunks and sends the edited chunk back to process 0 which places it in its right spot. It also signals process 0 that it is ready to receive more work. When all work is done, process 0 signals the termination of all the worker processes and then saves the whole edited image.

> [!NOTE]
> After execution, the output of all versions is verified against the ground truth (the serial version).
> Only executions that produce identical results are included in performance measurements.

### Threads

All the parallel versions are splitting the convolution task across multiple threads. The number of threads allocated for each process is:
```
NUM_CORES / (NUM_PROCESSES / NUM_WORKSTATIONS)
```
NUM_CORES = number of cores on each workstation  
NUM_PROCESSES = number of processes started  
NUM_WORKSTATIONS = number of workstations on which the processes are running

Example:
On a 16-core workstation
- 8 MPI Processes --> 2 threads per process
- 9 MPI Processes --> 1 thread per process

This strategy avoids oversubscription and explains observed performance drops when increasing the number of MPI processes.

## Experiment

This repository also includes the results of an experiment run on 1 workstation with 16 cores. The experiment tracked the time it took to perform `GAUSSBLUR5` on 2 - 16 processes.  
It also measures the speedup between each version and the serial one, and the speedup between Master/Worker and Parallel with no SFT.
In `centralized_measurements.xlsx`, each measured time of the Master/Worker version is the best out of all the times measured with different chunk sizes. It also includes the chunk size at which it achieved this time.
