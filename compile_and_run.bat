@echo off

gcc -c bmp.c -I "c:\Program Files (x86)\Microsoft SDKs\MPI\Include"
gcc -c convolution.c
gcc -c image_processing.c -I "c:\Program Files (x86)\Microsoft SDKs\MPI\Include" -fopenmp

:: compile main
gcc -g main.c -I "c:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L "c:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -o main.exe bmp.o convolution.o image_processing.o -lmsmpi -fopenmp

:: compile testbench
:: gcc -g testbench.c -I "c:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L "c:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -o testbench.exe bmp.o convolution.o image_processing.o -lmsmpi -fopenmp

:: run serial
:: mpiexec -n 5 main serial Photos\Large.bmp Photos\Serial_Large.bmp gaussblur5

:: run parallel SFT
:: mpiexec -n 5 main parallel Photos\Large.bmp Photos\Parallel_sft_Large.bmp gaussblur5 1

:: run parallel NO SFT
:: mpiexec -n 5 main parallel Photos\Large.bmp Photos\Parallel_no_sft_Large.bmp gaussblur5 0

:: run master
:: mpiexec -n 5 main master Photos\Large.bmp Photos\Master_Large.bmp gaussblur5