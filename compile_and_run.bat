@echo off

gcc -c bmp.c -I "c:\Program Files (x86)\Microsoft SDKs\MPI\Include"
gcc -c convolution.c
gcc -c image_processing.c -I "c:\Program Files (x86)\Microsoft SDKs\MPI\Include"

gcc -g main.c -I "c:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L "c:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -o main.exe bmp.o convolution.o image_processing.o -lmsmpi

:: serial
mpiexec -n 5 main serial Photos\Large.bmp Photos\Serial_Large.bmp ridge

:: parallel SFT
mpiexec -n 5 main parallel Photos\Large.bmp Photos\Parallel_sft_Large.bmp ridge 1

:: parallel NO SFT
mpiexec -n 5 main parallel Photos\Large.bmp Photos\Parallel_no_sft_Large.bmp ridge 0