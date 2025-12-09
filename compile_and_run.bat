@echo off

gcc -c bmp_common.c
gcc -c bmp_serial.c
gcc -c bmp_parallel.c -I "c:\Program Files (x86)\Microsoft SDKs\MPI\Include"
gcc -c convolution.c

gcc -g main.c -I "c:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L "c:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -o main.exe bmp_common.o bmp_serial.o bmp_parallel.o convolution.o -lmsmpi

:: mpiexec -n 5 main parallel Photos\Large.bmp Photos\Edited_Large.bmp ridge 1