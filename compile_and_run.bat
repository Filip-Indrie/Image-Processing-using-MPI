@echo off

gcc -c bmp_common.c -I "c:\Program Files (x86)\Microsoft SDKs\MPI\Include"
gcc -c bmp.c -I "c:\Program Files (x86)\Microsoft SDKs\MPI\Include"
gcc -c convolution.c -I "c:\Program Files (x86)\Microsoft SDKs\MPI\Include" -fopenmp
gcc -c image_processing.c -I "c:\Program Files (x86)\Microsoft SDKs\MPI\Include" -fopenmp

gcc -g feature_testing.c -I "c:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L "c:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -o feature_testing.exe bmp_common.o bmp.o convolution.o image_processing.o -lmsmpi -fopenmp
gcc -g experiments.c -I "c:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L "c:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -o experiments.exe bmp_common.o bmp.o convolution.o image_processing.o -lmsmpi -fopenmp


:: running experiments for 2 - 16 processes
for /l %%x in (2, 1, 16) do (
	mpiexec -n %%x experiments Measurements\measurements_%%x.txt
)


GOTO :SHUTDOWN

:: feature_testing serial Large.bmp
mpiexec -n 5 feature_testing serial Photos\Large.bmp Photos\Serial_Large.bmp gaussblur5

:: feature_testing parallel SFT Large.bmp
mpiexec -n 5 feature_testing parallel Photos\Large.bmp Photos\Parallel_sft_Large.bmp gaussblur5 1

:: feature_testing parallel NO SFT Large.bmp
mpiexec -n 5 feature_testing parallel Photos\Large.bmp Photos\Parallel_no_sft_Large.bmp gaussblur5 0

:: feature_testing master Large.bmp
mpiexec -n 5 feature_testing master Photos\Large.bmp Photos\Master_Large.bmp gaussblur5

:: feature_testing serial XL.bmp
mpiexec -n 5 feature_testing serial Photos\XL.bmp Photos\Serial_XL.bmp gaussblur5

:: feature_testing parallel SFT XL.bmp
mpiexec -n 5 feature_testing parallel Photos\XL.bmp Photos\Parallel_sft_XL.bmp gaussblur5 1

:: feature_testing parallel NO SFT XL.bmp
mpiexec -n 5 feature_testing parallel Photos\XL.bmp Photos\Parallel_no_sft_XL.bmp gaussblur5 0

:: feature_testing master XL.bmp
mpiexec -n 5 feature_testing master Photos\XL.bmp Photos\Master_XL.bmp gaussblur5

:: feature_testing serial XXL.bmp
mpiexec -n 5 feature_testing serial Photos\XXL.bmp Photos\Serial_XXL.bmp gaussblur5

:: feature_testing parallel SFT XXL.bmp
mpiexec -n 5 feature_testing parallel Photos\XXL.bmp Photos\Parallel_sft_XXL.bmp gaussblur5 1

:: feature_testing parallel NO SFT XXL.bmp
mpiexec -n 5 feature_testing parallel Photos\XXL.bmp Photos\Parallel_no_sft_XXL.bmp gaussblur5 0

:: feature_testing master XXL.bmp
mpiexec -n 5 feature_testing master Photos\XXL.bmp Photos\Master_XXL.bmp gaussblur5

:SHUTDOWN 

shutdown /s