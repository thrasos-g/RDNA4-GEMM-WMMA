@echo off
setlocal

:: Compile
echo Compiling mmm.cpp...
hipcc -O3 mmm.cpp --offload-arch=native -mcumode -lrocblas -o sgemm.exe

if %errorlevel% neq 0 (
    echo Compilation failed!
    exit /b %errorlevel%
)
echo Compilation successful.

:: Generate Assembly for inspection
echo Generating Assembly...
hipcc -std=c++20 -O3 --offload-arch=gfx1201 -mcumode -lrocblas --offload-device-only -S -g mmm.cpp -o kernel_rdna4.s

echo Running sgemm.exe...
.\sgemm.exe