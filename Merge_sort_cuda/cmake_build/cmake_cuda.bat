::File build cmake cho thu vien chua ma nguon CUDA (su dung MSVC + nvcc)
@echo off 
setlocal

set cmake_path="D:/C-C++_project/Project_2024-2/Merge_sort_cuda"
set cuda_compiler= "D:/CUDA/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe"

::Thiet lap moi truong build Visual Studio
call "D:/VisualStudio/Product/VC/Auxiliary/Build/vcvars64.bat"

::Di chuyen vao folder chua file.bat 
cd /d "%~dp0"
if not exist "../build/build_CMakeCUDA" mkdir "../build/build_CMakeCUDA"
cd ../build/build_CMakeCUDA

::Generate project voi CMake, chi ro CUDA compiler la nvcc
cmake %cmake_path% -G "NMake Makefiles" -DCMAKE_CUDA_COMPILER=nvcc ^
    -DENABLE_CUDA=ON ^
    -DCMAKE_C_COMPILER=cl ^
    -DCMAKE_CXX_COMPILER=cl 

if errorlevel 1 (
  echo [ERROR] CMake loi, khong tao duoc project !
  exit /b %errorlevel%
)

::Build
echo [INFO] Dang tien hanh build...
nmake 

if errorlevel 1 (
  echo [ERROR] Build that bai !
  exit /b %errorlevel%
) else (
  echo [SUCCESS] Build hoan tat !
)

endlocal