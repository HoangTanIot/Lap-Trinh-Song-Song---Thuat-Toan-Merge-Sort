::File build cmake cho thu vien chua ma nguon C/C++ thuong
@echo off
setlocal

::Thiet lap trinh bien dich su dung MSYS2 ucrt64 (khong thi se mac dinh la duong dan cua MSVC)
set CC="C:/msys64/ucrt64/bin/gcc.exe"
set CXX="C:/msys64/ucrt64/bin/g++.exe"

::Set duong dan toi CMakeLists tong 
set cmake_path="D:\C-C++_project\Project_2024-2\Merge_sort_cuda"

::Di chuyen vao thu muc chua file
cd /d "%~dp0"
if not exist "../build/build_CMakeMSYS2" mkdir "../build/build_CMakeMSYS2"
cd ../build/build_CMakeMSYS2

::Build bang MSYS2 (GCC toolchain)
cmake %cmake_path% -G "MinGW Makefiles" -DCMAKE_C_COMPILER=%CC% ^
 -DCMAKE_CXX_COMPILER=%CXX% ^
 -DENABLE_CUDA=OFF 

if errorlevel 1 (
  echo [ERROR] CMake loi voi MSYS2
  exit /b %errorlevel%
)

echo [INFO] Dang tien hanh build...
mingw32-make 

if errorlevel 1 (
  echo [ERROR] Build voi MSYS2 that bai !
  exit /b %errorlevel%
) else (
  echo [SUCCESS] Build voi MSYS2 thanh cong !
)

endlocal