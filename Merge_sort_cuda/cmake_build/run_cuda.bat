::File chay chuong trinh sau khi build xong cmake
@echo off
setlocal

:: Chuyen den thu muc chua exe da builc
cd .. 
cd /d build/build_CMakeCUDA

::Kiem tra xem file thuc thi co ton tai khong 
if exist main_execpp.exe (
  echo [INFO] Dang chay main_execpp.exe ...
  .\main_execpp.exe
) else (
  echo [ERROR] Khong tim thay main_execpp.exe trong thu muc build/build_CMakeCUDA
)

endlocal
pause