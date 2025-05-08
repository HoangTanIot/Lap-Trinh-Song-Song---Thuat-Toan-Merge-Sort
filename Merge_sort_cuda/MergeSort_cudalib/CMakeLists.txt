# Tranh loi CMake khi dung build CMake cho C/C++ thuong voi MSYS2/GCC khong ho tro CUDA 
option(ENABLE_CUDA "Enable CUDA build" ON)

if(ENABLE_CUDA)
  enable_language(CUDA)

  # Set kien truc phu hop voi RTX 3050
  set(CMAKE_CUDA_ARCHITECTURES 86)

  set(source_file "merge_sort_cudalib.cu" "Cuda_check_err.cu")

  add_library(MergeSort_cuda_lib STATIC ${source_file})

  include_directories(../NecessaryLib)

  target_link_directories(MergeSort_cuda_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                                             PRIVATE neces_lib)

  set_target_properties(MergeSort_cuda_lib PROPERTIES 
                      CUDA_SEPARABLE_COMPILATION ON
                      POSITION_INDEPENDENT_CODE ON
                      LINKER_LANGUAGE CUDA)
                      
endif()
