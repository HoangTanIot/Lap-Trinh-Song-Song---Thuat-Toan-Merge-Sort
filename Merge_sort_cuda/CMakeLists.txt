cmake_minimum_required(VERSION 3.30)
project(Merge_sort_cuda LANGUAGES CXX C)

# Chi enable CUDA neu bien nay duoc bat tu dong lenh
if(ENABLE_CUDA)
      enable_language(CUDA)
      set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

# Them thu muc con (thu vien can dung)
add_subdirectory(pseudo_num)

# Neu CUDA duoc bat thi moi build main_execpp
if(ENABLE_CUDA)
      add_executable(main_execpp "main/main.cu")

      add_subdirectory(NecessaryLib)
      add_subdirectory(MergeSort_cudalib) #Chi build ma CUDA neu ENABLE_CUDA duoc bat

      target_include_directories(main_execpp PRIVATE 
                        ${CMAKE_CURRENT_SOURCE_DIR}/NecessaryLib
                        ${CMAKE_CURRENT_SOURCE_DIR}/MergeSort_cudalib)

      # Lien ket cac thu vien con vao executable (Phai trung ten voi ten duoc khai bao trong add_library)
      target_link_libraries(main_execpp PRIVATE
            pseudo_num_lib
            MergeSort_cuda_lib
            neces_lib
            OpenMP::OpenMP_C)

      set_target_properties(main_execpp PROPERTIES 
            CUDA_SEPARABLE_COMPILATION ON
            CXX_STANDARD 17)
endif()
