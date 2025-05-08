# Khong dung de build file.cu vi GCC MSYS2 khong ho tro CUDA
set(source_file "pseudo_num_gen.c")

add_library(pseudo_num_lib STATIC ${source_file})

find_package(OpenMP REQUIRED)

#Lien ket thu vien omp vao chuong trinh pseudo_num
target_link_libraries(pseudo_num_lib INTERFACE OpenMP::OpenMP_CXX)

#Lenh chi duong dan thu muc chua cac file header (phai cung ten voi add_library)
target_include_directories(pseudo_num_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(pseudo_num_lib PROPERTIES LINKER_LANGUAGE C)