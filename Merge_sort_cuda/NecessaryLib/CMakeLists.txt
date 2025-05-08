# Tranh loi CMake khi dung build CMake cho C/C++ thuong voi MSYS2/GCC khong ho tro CUDA 
option(ENABLE_CUDA "Enable CUDA build" ON)

if(ENABLE_CUDA)
    enable_language(CUDA)
    
    set(CMAKE_CUDA_ARCHITECTURES 86)

    set(source_file "neces_lib.cu")

    add_library(neces_lib STATIC ${source_file})

    # Bao gom thu muc hien tai de tim thay file.cuh
    target_link_directories(neces_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

    # Bat compilation cho kien truc CUDA can thieu (neu can)
    set_target_properties(neces_lib PROPERTIES 
                          CUDA_SEPARABLE_COMPILATION ON 
                          POSITION_INDEPENDENT_CODE ON
                          LINKER_LANGUAGE CUDA)
              
endif()