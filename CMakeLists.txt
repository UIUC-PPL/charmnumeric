# Python package build (pip install . via scikit-build-core): the charmtyles core,
# server.out and the _native_region extension. src/ also builds standalone.
cmake_minimum_required(VERSION 3.20)
project(charmnumeric LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
set(_pyx ${CMAKE_CURRENT_SOURCE_DIR}/charmnumeric/_native_region.pyx)
set(_cpp ${CMAKE_CURRENT_BINARY_DIR}/_native_region.cpp)
add_custom_command(
    OUTPUT ${_cpp}
    COMMAND Python::Interpreter -m cython --cplus -3 -I ${CHARMTYLES_HOME} ${_pyx} -o ${_cpp}
    DEPENDS ${_pyx} ${CHARMTYLES_HOME}/charmtyles/_native_region.pxd
    COMMENT "Cythonizing _native_region.pyx")
python_add_library(_native_region MODULE WITH_SOABI ${_cpp})
target_include_directories(_native_region PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src ${CHARMTYLES_HOME}/include ${CHARM_HOME}/include)
target_link_libraries(_native_region PRIVATE Python::NumPy)
target_compile_options(_native_region PRIVATE -O3 -DNDEBUG)
install(TARGETS _native_region LIBRARY DESTINATION charmnumeric)
