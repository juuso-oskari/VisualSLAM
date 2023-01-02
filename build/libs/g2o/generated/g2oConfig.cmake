include(CMakeFindDependencyMacro)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules)

find_dependency(Eigen3)
find_dependency(OpenGL)
find_dependency(SuiteSparse)

include("${CMAKE_CURRENT_LIST_DIR}/g2oTargets.cmake")

