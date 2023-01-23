cmake_minimum_required(VERSION 3.11.0)
project(cpp_course_visual_slam VERSION 0.1.0)

enable_language(CXX)

# use C++ standard version 17
set(CMAKE_CXX_STANDARD 17)
# produce non-optimized code with debug symbols
set(CMAKE_BUILD_TYPE Debug)

message(STATUS "BUILD directory is ${CMAKE_BINARY_DIR}")
message(STATUS "SRC directory is ${CMAKE_SOURCE_DIR}")


# Find g2o
#find_package(g2o REQUIRED)
#message(STATUS "g2o directory is ${g2o_INCLUDE_DIRS}")
#include_directories( ${g2o_INCLUDE_DIRS} )

# Find opencv
find_package( OpenCV REQUIRED)
message(STATUS "OpenCV directory is ${OpenCV_INCLUDE_DIRS}")
include_directories( ${OpenCV_INCLUDE_DIRS} )
# Find Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)
# set(Easy3D_DIR "/usr/local/Cellar/easy3d/2.5.2/lib/CMake/Easy3D")
#include_directories(${CERES_INCLUDE_DIRS})

#set(G2O_USE_VENDORED_CERES ON CACHE BOOL "Use vendored Ceres from g2o/EXTERNAL/ceres")
# SET (G2O_LIB_TYPE SHARED)
LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set(G2O_INCLUDE_DIRS
/usr/local/include)
set(G2O_LIB_DIRS
/usr/local/lib)
include_directories(${G2O_INCLUDE_DIRS})
link_directories(${G2O_LIB_DIRS})
FIND_PACKAGE( G2O )
#INCLUDE_DIRECTORIES(
#${G2O_INCLUDE_DIRS}/core
#${G2O_INCLUDE_DIRS}/types
#${G2O_INCLUDE_DIRS}/stuff
#${G2O_INCLUDE_DIRS}/solvers)
#SET(G2O_LIBRARIES ${G2O_STUFF_LIBRARY} ${G2O_CORE_LIBRARY} ${G2O_SOLVER_CHOLMOD} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_DENSE} ${G2O_SOLVER_PCG} ${G2O_SOLVER_SLAM2D_LINEAR} ${G2O_SOLVER_STRUCTURE_ONLY} ${G2O_SOLVER_EIGEN} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_CSPARSE_EXTENSION})
# SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_viewer g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )
SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 )


# Include the FindOpenGL module
find_package(OpenGL REQUIRED)

# Add the include directories for the OpenGL headers
include_directories(${OPENGL_INCLUDE_DIR})
message(STATUS "OpenGL directory is ${OPENGL_INCLUDE_DIR}")




# Find Pangolin (https://github.com/stevenlovegrove/Pangolin)
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})


# Find YAML-CPP

find_package(YAML-CPP REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

find_package(Boost REQUIRED filesystem)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )


# Add all .c, - files under src/ directory
file(GLOB SOURCES_C "src/*.c")
# Add all .cpp, - files under src/ directory
file(GLOB SOURCES_CPP "src/*.cpp")
# Add all .hpp, - files under src/ directory
file(GLOB SOURCES_HPP "src/*.hpp")
# Add all .h, - files under src/ directory
file(GLOB SOURCES_H "src/*.h")
# Add all .yaml, - files under src/ directory
file(GLOB SOURCES_YAML "src/*.yaml")
#file(GLOB TEST_SOURCES_CPP "tests/*.cpp")


set(SOURCES ${SOURCES_C} ${SOURCES_CPP} ${SOURCES_H} ${SOURCES_HPP} ${TEST_SOURCES_CPP} ${SOURCES_YAML})
message(STATUS "SRC directory source code files: \n\t-- HEADERS: ${SOURCES_H}${SOURCES_HPP}\n\t-- C SOURCES: ${SOURCES_C}\n\t-- CPP SOURCE:${SOURCES_CPP}\n\t-- ALL:${SOURCES}")
# add the executable target
add_executable(run_slam)
# Add the OpenGL libraries to the target link libraries
target_link_libraries(run_slam ${TORCH_LIBRARIES})
target_link_libraries(run_slam pango_display)
target_link_libraries(run_slam ${OPENGL_LIBRARIES})
target_link_libraries(run_slam Eigen3::Eigen)
target_link_libraries(run_slam ${OpenCV_LIBS} )
target_link_libraries(run_slam ${G2O_LIBS})
target_link_libraries(run_slam Boost::filesystem)

target_link_libraries(run_slam ${YAML_CPP_LIBRARIES})
target_sources(run_slam PUBLIC ${SOURCES})
target_include_directories(run_slam PUBLIC .)
target_include_directories(run_slam PUBLIC src)


