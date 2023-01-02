# Source content

main.cpp
- Main program execution.

map.hpp
- Map object holds maps (std::map), with pointers (std::shared_ptr) to Frame and Point3D objects as values and corresponding frame ids and point ids as keys. Map also contains the main algorithmic workload

point.hpp
- Point object stores estimated 3D location and map (std::map) with correspondences to imagepoints and features in Frame objects that see this map point. 

frame.hpp
- Frame object stores information processed from individual video frames. Also includes helper classes FeatureMatcher and FeatureExtractor.

helper_functions.hpp
- Various helper functions for repetitive procedures like slicing of matrices and conversions to different matrix types for cv::Mat and Eigen.

isometry3d.hpp & transformation.hpp
- These are helper classes that are supposed to make the transformation definitions easier (not currently used).
