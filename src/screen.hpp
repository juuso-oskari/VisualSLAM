#ifndef VISUAL_SLAM_SCREEN
#define VISUAL_SLAM_SCREEN

#include <opencv2/core/core.hpp>

#include <easy3d/viewer/viewer.h>
#include <easy3d/renderer/camera.h>
#include <easy3d/renderer/renderer.h>
#include <easy3d/renderer/camera.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/drawable_triangles.h>
#include <easy3d/core/graph.h>
#include <easy3d/core/types.h>
#include <easy3d/core/point_cloud.h>
#include <easy3d/util/timer.h>
#include <easy3d/util/logging.h>
#include <easy3d/util/resource.h>
#include <easy3d/util/initializer.h>

#include "helper_functions.hpp"
#include <vector>
#include <thread>

/**
 * @brief The screen class is used to interface with the viewer output. 
 * It manages the state of the UI, holds handles to the OpenGL buffers 
 * used by Easy3D, and other things required to keep the visualization 
 * running. The screen class does NOT keep track of the point clouds
 * used by the program. That responsibility is in the PointClouds class.
 * 
 */
class Screen {
public:
    /**
     * @brief Initialize Easy3D and construct a new Screen object. 
     * This will also do some configuration to optimize the output.
     * 
     * @param title The title of the visualizer window
     */
    explicit Screen(std::string title) {
        viewer_ = new easy3d::Viewer(title);
        // Initialize some parameters to make the output easier to navigate
        viewer_->camera()->setViewDirection(easy3d::vec3(1, 0, 0));
        viewer_->camera()->setUpVector(easy3d::vec3(1, 0, 0));
    }
    /**
     * @brief Registers a point cloud to be rendered using this screen
     * object. Any updates sent to the cloud will now be shown on screen.
     * 
     * @param cloud 
     */
    void RegisterPointCloud(easy3d::PointCloud* cloud) {
        viewer_->add_model(cloud);
    }
    /**
     * @brief Blocking call; boots up the visualizer and runs until it completes
     * 
     * @return int the return code of the visualizer; 0 if success, nonzero otherwise
     */
    int Run() {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        return viewer_->run();
    }
private:
    easy3d::Viewer * viewer_;
};

/**
 * @brief Configures the settings of a point cloud / graph. This is typically done at
 * the start of the problem, before any points are added, but can be done during
 * the middle of execution as well.
 * 
 * @param model The model to configure
 * @param size The radius of the points being rendered
 * @param plain_style If true, points will be rendered as untextured pixels. 
 * If set to false instead (the default), points will be rendered as spheres.
 * @param color The color of points in the point cloud. Every point will have the
 * same color; you cannot customize the color of individual points. This is a vec4
 * in normalized (0.0 to 0.1) RGBA format.
 */
void ConfigureModel(
    easy3d::Model * model,
    float size = 1.0f,
    bool plain_style = false,
    easy3d::vec4 color = easy3d::vec4(1.0, 1.0, 1.0, 1.0)
) {
    auto vertices = model->renderer()->get_points_drawable("vertices");
    if (vertices != nullptr) {
        vertices->set_point_size(size);
        if (plain_style) {
            vertices->set_impostor_type(easy3d::PointsDrawable::PLAIN);
        }
        else {
            vertices->set_impostor_type(easy3d::PointsDrawable::SPHERE);
        }
        vertices->set_color(color);
    }
    auto edges = model->renderer()->get_lines_drawable("edges"); 
    if (edges != nullptr) {
        edges->set_line_width(size);
        if (plain_style) {
            edges->set_impostor_type(easy3d::LinesDrawable::PLAIN);
        }
        else {
            edges->set_impostor_type(easy3d::LinesDrawable::CYLINDER);
        }
        edges->set_color(color);
    }
}

/**
 * @brief Spawns a thread with access to the point clouds that 
 * can be used to manipulate their contents at runtime. This is
 * necessary as calling Screen::Run() will take control of the main
 * thread for visualization.
 * 
 * @param callback 
 */
void SpawnWorkerThread(std::function<void ()> callback) {
    easy3d::Timer<>::single_shot(0, [&](){
        callback();
    });
}

/**
 * @brief The PointClouds class manages the contents of two PointCloud objects, 
 * used to keep track of detected points and camera poses. This class can be 
 * interfaced to add and remove points from the clouds, as well as to request
 * a screen refresh from the Easy3D viewer.
 * 
 * Example usage:
 * 
 * PointClouds clouds(points, poses);
 * clouds.AddPoint(0.0, 1.0, -0.5) // Adds to the `points` point cloud
 * clouds.AddPoint(2.0, -1.0, 0.0, true) // Adds to the `poses` point cloud
 * clouds.UpdateView();
 * clouds.AddPoinstMatUpdate(<vector of cv::Mat objects>);
 * clouds.ClearAll();
 */
class PointClouds {
public:
    explicit PointClouds(
        easy3d::PointCloud * points,
        easy3d::PointCloud * poses,
        easy3d::Graph * pose_vectors
    ) : points_(points), poses_(poses), pose_vectors_(pose_vectors) {}
    /**
     * @brief Adds a single point to either of the point clouds, using xyz coordinates.
     * 
     * @param x 
     * @param y 
     * @param z 
     * @param poses If true, the point will be added to the poses point cloud.
     * Otherwise, the points point cloud will be used.
     */
    void AddPoint(double x, double y, double z, bool poses = false) {
        // It would be possible to use the same point cloud for both points and poses,
        // but this solution also works and requires less extra configuration.
        if (poses) {
            poses_->add_vertex(easy3d::vec3(x, y, z));
        }
        else {
            points_->add_vertex(easy3d::vec3(x, y, z));
        }
    }

    /**
     * @brief Adds a single point to either of the point clouds using cv::Mat format.
     * 
     * @param point 
     * @param poses If true, the point will be added to the poses point cloud.
     * Otherwise, the points point cloud will be used.
     */
    void AddPointMat(cv::Mat point, bool poses = false) {
        AddPoint(point.at<double>(0), point.at<double>(1), point.at<double>(2), poses);
    }
    /**
     * @brief Shorthand for AddPoint(x, y, z, true);
     * 
     * @param x 
     * @param y 
     * @param z 
     */
    void AddPose(double x, double y, double z) {
        AddPoint(x, y, z, true);
    }
    /**
     * @brief Adds a short vector on screen representing a camera facing direction.
     * 
     * @param x1 
     * @param y1 
     * @param z1 
     * @param x2 
     * @param y2 
     * @param z2 
     */
    void AddPoseAngle(double x1, double y1, double z1, double x2, double y2, double z2) {
        auto v1 = pose_vectors_->add_vertex(easy3d::vec3(x1, y1, z1));
        auto v2 = pose_vectors_->add_vertex(easy3d::vec3(x2, y2, z2));
        pose_vectors_->add_edge(v1, v2);
    }

    /**
     * @brief Same as AddPoseAnglesMat, but updates the view.
    */
    void AddPoseAnglesMatUpdate(std::vector<cv::Mat> poses) {
        for (auto angle : poses) {
            cv::Mat trans = -GetRotation(angle).inv() * GetTranslation(angle);
            auto rot = GetRotation(angle);
            cv::Mat i = (cv::Mat1d(3, 1) << 0.0, 0.0, 0.1);
            cv::Mat dir = trans + rot * i;
            AddPoseAngle(
                trans.at<double>(0),
                trans.at<double>(1),
                trans.at<double>(2),
                dir.at<double>(0),
                dir.at<double>(1),
                dir.at<double>(2)
            );
        }
        UpdateView();
    }
    /**
     * @brief Manually refresh the viewer associated with the point clouds.
     * 
     */
    void UpdateView() {
        points_->renderer()->update();
        poses_->renderer()->update();
        pose_vectors_->renderer()->update();
    }

    /**
     * @brief Remove all the points in the selected point cloud
     * 
     * @param poses If true, clear the POSES point cloud instead of the POINTS one.
     */
    void Clear(bool poses = false) {
        if (poses) {
            poses_->clear();
            pose_vectors_->clear();
        }
        else {
            points_->clear();
        }
    }

    /**
     * @brief Clears all points in all point clouds
     * 
     */
    void ClearAll() {
        Clear(true);
        Clear(false);
    }

    /**
     * @brief Similar to AddPointsMatUpdate(), except this will clear points first.
     * 
     * @param points points to replace current point cloud with
     * @param poses FALSE if the points point cloud should be replaced, TRUE if the
     * poses point cloud should be replaced
     */
    void SetPointsMatUpdate(std::vector<cv::Mat> points, bool poses = false) {
        Clear(poses);
        AddPointsMatUpdate(points, poses);
    }

    /**
     * @brief Extends the selected point cloud with the contents of the vector.
     * Signals for the viewer to refresh the screen.
     * 
     * @param points Vector of points in cv::Mat form
     * @param poses If true, add to the POSES point cloud instead of the POINTS one.
     */
    void AddPointsMatUpdate(std::vector<cv::Mat> points, bool poses = false) {
        for (auto point : points) {
            AddPointMat(point, poses);
        }
        UpdateView();
    }
private:
    easy3d::PointCloud * points_;
    easy3d::PointCloud * poses_;
    easy3d::Graph * pose_vectors_;
};

#endif