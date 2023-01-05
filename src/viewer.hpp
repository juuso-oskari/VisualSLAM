
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/gldraw.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <mutex>

class Viewer {
public:
    Viewer() {
        // pangolin::CreateWindowAndBind("Map Viewer",1024,768);
        // glEnable(GL_DEPTH_TEST);
        // glEnable(GL_BLEND);
        // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // s_cam_ = pangolin::OpenGlRenderState(
        //     pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),
        //     pangolin::ModelViewLookAt(0,-0.7,-1.8, 0,0,0, 0.0,-1.0, 0.0)
        // );

        // d_cam_ = &pangolin::CreateDisplay()
        //     .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
        //     .SetHandler(new pangolin::Handler3D(s_cam_));
    }
    
    void Run() {
        pangolin::CreateWindowAndBind("Map Viewer",1024,768);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        s_cam_ = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),
            pangolin::ModelViewLookAt(0,-0.7,-1.8, 0,0,0, 0.0,-1.0, 0.0)
        );

        d_cam_ = &pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam_));
        std::cout << "Viewer thread running" << std::endl;
        while(!pangolin::ShouldQuit()) {
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);  // Set background color to white
            glClear(GL_COLOR_BUFFER_BIT);
            glClear(GL_DEPTH_BUFFER_BIT);
            d_cam_->Activate(s_cam_);
            //glClearColor(1.0f,1.0f,1.0f,1.0f);
            AddPoints(points_);
            AddPoses(poses_);
            pangolin::FinishFrame();
        }
        std::cout << "Viewer thread stop" << std::endl;

    }
    void AddPoints(const std::vector<Eigen::Vector3d>& points) {
        glPointSize(2);
        glBegin(GL_POINTS);
        for(size_t i=0; i<points.size(); ++i) {
            glColor3f(0.0, 0.0, 0.0);
            glVertex3f(points[i][0], points[i][1], points[i][2]);
        }
        glEnd();
    }
    void AddPoses(const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& poses) {
        for (size_t i = 0; i < poses.size(); i++) {
            // Convert the pose to a 4x4 matrix
            Eigen::Matrix4d m = poses[i].matrix().inverse();
            // Draw the coordinate axes
            glBegin(GL_LINES);
            glColor3f(1.0,0.0,0.0);
            glVertex3d(m(0,3), m(1,3), m(2,3));
            glVertex3d(m(0,3) + 0.1 * m(0,0), m(1,3) + 0.1 * m(1,0), m(2,3) + 0.1 * m(2,0));
            glColor3f(0,1,0);
            glVertex3d(m(0,3), m(1,3), m(2,3));
            glVertex3d(m(0,3) + 0.1 * m(0,1), m(1,3) + 0.1 * m(1,1), m(2,3) + 0.1 * m(2,1));
            glColor3f(0,0,1);
            glVertex3d(m(0,3), m(1,3), m(2,3));
            glVertex3d(m(0,3) + 0.1* m(0,2), m(1,3) + 0.1 * m(1,2), m(2,3) + 0.1 * m(2,2));
            glEnd();
        }
        }
    void SetPoints(const std::vector<Eigen::Vector3d>& points) {
        std::lock_guard<std::mutex> lock(points_mutex_);
        points_ = points;
        //colors_ = colors;
    }

    void SetPoses(const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& poses) {
        std::lock_guard<std::mutex> lock(poses_mutex_);
        poses_ = poses;
    }
    private:
    std::vector<Eigen::Vector3d> points_;
    std::vector<Eigen::Vector3d> colors_;
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses_;
    pangolin::OpenGlRenderState s_cam_;
    pangolin::View* d_cam_;
    std::mutex points_mutex_;
    std::mutex poses_mutex_;
};