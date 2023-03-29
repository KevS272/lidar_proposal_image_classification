#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2_ros/transform_listener.h>
#include <string>
#include <vector>
#include <list>
#include <geometry_msgs/TransformStamped.h>
#include <fs_msgs/Cone.h>
#include <fs_msgs/Cones.h>

#include <lidar_proposal_image_classification/Classifier.hpp>

class Projector{

    private:

        // Node-specific variables
        ros::NodeHandle nh_;
        ros::NodeHandle n_;
        tf2_ros::Buffer tfBuffer;
        tf2_ros::TransformListener tfListener;
        ros::Publisher cones_pub_;
        typedef message_filters::Subscriber<sensor_msgs::Image> ImageSubscriber;
        typedef message_filters::Subscriber<fs_msgs::Cones> ConeSubscriber;
        ImageSubscriber img_sub_;
        ConeSubscriber cones_sub_;
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, fs_msgs::Cones> MySyncPolicy;
        message_filters::Synchronizer<MySyncPolicy> img_sync;

        // Camera parameters
        float focal_x_;
        float focal_y_;
        float skew_;
        float cx_;
        float cy_;
        float k1_;
        float k2_;
        float k3_;
        float p1_;
        float p2_;
        int height_;
        int width_;
        // Other parameters
        bool get_automatic_transform_;
        std::string frame_id_lidar_; 
        std::string frame_id_cam_; 

        // Other necessary variables
        cv::Mat i_mat; // intrinsic camera matrix
        cv::Mat r_mat; // rotation part of extrinsic matrix
        cv::Mat t_vec; // translation part of extrinisc matrix
        cv::Mat dist_coeffs; //camera distortion coefficients
        geometry_msgs::TransformStamped transformStamped;
        std::vector<cv::Point2d> img_points_;
        Timer timer;

        // Bounding box size coefficient
        float bb_height_coef;
        float bb_width_factor;
        int classifier_img_size;
        std::string engine_path;

        // Classifier
        Engine engine;
        std::vector<std::string> classes;
        int num_classes;
        std::vector<std::vector<int>> colors;
        std::vector<int> class_preds;
        std::vector<float> class_pred_confs;
        std::vector<float> output;

    public:

        // Constructor
        Projector();

        // Main callback of the Projector
        void callback(const sensor_msgs::Image::ConstPtr& img_msg, const fs_msgs::Cones::ConstPtr& cones_msg);

        // Function to transform ros pc2 messages into a vector of 3D points
        std::vector<cv::Point3d> conesToCvVec (const fs_msgs::Cones::ConstPtr& cones_msg);

        // Function that draws projected points onto the original image
        void drawBBsOnImg(cv_bridge::CvImagePtr& cv_ptr, const std::vector<cv::Point2d>& pts2d, const std::vector<cv::Rect>& bbs, const std::vector<int>& class_preds, const std::vector<float>& class_pred_confs, int width, int height);
        
        // Function that stimates cone hight from 3D point
        int estimateConeHeight(const cv::Point3d& pt3d, float coeff, int img_height);

        // Function that defines the area to crop around the projected point
        cv::Rect defineBoundingBox(const cv::Point2d& pt2d, int cone_height, float width_factor);

};