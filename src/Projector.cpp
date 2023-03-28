#include <lidar_proposal_image_classification/Projector.hpp>
#include <iostream>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <sensor_msgs/Image.h>
#include <tf2/LinearMath/Quaternion.h>
#include <sstream>
#include <chrono>
#include "ros/ros.h"
#include "ros/package.h"
#include <opencv2/opencv.hpp>
#include <fs_msgs/Cone.h>
#include <fs_msgs/Cones.h>
#include <math.h>
#include <algorithm>


Projector::Projector() : nh_("~"), n_(),
        img_sub_(nh_, "sub_topic_img", 10),
        cones_sub_(nh_, "sub_topic_cones", 10),
        tfListener(tfBuffer),
        img_sync(MySyncPolicy(200), img_sub_, cones_sub_){

            // Define Publisher
            cones_pub_ = nh_.advertise<sensor_msgs::Image>("image", 1000);

            // Register message filter callback
            img_sync.registerCallback(boost::bind(&Projector::callback, this, _1, _2 ));

            // Get ROS Parameters
            nh_.getParam("focal_x", focal_x_);
            nh_.getParam("focal_y", focal_y_);
            nh_.getParam("cx", cx_);
            nh_.getParam("cy", cy_);
            nh_.getParam("skew", skew_);
            nh_.getParam("k1", k1_);
            nh_.getParam("k2", k2_);
            nh_.getParam("k3", k3_);
            nh_.getParam("p1", p1_);
            nh_.getParam("p2", p2_);
            nh_.getParam("width", width_);
            nh_.getParam("height", height_);
            nh_.getParam("get_auto_tf", get_automatic_transform_);
            nh_.getParam("frame_id_lidar", frame_id_lidar_);
            nh_.getParam("frame_id_cam", frame_id_cam_);
            
            // Cone classifier parameters
            nh_.getParam("bb_height_coef", bb_height_coef);
            nh_.getParam("bb_width_factor", bb_width_factor);
            nh_.getParam("classifier_img_size", classifier_img_size);
            nh_.getParam("engine_path", engine_path);
            XmlRpc::XmlRpcValue class_values;
            nh_.getParam("classes", class_values);
            XmlRpc::XmlRpcValue color_values;
            nh_.getParam("colors", color_values);

            // Color value parser
            // colors.resize(color_values.size(), std::vector<int>(3));
            colors.resize(color_values.size(), std::vector<int>(3, 0));
            ROS_ASSERT(color_values.getType() == XmlRpc::XmlRpcValue::TypeArray);
            for (int32_t i = 0; i < color_values.size(); ++i) {
                ROS_ASSERT(color_values[i].getType() == XmlRpc::XmlRpcValue::TypeArray);
                for (int32_t j = 0; j < color_values[i].size(); ++j) {
                    ROS_ASSERT(color_values[i][j].getType() == XmlRpc::XmlRpcValue::TypeInt);
                    colors.at(i).at(j) = static_cast<int>(color_values[i][j]);
                }
            }

            // Class parser
            classes.resize(class_values.size());
            ROS_ASSERT(class_values.getType() == XmlRpc::XmlRpcValue::TypeArray);
                for (int32_t i = 0; i < class_values.size(); ++i) {
                    ROS_ASSERT(class_values[i].getType() == XmlRpc::XmlRpcValue::TypeString);
                    classes.at(i) = static_cast<std::string>(class_values[i]);
                }

            // Get Intrinsic matrix
            i_mat = cv::Mat(3, 3, cv::DataType<double>::type);
            i_mat.at<double>(0, 0) = focal_x_;
            i_mat.at<double>(1, 0) = skew_;
            i_mat.at<double>(2, 0) = 0;

            i_mat.at<double>(0, 1) = 0;
            i_mat.at<double>(1, 1) = focal_y_;
            i_mat.at<double>(2, 1) = 0;

            i_mat.at<double>(0, 2) = cx_;
            i_mat.at<double>(1, 2) = cy_;
            i_mat.at<double>(2, 2) = 1;
     
            ROS_INFO("[LiProIC] Got intrinsic matrix.");

            // Distortion coefficients
            dist_coeffs = cv::Mat(5, 1, cv::DataType<double>::type);
            dist_coeffs.at<double>(0) = k1_;
            dist_coeffs.at<double>(1) = k2_;
            dist_coeffs.at<double>(2) = p1_;
            dist_coeffs.at<double>(3) = p2_;
            dist_coeffs.at<double>(4) = k3_;

            // Get Lidar->Camera transform
            t_vec = cv::Mat(3, 1, cv::DataType<double>::type);

            // Set up the image classifier
            std::string path = ros::package::getPath("lidar_proposal_image_classification") + engine_path;
            const char *planPath = path.c_str();

            ROS_INFO("[LiProIC] Loading engine from %s", planPath);

            std::vector<char> plan;
            engine.ReadPlan(planPath, plan);

            engine.Init(plan);
            engine.DiagBindings();

            if(get_automatic_transform_){
                ros::Rate rate(1.0);
                bool _got_tf = false;
                while (nh_.ok() && !_got_tf){
                    try{
                        transformStamped = tfBuffer.lookupTransform(frame_id_cam_, frame_id_lidar_, ros::Time(0));
                        _got_tf = true;
                    }
                    catch (tf2::TransformException &ex){
                        ROS_WARN("%s",ex.what());
                        ros::Duration(1.0).sleep();
                        continue;
                    }
                }
                // Set translation vector
                t_vec.at<double>(0) = transformStamped.transform.translation.x;
                t_vec.at<double>(1) = transformStamped.transform.translation.y;
                t_vec.at<double>(2) = transformStamped.transform.translation.z;
            } else {
                XmlRpc::XmlRpcValue t_values;
                nh_.getParam("t_vec", t_values);
                ROS_ASSERT(t_values.getType() == XmlRpc::XmlRpcValue::TypeArray);
                for (int32_t i = 0; i < t_values.size(); ++i) {
                    ROS_ASSERT(t_values[i].getType() == XmlRpc::XmlRpcValue::TypeDouble);
                    t_vec.at<double>(i) = static_cast<double>(t_values[i]);
                }
            }

            std::cout << "Translation vector: " << t_vec.at<double>(0) << " " << t_vec.at<double>(1) << " "<< t_vec.at<double>(2) << std::endl;
            ROS_INFO("[LiProIC] Got lidar->camera transform.");            

            // Transform quaternions and set rotation matrix
            // TODO: add temporary additional rotation from transformed lidar frame back to x=forward, y=left, z=up to use the TF values again
            tf2::Quaternion q;
            q.setW(0.5); //(transformStamped.transform.rotation.w);
            q.setX(0.5); //(transformStamped.transform.rotation.x);
            q.setY(-0.5); //(transformStamped.transform.rotation.y);
            q.setZ(0.5); //(transformStamped.transform.rotation.z);
            std::cout << "Qaternion X,Y,Z,W: " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;

            tf2::Matrix3x3 m;
            m.setRotation(q);

            r_mat = cv::Mat(3, 3, cv::DataType<double>::type);
            for(int i = 0; i <= 2; i++){
                for(int j = 0; j <= 2; j++){
                    r_mat.at<double>(i, j) = m[i][j];
                    std::cout << r_mat.at<double>(i, j) << " ";
                }
                std::cout << std::endl;
            }
            ROS_INFO("[LiProIC] Set up extrinsic matrix.");

            ROS_INFO("[LiProIC] Setup finished successfully!");    

}


// Main callback of the Projector
void Projector::callback(const sensor_msgs::Image::ConstPtr& img_msg, const fs_msgs::Cones::ConstPtr& cones_msg){

    // Read image data and convert it to OpenCV format
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    cv_bridge::CvImagePtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e){
        ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "[TIME] Read in image data = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  << "s"  <<std::endl;

    // Read point read cone data and convert it to cv vector
    begin = std::chrono::steady_clock::now();

    std::vector<cv::Point3d> pts = Projector::conesToCvVec(cones_msg);

    end = std::chrono::steady_clock::now();
    std::cout << "[TIME] Read & transform cone data = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  << "s" <<std::endl;


    // Do the projection
    begin = std::chrono::steady_clock::now();
    std::vector<cv::Point2d> image_points;
    cv::projectPoints(pts, r_mat, t_vec, i_mat,
                    dist_coeffs, image_points);

    end = std::chrono::steady_clock::now();
    std::cout << "[TIME] Projection = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  << "s"  <<std::endl;    

    // Remove points that are outside of the image
    int arr_size = image_points.size();
    for (int i = 0; i < arr_size; i++){
        if (image_points[i].x < 0 || image_points[i].x > width_ || image_points[i].y < 0 || image_points[i].y > height_){
            image_points.erase(image_points.begin() + i);
            pts.erase(pts.begin() + i);
            i--;
            arr_size--;
        }
    }

    // Get cone proposal images for classification
    begin = std::chrono::steady_clock::now();

    std::vector<cv::Rect> bbs;

    std::vector<cv::Mat> cone_imgs;

    for (int i = 0; i < pts.size(); i++){
        int cone_height = Projector::estimateConeHeight(pts[i], bb_height_coef, height_);
        cv::Rect bb = Projector::defineBoundingBox(image_points[i], cone_height, bb_width_factor);
        bbs.push_back(bb);
        cv::Mat cropped_img;
        cv::resize(cv_ptr->image(bb), cropped_img, cv::Size(classifier_img_size, classifier_img_size));
        cone_imgs.push_back(cropped_img);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "[TIME] Create cone images = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  << "s"  <<std::endl;

    // Classify the cone proposal images
    begin = std::chrono::steady_clock::now();
    std::vector<int> class_preds;
    std::vector<float> class_pred_confs;

    std::vector<float> output;

    int one_image_mem = cone_imgs[0].total()*3;
    int curr_batch_size = cone_imgs.size();
    std::vector<float> image_vec(one_image_mem*curr_batch_size);

    int offset = 0;
    for (int i = 0; i < curr_batch_size; i++){
        // // Convert the image from BGR to RGB color space
        cv::cvtColor(cone_imgs[i], cone_imgs[i], cv::COLOR_BGR2RGB);
        cv::Mat transposedImage;
        cv::transpose(cone_imgs[i], transposedImage);
        transposedImage = transposedImage.reshape(0, 1); // flatten the image into a row vector

        cv::Mat image_float;
        transposedImage.convertTo(image_float, CV_32FC3, 1.0 / 255);
        // cone_imgs[i].convertTo(image_float, CV_32FC3, 1.0 / 255);

        memcpy(&image_vec[offset], image_float.data, one_image_mem * sizeof(float)); // copy the data directly
        offset += one_image_mem;
    }

    engine.Infer(image_vec, output, classifier_img_size, curr_batch_size);
    int outsize = output.size();

    offset = classes.size();
    for (int i = 0; i < curr_batch_size; i++){
        std::vector<float> one_output(output.begin() + i*offset, output.begin() + (i+1)*offset);
        // Softmax(static_cast<int>(one_output.size()), one_output.data());
        one_output = new_softmax(one_output);
        auto max_element_it = std::max_element(one_output.begin(), one_output.end());
        float pred_conf = *max_element_it;
        class_pred_confs.push_back(pred_conf);
        int class_pred = std::distance(one_output.begin(), max_element_it);
        class_preds.push_back(class_pred);
    }
    
    end = std::chrono::steady_clock::now();
    std::cout << "[TIME] Cone classification = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  << "s"  <<std::endl;

    // Draw the points onto the image
    begin = std::chrono::steady_clock::now();

    Projector::drawBBsOnImg(cv_ptr, image_points, bbs, class_preds, class_pred_confs, width_, height_);

    end = std::chrono::steady_clock::now();
    std::cout << "[TIME] Draw on image = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  << "s"  <<std::endl;


    // Publish the new image
    begin = std::chrono::steady_clock::now();
    cones_pub_.publish(cv_ptr->toImageMsg());
    end = std::chrono::steady_clock::now();
    std::cout << "[TIME] Publish image = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  << "s"  <<std::endl;
}


// Function to transform ros pc2 messages into a vector of 3D points
std::vector<cv::Point3d> Projector::conesToCvVec (const fs_msgs::Cones::ConstPtr& cones_msg){

    std::vector<cv::Point3d> pts(cones_msg->cones.size());

    for(int i = 0; i < cones_msg->cones.size(); i++){
        pts[i].x = cones_msg->cones[i].x; 
        pts[i].y = cones_msg->cones[i].y;
        pts[i].z = cones_msg->cones[i].z;   
    }
    return pts;
}


// Function that draws projected points onto the original image
void Projector::drawBBsOnImg(cv_bridge::CvImagePtr& cv_ptr, const std::vector<cv::Point2d>& pts2d, const std::vector<cv::Rect>& bbs, const std::vector<int>& class_preds, const std::vector<float>& class_pred_confs, int width, int height){

    for (int i = 0; i < pts2d.size(); i++) {
        int x = static_cast<int>(pts2d[i].x);
        int y = static_cast<int>(pts2d[i].y);
  
        int cid = class_preds[i];
        cv::circle(cv_ptr->image, cv::Point(x, y), 1, CV_RGB(0,255,0), -1, 0);
        cv::rectangle(cv_ptr->image, bbs[i], CV_RGB(colors.at(cid).at(0),colors.at(cid).at(1),colors.at(cid).at(2)), 1, cv::LINE_8);
    }
}

int Projector::estimateConeHeight(const cv::Point3d& pt3d, float coeff, int img_height){
    float dist = sqrt((pow(pt3d.x,2)) + (pow(pt3d.y,2)));
    float bb_height_relation_factor = 1/(1 + coeff * dist);
    float bb_height = img_height * bb_height_relation_factor;
    return static_cast<int>(bb_height);
}

cv::Rect Projector::defineBoundingBox(const cv::Point2d& pt2d, int cone_height, float width_factor){
    int cone_width = static_cast<int>(cone_height * width_factor);
    cv::Rect bb(
        static_cast<int>(pt2d.x - cone_width/2),
        static_cast<int>(pt2d.y - cone_height/2),
        static_cast<int>(cone_width),
        static_cast<int>(cone_height)
    );
    return bb;
}
