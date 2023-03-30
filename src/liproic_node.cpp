#include <lidar_proposal_image_classification/Projector.hpp>
#include "ros/ros.h"

int main(int argc, char **argv)
{
// Initialize node
    ros::init(argc, argv, "lidar_projection_node");

    Projector PObject;

    ros::spin();

    return 0;
}