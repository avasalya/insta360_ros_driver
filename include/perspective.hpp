#ifndef PERSPECTIVE_HPP
#define PERSPECTIVE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class PerspectiveNode : public rclcpp::Node
{
public:
    PerspectiveNode();
    ~PerspectiveNode();

private:
    void loadParameters();
    void updateCameraParameters();

    // New optimized map functions
    void allocateMaps(int rows_size, int cols_size);
    void updateMaps();

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter> &parameters);

    // ROS Subscriptions & Publishers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr dual_fisheye_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Quaternion>::SharedPtr camera_orientation_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr perspective_pub_;
    OnSetParametersCallbackHandle::SharedPtr params_callback_handle;

    // State Tracking
    bool maps_initialized_;
    bool params_changed_;
    bool new_orientation_;

    // Image / Crop Dimensions
    int img_height_;
    int img_width_;
    int img_height_crop_;
    int img_width_crop_;
    int x_offset_crop_;
    int y_offset_crop_;

    // Node Parameters
    double cx_offset_;
    double cy_offset_;
    int crop_size_;
    int out_width_;
    int out_height_;
    bool gpu_enabled_;
    double horizontal_fov_;
    double vertical_fov_;

    // Camera Calibration
    double tx_, ty_, tz_;
    double roll_, pitch_, yaw_;
    double cx_, cy_;

    // Transformation Matrices
    cv::Matx33d camera_orientation_matrix_;
    tf2::Quaternion camera_orientation_quaternion_;
    cv::Matx33d back_to_front_rotation_;
    cv::Vec3d back_to_front_translation_;

    // Persistent Memory Maps (Float32 for speed)
    cv::Mat map_x_;
    cv::Mat map_y_;
    cv::Mat perspective_img;
};

#endif // PERSPECTIVE_HPP