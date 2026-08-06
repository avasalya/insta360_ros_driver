#include "perspective.hpp"
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <cmath>
#include <thread>
#include <chrono>
#include <algorithm>

PerspectiveNode::PerspectiveNode()
    : Node("perspective_node"),
      maps_initialized_(false),
      params_changed_(true),
      new_orientation_(false),
      img_height_(0),
      img_width_(0)
{
    // Initialize orientation to Identity
    camera_orientation_matrix_ = cv::Matx33d::eye();

    declare_parameter("cx_offset", 0.0);
    declare_parameter("cy_offset", 0.0);
    declare_parameter("crop_size", 960);
    declare_parameter("translation", std::vector<double>{0.0, 0.0, -0.105});
    declare_parameter("rotation_deg", std::vector<double>{-0.5, 0.0, 1.1});
    declare_parameter("gpu", true);
    declare_parameter("out_width", 1152);
    declare_parameter("out_height", 1152);
    declare_parameter("horizontal_fov", 120.0);
    declare_parameter("vertical_fov", 81.7867893);

    loadParameters();

    RCLCPP_INFO(get_logger(), "C++ perspective node (Dynamic Orientation Optimized)");

    params_callback_handle = add_on_set_parameters_callback(
        std::bind(&PerspectiveNode::parametersCallback, this, std::placeholders::_1));

    updateCameraParameters();

    auto qos = rclcpp::QoS(1).best_effort(); // Use best effort for high-freq image data to prevent queuing freezes

    dual_fisheye_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/dual_fisheye/image", qos,
        std::bind(&PerspectiveNode::imageCallback, this, std::placeholders::_1));

    camera_orientation_sub_ = create_subscription<geometry_msgs::msg::Quaternion>(
        "/camera_orientation/quaternion", rclcpp::QoS(10), // Short queue for IMU
        [this](const geometry_msgs::msg::Quaternion::SharedPtr msg) {
            tf2::fromMsg(*msg, camera_orientation_quaternion_);
            tf2::Matrix3x3 matrix(camera_orientation_quaternion_);
            camera_orientation_matrix_ = cv::Matx33d(
                matrix[0][0], matrix[0][1], matrix[0][2],
                matrix[1][0], matrix[1][1], matrix[1][2],
                matrix[2][0], matrix[2][1], matrix[2][2]);
            new_orientation_ = true;
        });

    perspective_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "/perspective/image", qos);
}

PerspectiveNode::~PerspectiveNode() {}

void PerspectiveNode::loadParameters()
{
    try {
        cx_offset_ = get_parameter("cx_offset").as_double();
        cy_offset_ = get_parameter("cy_offset").as_double();
        out_width_ = get_parameter("out_width").as_int();
        out_height_ = get_parameter("out_height").as_int();
        gpu_enabled_ = get_parameter("gpu").as_bool();
        horizontal_fov_ = get_parameter("horizontal_fov").as_double();
        vertical_fov_ = get_parameter("vertical_fov").as_double();
        crop_size_ = get_parameter("crop_size").as_int();

        auto translation = get_parameter("translation").as_double_array();
        tx_ = translation[0]; ty_ = translation[1]; tz_ = translation[2];

        auto rotation_deg = get_parameter("rotation_deg").as_double_array();
        roll_ = rotation_deg[0] * M_PI / 180.0;
        pitch_ = rotation_deg[1] * M_PI / 180.0;
        yaw_ = rotation_deg[2] * M_PI / 180.0;

        RCLCPP_INFO(get_logger(), "Loaded parameters from ROS parameter server");
        RCLCPP_INFO(get_logger(), "  Crop size: %d", crop_size_);
        RCLCPP_INFO(get_logger(), "  Center offset: (%.1f, %.1f)", cx_offset_, cy_offset_);
        RCLCPP_INFO(get_logger(), "  Translation: [%.3f, %.3f, %.3f]", tx_, ty_, tz_);
        RCLCPP_INFO(get_logger(), "  Rotation (deg): [%.1f, %.1f, %.1f]", 
                    rotation_deg[0], rotation_deg[1], rotation_deg[2]);
        RCLCPP_INFO(get_logger(), "  Output size: %dx%d", out_width_, out_height_);
        RCLCPP_INFO(get_logger(), "  Horizontal FOV: %.1f", horizontal_fov_);
        RCLCPP_INFO(get_logger(), "  Vertical FOV: %.1f", vertical_fov_);
        RCLCPP_INFO(get_logger(), "  GPU enabled: %s", gpu_enabled_ ? "true" : "false");
        RCLCPP_INFO(get_logger(), "Loaded parameters: Output %dx%d", out_width_, out_height_);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error loading parameters: %s", e.what());
        throw;
    }
}

void PerspectiveNode::updateCameraParameters()
{
    cv::Matx33d Rx(1.0, 0.0, 0.0, 0.0, cos(roll_), -sin(roll_), 0.0, sin(roll_), cos(roll_));
    cv::Matx33d Ry(cos(pitch_), 0.0, sin(pitch_), 0.0, 1.0, 0.0, -sin(pitch_), 0.0, cos(pitch_));
    cv::Matx33d Rz(cos(yaw_), -sin(yaw_), 0.0, sin(yaw_), cos(yaw_), 0.0, 0.0, 0.0, 1.0);

    back_to_front_rotation_ = Rz * Ry * Rx;
    back_to_front_translation_ = cv::Vec3d(tx_, ty_, tz_);

    params_changed_ = true;
}

// Memory Allocation is isolated here so it NEVER happens during active video streaming
void PerspectiveNode::allocateMaps(int rows_size, int cols_size)
{
    img_height_ = rows_size;
    img_width_ = cols_size;

    int current_crop_size = crop_size_;
    y_offset_crop_ = 0;
    x_offset_crop_ = 0;

    img_height_crop_ = img_height_;
    img_width_crop_ = img_width_;

    if (img_height_ != current_crop_size || img_width_ != current_crop_size) {
        int y_start = (img_height_ - current_crop_size) / 2;
        int x_start = (img_width_ - current_crop_size) / 2;

        if (y_start >= 0 && x_start >= 0) {
            img_height_crop_ = current_crop_size;
            img_width_crop_ = current_crop_size;
            y_offset_crop_ = (img_height_ - crop_size_) / 2;
            x_offset_crop_ = (img_width_ - crop_size_) / 2;
        }
    }

    cx_ = img_width_crop_ / 2.0f + cx_offset_;
    cy_ = img_height_crop_ / 2.0f + cy_offset_;

    // Allocate persistent memory ONCE
    if (map_x_.size() != cv::Size(out_width_, out_height_)) {
        map_x_ = cv::Mat(out_height_, out_width_, CV_32FC1);
        map_y_ = cv::Mat(out_height_, out_width_, CV_32FC1);
    }

    maps_initialized_ = true;
}

// Blazing fast update using existing memory and hoisted math
void PerspectiveNode::updateMaps()
{
    const float tan_horizontal = std::tan(horizontal_fov_ / 2.0 * M_PI / 180.0);
    const float tan_vertical = std::tan(vertical_fov_ / 2.0 * M_PI / 180.0);

    const float c00 = camera_orientation_matrix_(0,0), c01 = camera_orientation_matrix_(0,1), c02 = camera_orientation_matrix_(0,2);
    const float c10 = camera_orientation_matrix_(1,0), c11 = camera_orientation_matrix_(1,1), c12 = camera_orientation_matrix_(1,2);
    const float c20 = camera_orientation_matrix_(2,0), c21 = camera_orientation_matrix_(2,1), c22 = camera_orientation_matrix_(2,2);

    const float r00 = back_to_front_rotation_(0,0), r01 = back_to_front_rotation_(0,1), r02 = back_to_front_rotation_(0,2);
    const float r10 = back_to_front_rotation_(1,0), r11 = back_to_front_rotation_(1,1), r12 = back_to_front_rotation_(1,2);
    const float r20 = back_to_front_rotation_(2,0), r21 = back_to_front_rotation_(2,1), r22 = back_to_front_rotation_(2,2);
    const float tx = back_to_front_translation_(0), ty = back_to_front_translation_(1), tz = back_to_front_translation_(2);

    const float Z_ray = -1.0f;

    // PRECOMPUTE X_rays to save thousands of multiplies per frame
    std::vector<float> X_rays(out_width_);
    for(int x = 0; x < out_width_; ++x) {
        X_rays[x] = ((float)x / out_width_) * 2.0f * tan_horizontal - tan_horizontal;
    }

    cv::parallel_for_(cv::Range(0, out_height_), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            float Y_ray = ((float)y / out_height_) * 2.0f * tan_vertical - tan_vertical;

            // HOISTING: These are perfectly constant for the entire row
            float base_X = c01 * Y_ray + c02 * Z_ray;
            float base_Y = c11 * Y_ray + c12 * Z_ray;
            float base_Z = c21 * Y_ray + c22 * Z_ray;

            // Access directly into the pre-allocated memory pointers
            float* ptr_map_x = map_x_.ptr<float>(y);
            float* ptr_map_y = map_y_.ptr<float>(y);

            for (int x = 0; x < out_width_; ++x) {
                float X_ray = X_rays[x];

                // Extremely fast per-pixel rotation
                float X_rot = c00 * X_ray + base_X;
                float Y_rot = c10 * X_ray + base_Y;
                float Z_rot = c20 * X_ray + base_Z;

                // Hardware 90-degree sensor roll correction
                float X_val = Y_rot;
                float Y_val = -X_rot;
                float Z_val = Z_rot;

                bool is_front = (Z_val >= 0.0f);

                if (!is_front) {
                    float X_trans = r00 * X_val + r01 * Y_val + r02 * Z_val + tx;
                    float Y_trans = r10 * X_val + r11 * Y_val + r12 * Z_val + ty;
                    float Z_trans = r20 * X_val + r21 * Y_val + r22 * Z_val + tz;

                    X_val = -X_trans;
                    Y_val = Y_trans;
                    Z_val = Z_trans;
                }

                float r = std::sqrt(X_val * X_val + Y_val * Y_val);
                r = std::max(r, 1e-6f); // Prevent divide by zero

                float theta = std::atan2(r, std::fabs(Z_val));
                float r_fisheye = (2.0f * theta / M_PI) * (img_width_crop_ / 2.0f);

                float u = cx_ + (X_val / r) * r_fisheye;
                float v = cy_ + (Y_val / r) * r_fisheye;

                if (is_front) {
                    ptr_map_x[x] = img_width_crop_ + x_offset_crop_ + ((img_height_crop_ - 1.0f) - v);
                    ptr_map_y[x] = y_offset_crop_ + u;
                } else {
                    ptr_map_x[x] = x_offset_crop_ + v;
                    ptr_map_y[x] = y_offset_crop_ + ((img_width_crop_ - 1.0f) - u);
                }
            }
        }
    });
}

void PerspectiveNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr dual_fisheye_msg)
{
    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(dual_fisheye_msg, "rgb8");
        int rows_size = cv_ptr->image.rows;
        int cols_size = cv_ptr->image.cols / 2;

        // Ensure memory is allocated based on camera resolution
        if (!maps_initialized_ || rows_size != img_height_ || cols_size != img_width_) {
            allocateMaps(rows_size, cols_size);
            params_changed_ = true; // Force map update on next step
        }

        // Only rebuild maps if IMU has moved or parameters changed
        if (new_orientation_ || params_changed_) {
            updateMaps();
            new_orientation_ = false;
            params_changed_ = false;
        }

        // Remap straight from the Float32 arrays. Skips massive CPU bottleneck.
        cv::remap(cv_ptr->image, perspective_img, map_x_, map_y_, cv::INTER_LINEAR);

        cv_bridge::CvImage out_msg;
        out_msg.header = dual_fisheye_msg->header;
        out_msg.encoding = "rgb8";
        out_msg.image = perspective_img;
        perspective_pub_->publish(*out_msg.toImageMsg());

    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    }
}

rcl_interfaces::msg::SetParametersResult PerspectiveNode::parametersCallback(
    const std::vector<rclcpp::Parameter> &parameters)
{
    bool update_needed = false;

    for (const auto &param : parameters) {
        if (param.get_name() == "cx_offset") {
            cx_offset_ = param.as_double();
            update_needed = true;
        }
        else if (param.get_name() == "cy_offset") {
            cy_offset_ = param.as_double();
            update_needed = true;
        }
        else if (param.get_name() == "crop_size") {
            crop_size_ = param.as_int();
            update_needed = true;
        }
        else if (param.get_name() == "out_width") {
            out_width_ = param.as_int();
            update_needed = true;
        }
        else if (param.get_name() == "out_height") {
            out_height_ = param.as_int();
            update_needed = true;
        }
        else if (param.get_name() == "gpu") {
            gpu_enabled_ = param.as_bool();
            update_needed = true;
        }
        else if (param.get_name() == "translation") {
            auto translation = param.as_double_array();
            if (translation.size() >= 3) {
                tx_ = translation[0];
                ty_ = translation[1];
                tz_ = translation[2];
                update_needed = true;
            }
        }
        else if (param.get_name() == "rotation_deg") {
            auto rotation_deg = param.as_double_array();
            if (rotation_deg.size() >= 3) {
                roll_ = rotation_deg[0] * M_PI / 180.0;
                pitch_ = rotation_deg[1] * M_PI / 180.0;
                yaw_ = rotation_deg[2] * M_PI / 180.0;
                update_needed = true;
            }
        } else if (param.get_name() == "horizontal_fov") {
            horizontal_fov_ = param.as_double();
            update_needed = true;
        } else if (param.get_name() == "vertical_fov") {
            vertical_fov_ = param.as_double();
            update_needed = true;
        }
    }

    if (update_needed) {
        updateCameraParameters();
    }

    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    params_changed_ = true;
    return result;
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<PerspectiveNode>();

    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "Exception during spin: %s", e.what());
    }

    rclcpp::shutdown();
    return 0;
}