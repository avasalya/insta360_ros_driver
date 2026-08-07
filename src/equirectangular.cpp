#include "equirectangular.hpp"
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <cmath>
#include <thread>
#include <chrono>

EquirectangularNode::EquirectangularNode()
    : Node("equirectangular_node"),
      maps_initialized_(false),
      params_changed_(true),
      img_height_(0),
      img_width_(0)
{
    declare_parameter("cx_offset", 0.0);
    declare_parameter("cy_offset", 0.0);
    declare_parameter("crop_size", 960);
    declare_parameter("translation", std::vector<double>{0.0, 0.0, -0.105});
    declare_parameter("rotation_deg", std::vector<double>{-0.5, 0.0, 1.1});
    declare_parameter("gpu", true);
    declare_parameter("out_width", 1920);
    declare_parameter("out_height", 960);

    loadParameters();

    RCLCPP_INFO(get_logger(), "C++ equirectangular node (Optimized)");

    params_callback_handle = add_on_set_parameters_callback(
        std::bind(&EquirectangularNode::parametersCallback, this, std::placeholders::_1));

    updateCameraParameters();

    auto qos = rclcpp::QoS(1).reliable();

    dual_fisheye_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/dual_fisheye/image", qos,
        std::bind(&EquirectangularNode::imageCallback, this, std::placeholders::_1));

    equirect_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "/equirectangular/image", qos);
}

EquirectangularNode::~EquirectangularNode() {}

void EquirectangularNode::loadParameters()
{
    try {
        cx_offset_ = get_parameter("cx_offset").as_double();
        cy_offset_ = get_parameter("cy_offset").as_double();
        crop_size_ = get_parameter("crop_size").as_int();
        out_width_ = get_parameter("out_width").as_int();
        out_height_ = get_parameter("out_height").as_int();
        gpu_enabled_ = get_parameter("gpu").as_bool();

        auto translation = get_parameter("translation").as_double_array();
        tx_ = translation[0];
        ty_ = translation[1];
        tz_ = translation[2];

        auto rotation_deg = get_parameter("rotation_deg").as_double_array();
        roll_ = rotation_deg[0] * M_PI / 180.0;
        pitch_ = rotation_deg[1] * M_PI / 180.0;
        yaw_ = rotation_deg[2] * M_PI / 180.0;

        RCLCPP_INFO(get_logger(), "Loaded parameters from ROS parameter server");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error loading parameters: %s", e.what());
        gpu_enabled_ = true;
        throw;
    }
}

void EquirectangularNode::updateCameraParameters()
{
    cv::Matx33d Rx(
        1.0, 0.0, 0.0,
        0.0, cos(roll_), -sin(roll_),
        0.0, sin(roll_), cos(roll_)
    );

    cv::Matx33d Ry(
        cos(pitch_), 0.0, sin(pitch_),
        0.0, 1.0, 0.0,
        -sin(pitch_), 0.0, cos(pitch_)
    );

    cv::Matx33d Rz(
        cos(yaw_), -sin(yaw_), 0.0,
        sin(yaw_), cos(yaw_), 0.0,
        0.0, 0.0, 1.0
    );

    back_to_front_rotation_ = Rz * Ry * Rx;
    back_to_front_translation_ = cv::Vec3d(tx_, ty_, tz_);

    if (maps_initialized_) {
        maps_initialized_ = false;
        RCLCPP_INFO(get_logger(), "Parameters updated, remapping will occur on next image");
    }
}

void EquirectangularNode::initMapping(int img_height, int img_width)
{
    RCLCPP_INFO(get_logger(), "Initializing equirectangular projection: fusing two %dx%d fisheye images to %dx%d",
                img_width, img_height, out_width_, out_height_);

    img_height_ = img_height;
    img_width_ = img_width;
    int current_crop_size = crop_size_;
    int y_offset_crop = 0;
    int x_offset_crop = 0;

    if (img_height_ != current_crop_size || img_width_ != current_crop_size) {
        int y_start = (img_height_ - current_crop_size) / 2;
        int x_start = (img_width_ - current_crop_size) / 2;

        if (y_start >= 0 && x_start >= 0 &&
            y_start + current_crop_size <= img_height_ &&
            x_start + current_crop_size <= img_width_) {
            img_height = current_crop_size;
            img_width = current_crop_size;
            y_offset_crop = (img_height_ - crop_size_) / 2;
            x_offset_crop = (img_width_ - crop_size_) / 2;
        }
    }

    cx_ = img_width / 2.0f + cx_offset_;
    cy_ = img_height / 2.0f + cy_offset_;

    // Create temporary float maps
    cv::Mat temp_map_x(out_height_, out_width_, CV_32F);
    cv::Mat temp_map_y(out_height_, out_width_, CV_32F);

    // Pre-extract transformation matrix scalars to avoid cv::Matx overhead in the inner loop
    const float r00 = back_to_front_rotation_(0,0), r01 = back_to_front_rotation_(0,1), r02 = back_to_front_rotation_(0,2);
    const float r10 = back_to_front_rotation_(1,0), r11 = back_to_front_rotation_(1,1), r12 = back_to_front_rotation_(1,2);
    const float r20 = back_to_front_rotation_(2,0), r21 = back_to_front_rotation_(2,1), r22 = back_to_front_rotation_(2,2);
    const float tx = back_to_front_translation_(0), ty = back_to_front_translation_(1), tz = back_to_front_translation_(2);

    // Parallelize the map generation across Y rows
    cv::parallel_for_(cv::Range(0, out_height_), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {

            // Direct pointer access is significantly faster than .at<float>()
            float* ptr_map_x = temp_map_x.ptr<float>(y);
            float* ptr_map_y = temp_map_y.ptr<float>(y);

            float lat = ((float)y / out_height_) * M_PI - (M_PI / 2.0f);
            float cos_lat_val = std::cos(lat);
            float sin_lat_val = std::sin(lat);

            for (int x = 0; x < out_width_; ++x) {

                // To make front lense appear on the center of the equirectangular image.
                // Shifted by +PI to rotate the panorama 180 degrees horizontally
                float lon = ((float)x / out_width_) * 2.0f * M_PI;

                // To make rear lense appear on the center of the equirectangular image.
                // float lon = ((float)x / out_width_) * 2.0f * M_PI - M_PI;

                float cos_lon_val = std::cos(lon);
                float sin_lon_val = std::sin(lon);

                // 1. Calculate the raw un-rotated coordinates
                float orig_X = cos_lat_val * sin_lon_val;
                float orig_Y = sin_lat_val;

                // 2. Apply a 90-degree roll correction to fix the camera pose
                // (This swaps the X and Y axes to mathematically rotate the sphere)
                float X_val = orig_Y;
                float Y_val = -orig_X;
                float Z_val = cos_lat_val * cos_lon_val;

                bool is_front = (Z_val >= 0);

                if (!is_front) {
                    // Manual dot-product expansion is much faster than matrix multiplication here
                    float X_trans = r00 * X_val + r01 * Y_val + r02 * Z_val + tx;
                    float Y_trans = r10 * X_val + r11 * Y_val + r12 * Z_val + ty;
                    float Z_trans = r20 * X_val + r21 * Y_val + r22 * Z_val + tz;

                    X_val = -X_trans;
                    Y_val = Y_trans;
                    Z_val = Z_trans;
                }

                float r = std::sqrt(X_val * X_val + Y_val * Y_val);
                if (r < 1e-6f) r = 1e-6f;

                float theta = std::atan2(r, std::fabs(Z_val));
                float r_fisheye = (2.0f * theta / M_PI) * (img_width / 2.0f);

                float u = cx_ + (X_val / r) * r_fisheye;
                float v = cy_ + (Y_val / r) * r_fisheye;

                float rot_u, rot_v;

                if (is_front) {
                    rot_u = (img_height - 1) - v;
                    rot_v = u;
                } else {
                    rot_u = v;
                    rot_v = (img_width - 1) - u;
                }

                if (is_front) {
                    ptr_map_x[x] = img_width + x_offset_crop + rot_u;
                    ptr_map_y[x] = y_offset_crop + rot_v;
                } else {
                    ptr_map_x[x] = x_offset_crop + rot_u;
                    ptr_map_y[x] = y_offset_crop + rot_v;
                }
            }
        }
    });

    // OPTIMIZATION: Convert the float maps to fixed-point integer maps (CV_16SC2).
    // This allows cv::remap in the callback to run up to 3x faster per frame.
    cv::convertMaps(temp_map_x, temp_map_y, full_map_x_, full_map_y_, CV_16SC2);

    maps_initialized_ = true;
    RCLCPP_INFO(get_logger(), "Mapping matrices initialization complete");
}

void EquirectangularNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr dual_fisheye_msg)
{
    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(dual_fisheye_msg, "rgb8");
        int rows_size = cv_ptr->image.rows;
        int cols_size = cv_ptr->image.cols / 2;

        if (!maps_initialized_ || params_changed_ ||
            rows_size != img_height_ || cols_size != img_width_) {
            initMapping(rows_size, cols_size);
            params_changed_ = false;
        }

        auto start_time = now();

        // cv::remap seamlessly handles the optimized CV_16SC2 maps we generated
        cv::remap(cv_ptr->image, equirect_img, full_map_x_, full_map_y_, cv::INTER_LINEAR);

        cv_bridge::CvImage out_msg;
        out_msg.header = dual_fisheye_msg->header;
        out_msg.encoding = "rgb8";
        out_msg.image = equirect_img;
        equirect_pub_->publish(*out_msg.toImageMsg());

        auto process_time = (now() - start_time).seconds();
        RCLCPP_DEBUG(get_logger(), "Processing time: %.3f seconds", process_time);

    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error processing images: %s", e.what());
    }
}

rcl_interfaces::msg::SetParametersResult EquirectangularNode::parametersCallback(
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

    auto node = std::make_shared<EquirectangularNode>();

    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "Exception during spin: %s", e.what());
    }

    rclcpp::shutdown();
    return 0;
}