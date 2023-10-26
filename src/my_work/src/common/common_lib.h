#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <opencv/cv.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <my_work/States.h>
#include <my_work/Pose6D.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <nav_msgs/Odometry.h>
#include <rosbag/bag.h>

#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include "tools/tools_color_printf.hpp"
#include "tools/tools_eigen.hpp"
#include <queue>
#include <deque>
#include "tools/r2live_sophus/se3.hpp"
#include "tools/r2live_sophus/so3.hpp"
// #define DEBUG_PRINT
#define USE_ikdtree
#define ESTIMATE_GRAVITY 1
// #define USE_FOV_Checker

#define printf_line std::cout << __FILE__ << " " << __LINE__ << std::endl;

#define PI_M (3.14159265358)
#define G_m_s2 (9.805)     // Gravaty const in GuangDong/China
#define DIM_OF_STATES (18) // Dimension of states (Let Dim(SO(3)) = 3)
#define DIM_OF_PROC_N (12) // Dimension of process noise (Let Dim(SO(3)) = 3)
#define CUBE_LEN (6.0)
#define LIDAR_SP_LEN (2)
#define INIT_COV (0.0001)

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define CONSTRAIN(v, min, max) ((v > min) ? ((v < max) ? v : max) : min)
#define ARRAY_FROM_EIGEN(mat) mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat) std::vector<decltype(mat)::Scalar>(mat.data(), mat.data() + mat.rows() * mat.cols())

#define DEBUG_FILE_DIR(name) (std::string(std::string(ROOT_DIR) + "Log/" + name))
// using vins_estimator = fast_lio;

typedef my_work::Pose6D Pose6D;
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

static const Eigen::Matrix3d Eye3d(Eigen::Matrix3d::Identity());
static const Eigen::Matrix3f Eye3f(Eigen::Matrix3f::Identity());
static const Eigen::Vector3d Zero3d(0, 0, 0);
static const Eigen::Vector3f Zero3f(0, 0, 0);
// Eigen::Vector3d Lidar_offset_to_IMU(0.05512, 0.02226, 0.0297); // Horizon
static const Eigen::Vector3d Lidar_offset_to_IMU(0.04165, 0.02326, -0.0284); // Avia

template <typename T>
T get_ros_parameter(ros::NodeHandle &nh, const std::string parameter_name, T &parameter, T default_val)
{
    nh.param<T>(parameter_name.c_str(), parameter, default_val);
    // ENABLE_SCREEN_PRINTF;
    cout << "[Ros_parameter]: " << parameter_name << " ==> " << parameter << std::endl;
    return parameter;
}


template <typename T = double>
inline Eigen::Matrix<T, 3, 3> vec_to_hat(Eigen::Matrix<T, 3, 1> &omega)
{
    Eigen::Matrix<T, 3, 3> res_mat_33;
    res_mat_33.setZero();
    res_mat_33(0, 1) = -omega(2);
    res_mat_33(1, 0) = omega(2);
    res_mat_33(0, 2) = omega(1);
    res_mat_33(2, 0) = -omega(1);
    res_mat_33(1, 2) = -omega(0);
    res_mat_33(2, 1) = omega(0);
    return res_mat_33;
}

template < typename T = double > 
T cot(const T theta)
{
    return 1.0 / std::tan(theta);
}

template < typename T = double >
inline Eigen::Matrix< T, 3, 3 > right_jacobian_of_rotion_matrix(const Eigen::Matrix< T, 3, 1 > & omega)
{
    //Barfoot, Timothy D, State estimation for robotics. Page 232-237
    Eigen::Matrix< T, 3, 3>   res_mat_33;

    T theta = omega.norm();
    if(std::isnan(theta) || theta == 0)
        return Eigen::Matrix< T, 3, 3>::Identity();
    Eigen::Matrix< T, 3, 1 > a = omega/ theta;
    Eigen::Matrix< T, 3, 3 > hat_a = vec_to_hat(a);
    res_mat_33 = sin(theta)/theta * Eigen::Matrix< T, 3, 3 >::Identity()
                    + (1 - (sin(theta)/theta))*a*a.transpose() 
                    + ((1 - cos(theta))/theta)*hat_a;
    // cout << "Omega: " << omega.transpose() << endl;
    // cout << "Res_mat_33:\r\n"  <<res_mat_33 << endl;
    return res_mat_33;
}

template < typename T = double >
Eigen::Matrix< T, 3, 3 > inverse_right_jacobian_of_rotion_matrix(const Eigen::Matrix< T, 3, 1> & omega)
{
    //Barfoot, Timothy D, State estimation for robotics. Page 232-237
    Eigen::Matrix< T, 3, 3>   res_mat_33;

    T theta = omega.norm();
    if(std::isnan(theta) || theta == 0)
        return Eigen::Matrix< T, 3, 3>::Identity();
    Eigen::Matrix< T, 3, 1 > a = omega/ theta;
    Eigen::Matrix< T, 3, 3 > hat_a = vec_to_hat(a);
    res_mat_33 = (theta / 2) * (cot(theta / 2)) * Eigen::Matrix<T, 3, 3>::Identity() 
                + (1 - (theta / 2) * (cot(theta / 2))) * a * a.transpose() 
                + (theta / 2) * hat_a;
    // cout << "Omega: " << omega.transpose() << endl;
    // cout << "Res_mat_33:\r\n"  <<res_mat_33 << endl;
    return res_mat_33;
}


struct Camera_Lidar_queue
{
    double m_first_imu_time = -3e8;
    double m_sliding_window_tim = 10000;
    double m_last_imu_time = -3e8;
    double m_last_visual_time = -3e8;
    double m_visual_init_time = 3e88;
    double m_lidar_drag_cam_tim = 5.0;
    double m_if_lidar_start_first = 1;
    double m_camera_imu_td = 0;

    int m_if_acc_mul_G = 0;

    int m_if_have_lidar_data = 0;
    int m_if_have_camera_data = 0;
    int m_if_lidar_can_start = 1;
    Eigen::Vector3d g_noise_cov_acc;
    Eigen::Vector3d g_noise_cov_gyro;

    std::string m_bag_file_name;
    int m_if_write_res_to_bag = 0;
    int m_if_dump_log = 1;
    rosbag::Bag m_bag_for_record;

    std::deque<sensor_msgs::PointCloud2::ConstPtr> *m_liar_frame_buf = nullptr; //雷达数据缓存

    void init_rosbag_for_recording()
    {
        if (m_if_write_res_to_bag)
        {
            cout << ANSI_COLOR_YELLOW_BG << "Record result to " << m_bag_file_name << ANSI_COLOR_RESET << endl;
            m_bag_for_record.open(m_bag_file_name.c_str(), rosbag::bagmode::Write);
        }
    }

    Camera_Lidar_queue()
    {
        m_if_have_lidar_data = 0;
        m_if_have_camera_data = 0;
    };
    ~Camera_Lidar_queue(){};

    void imu_in(const double &in_time)
    {
        if (m_first_imu_time < 0) //初始化IMU时间
        {
            m_first_imu_time = in_time;
        }

        m_last_imu_time = std::max(in_time, m_last_imu_time);
        // m_last_imu_time = in_time;
    }

    int lidar_in(const double &in_time)
    {
        // cout << "LIDAR in " << endl;
        if (m_if_have_lidar_data == 0)
        {
            m_if_have_lidar_data = 1;
            cout << ANSI_COLOR_BLUE_BOLD << "Have LiDAR data" << endl;
        }
        if (in_time < m_last_imu_time - m_sliding_window_tim)
        {
            std::cout << ANSI_COLOR_RED_BOLD << "LiDAR incoming frame too old, need to be drop!!!" << ANSI_COLOR_RESET << std::endl;
            // TODO: Drop LiDAR frame
        }
        return 1;
    }

    int camera_in(const double &in_time)
    {
        if (in_time < m_last_imu_time - m_sliding_window_tim)
        {
            std::cout << ANSI_COLOR_RED_BOLD << "Camera incoming frame too old, need to be drop!!!" << ANSI_COLOR_RESET << std::endl;
            // TODO: Drop camera frame
        }
        return 1;
    }

    double get_lidar_front_time()
    {
        if (m_liar_frame_buf != nullptr && m_liar_frame_buf->size())
        {
            return m_liar_frame_buf->front()->header.stamp.toSec() + 0.1;
        }
        else
        {
            return -3e8;
        }
    }

    double get_camera_front_time()
    {
        return m_last_visual_time + m_camera_imu_td;
        // if (m_camera_frame_buf != nullptr && m_camera_frame_buf->size())
        // {
        //     double min_time;

        //     // return std::min(m_camera_frame_buf->front()->second()->header.stamp.toSec());
        //     return (m_camera_frame_buf->front().second->header.stamp.toSec());
        // }
        // else
        // {
        //     return -3e8;
        // }
    }

    bool if_camera_can_process()
    {
        m_if_have_camera_data = 1;
        double cam_last_time = get_camera_front_time();
        double lidar_last_time = get_lidar_front_time();
        if (m_if_have_lidar_data != 1)
        {
            // scope_color(ANSI_COLOR_YELLOW_BOLD);
            // cout << "Camera can update , no LiDAR data" << endl;
            // printf_line;
            return true;
        }

        if (cam_last_time < 0 || lidar_last_time < 0)
        {
            // printf_line;
            return false;
        }

        // if (get_camera_front_time() > m_last_imu_time - m_sliding_window_tim)
        // {
        //     return false;
        // }

        if (lidar_last_time <= cam_last_time)
        {
            // LiDAR data need process first.
            return false;
        }
        else
        {
            // scope_color(ANSI_COLOR_YELLOW_BOLD);
            // cout << "Camera can update, " << get_lidar_front_time() - m_first_imu_time << " | " << get_camera_front_time() - m_first_imu_time << endl;
            return true;
        }
        return false;
    }

    bool if_lidar_can_process()
    {
        // m_if_have_lidar_data = 1;
        double cam_last_time = get_camera_front_time();
        double lidar_last_time = get_lidar_front_time();

        if (m_if_have_camera_data == 0)
        {
            // scope_color(ANSI_COLOR_BLUE_BOLD);
            // cout << "LiDAR can update , no camera data" << endl;
            // printf_line;
            return true;
        }

        if (cam_last_time < 0 || lidar_last_time < 0)
        {
            // printf_line;
            // cout << "Cam_tim = " << cam_last_time << ", lidar_last_time = " << lidar_last_time << endl; 
            return false;
        }

        // if (get_lidar_front_time() > m_last_imu_time - m_sliding_window_tim)
        // {
        //     return false;
        // }

        if (lidar_last_time > cam_last_time)
        {
            // Camera data need process first.
            ;
            return false;
        }
        else
        {
            // scope_color(ANSI_COLOR_BLUE_BOLD);
            // cout << "LiDAR can update, " << get_lidar_front_time() - m_first_imu_time << " | " << get_camera_front_time() - m_first_imu_time << endl;
            // printf_line;
            return true;
        }
        return false;
    }
};

struct MeasureGroup // Lidar data and imu dates for the curent process
{
    MeasureGroup()
    {
        this->lidar.reset(new PointCloudXYZI());
    };
    double lidar_beg_time;
    double lidar_end_time;
    PointCloudXYZI::Ptr lidar;
    std::deque<sensor_msgs::Imu::ConstPtr> imu;
};

/*
 * @brief IMUState State for IMU
 */
struct IMUState
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef long long int StateIDType;

    // An unique identifier for the IMU state.
    StateIDType id;

    // id for next IMU state
    static StateIDType next_id;

    // Time when the state is recorded
    double time;

    // Orientation
    // Take a vector from the world frame to
    // the IMU (body) frame.
    // Rbw
    Eigen::Vector4d orientation;

    // Position of the IMU (body) frame
    // in the world frame.
    // twb
    Eigen::Vector3d position;

    // Velocity of the IMU (body) frame
    // in the world frame.
    Eigen::Vector3d velocity;

    // Bias for measured angular velocity
    // and acceleration.
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d acc_bias;

    // Transformation between the IMU and the
    Eigen::Matrix3d R_imu_cam0;
    Eigen::Vector3d t_cam0_imu;

    // These three variables should have the same physical
    // interpretation with `orientation`, `position`, and
    // `velocity`. There three variables are used to modify
    // the transition matrices to make the observability matrix
    // have proper null space.
    // 用于使可观测性矩阵具有适当的零空间
    Eigen::Vector4d orientation_null;
    Eigen::Vector3d position_null;
    Eigen::Vector3d velocity_null;

    // Process noise
    static double gyro_noise;
    static double acc_noise;
    static double gyro_bias_noise;
    static double acc_bias_noise;

    // Gravity vector in the world frame
    static Eigen::Vector3d gravity;

    // Transformation offset from the IMU frame to
    // the body frame. The transformation takes a
    // vector from the IMU frame to the body frame.
    // The z axis of the body frame should point upwards.
    // Normally, this transform should be identity.
    static Eigen::Isometry3d T_imu_body;

    IMUState() 
        : id(0), time(0),
        orientation(Eigen::Vector4d(0, 0, 0, 1)),
        position(Eigen::Vector3d::Zero()),
        velocity(Eigen::Vector3d::Zero()),
        gyro_bias(Eigen::Vector3d::Zero()),
        acc_bias(Eigen::Vector3d::Zero()),
        orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
        position_null(Eigen::Vector3d::Zero()),
        velocity_null(Eigen::Vector3d::Zero()) {}

    IMUState(const StateIDType &new_id)
        : id(new_id), time(0),
        orientation(Eigen::Vector4d(0, 0, 0, 1)),
        position(Eigen::Vector3d::Zero()),
        velocity(Eigen::Vector3d::Zero()),
        gyro_bias(Eigen::Vector3d::Zero()),
        acc_bias(Eigen::Vector3d::Zero()),
        orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
        position_null(Eigen::Vector3d::Zero()),
        velocity_null(Eigen::Vector3d::Zero()) {}
};

// 别看他长，其实就是一个map类
// key是 StateIDType 由 long long int typedef而来，把它当作int看就行
// value是CAMState
typedef long long int StateIDType;
/*
    * @brief CAMState Stored camera state in order to
    *    form measurement model.
    */
struct CAMState
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // An unique identifier for the CAM state.
    StateIDType id;

    // Time when the state is recorded
    double time;

    // Orientation
    // Take a vector from the world frame to the camera frame.
    // Eigen::Vector4d orientation;
    Eigen::Matrix3d orientation;

    // Position of the camera frame in the world frame.
    Eigen::Vector3d position;

    // These two variables should have the same physical
    // interpretation with `orientation` and `position`.
    // There two variables are used to modify the measurement
    // Jacobian matrices to make the observability matrix
    // have proper null space.
    // 使可观测性矩阵具有适当的零空间的旋转平移
    Eigen::Matrix3d orientation_null;
    Eigen::Vector3d position_null;

    // Takes a vector from the cam0 frame to the cam1 frame.
    // 两个相机间的外参
    // static Eigen::Isometry3d T_cam0_cam1;

    CAMState() 
    : id(0), time(0),
    // orientation(Eigen::Vector4d(0, 0, 0, 1)),
    orientation(Eigen::Matrix3d::Identity()), //TODO 这里目前用的旋转矩阵，如果后面有问题，可能需要改回四元数的形式
    position(Eigen::Vector3d::Zero()),
    orientation_null(Eigen::Matrix3d::Identity()),
    // orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
    position_null(Eigen::Vector3d(0, 0, 0))
    {

    }

    explicit CAMState(const StateIDType &new_id)
    : id(new_id), time(0),
    orientation(Eigen::Matrix3d::Identity()), 
    position(Eigen::Vector3d::Zero()),
    orientation_null(Eigen::Matrix3d::Identity()),
    position_null(Eigen::Vector3d::Zero())
    {

    }
};
typedef std::map<StateIDType, CAMState,
    std::less<>, Eigen::aligned_allocator< std::pair<const StateIDType, CAMState>>>
    CamStateServer;

struct StatesGroup
{
    StatesGroup()
    {
        this->rot_end = Eigen::Matrix3d::Identity();
        this->pos_end = Zero3d;
        this->vel_end = Zero3d;
        this->bias_g = Zero3d;
        this->bias_a = Zero3d;
        this->gravity = Eigen::Vector3d(0.0, 0.0, 9.805);
        this->cov = Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES>::Identity() * INIT_COV;
        this->last_update_time = 0;
        this->id = 0;
    };
    ~StatesGroup()
    = default;

    StatesGroup(const StatesGroup &b)
    {
        this->rot_end = b.rot_end;
        this->pos_end = b.pos_end;
        this->vel_end = b.vel_end;
        this->bias_g = b.bias_g;
        this->bias_a = b.bias_a;
        this->gravity = b.gravity;
        this->cov = b.cov;
        this->last_update_time = b.last_update_time;
        this->id = b.id;
    };

    StatesGroup &operator=(const StatesGroup &b)
    {
        this->rot_end = b.rot_end;
        this->pos_end = b.pos_end;
        this->vel_end = b.vel_end;
        this->bias_g = b.bias_g;
        this->bias_a = b.bias_a;
        this->id = b.id;
#if ESTIMATE_GRAVITY
        this->gravity = b.gravity;
#else
        this->gravity = Eigen::Vector3d(0.0, 0.0, 9.805);
#endif
        this->cov = b.cov;
        this->last_update_time = b.last_update_time;
        return *this;
    };

    StatesGroup operator+(const Eigen::Matrix<double, DIM_OF_STATES, 1> &state_add)
    {
        StatesGroup a;
        a.rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
        a.pos_end = this->pos_end + state_add.block<3, 1>(3, 0);
        a.vel_end = this->vel_end + state_add.block<3, 1>(6, 0);
        a.bias_g = this->bias_g + state_add.block<3, 1>(9, 0);
        a.bias_a = this->bias_a + state_add.block<3, 1>(12, 0);
#if ESTIMATE_GRAVITY
        a.gravity = this->gravity + state_add.block<3, 1>(15, 0);
#endif

        a.cov = this->cov;
        a.last_update_time = this->last_update_time;
        return a;
    };

    StatesGroup &operator+=(const Eigen::Matrix<double, DIM_OF_STATES, 1> &state_add)
    {
        this->rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
        this->pos_end += state_add.block<3, 1>(3, 0);
        this->vel_end += state_add.block<3, 1>(6, 0);
        this->bias_g += state_add.block<3, 1>(9, 0);
        this->bias_a += state_add.block<3, 1>(12, 0);
#if ESTIMATE_GRAVITY
        this->gravity += state_add.block<3, 1>(15, 0);
#endif
        return *this;
    };

    Eigen::Matrix<double, DIM_OF_STATES, 1> operator-(const StatesGroup &b)
    {
        Eigen::Matrix<double, DIM_OF_STATES, 1> a;
        Eigen::Matrix3d rotd(b.rot_end.transpose() * this->rot_end);
        a.block<3, 1>(0, 0) = Log(rotd);
        a.block<3, 1>(3, 0) = this->pos_end - b.pos_end;
        a.block<3, 1>(6, 0) = this->vel_end - b.vel_end;
        a.block<3, 1>(9, 0) = this->bias_g - b.bias_g;
        a.block<3, 1>(12, 0) = this->bias_a - b.bias_a;
        a.block<3, 1>(15, 0) = this->gravity - b.gravity;
        return a;
    };

    static void display(const StatesGroup &state, std::string str = std::string("State: "))
    {
        vec_3 angle_axis = Log(state.rot_end) * 57.3;
        printf("%s |", str.c_str());
        printf("[%.5f] | ", state.last_update_time);
        printf("(%.3f, %.3f, %.3f) | ", angle_axis(0), angle_axis(1), angle_axis(2));
        printf("(%.3f, %.3f, %.3f) | ", state.pos_end(0), state.pos_end(1), state.pos_end(2));
        printf("(%.3f, %.3f, %.3f) | ", state.vel_end(0), state.vel_end(1), state.vel_end(2));
        printf("(%.3f, %.3f, %.3f) | ", state.bias_g(0), state.bias_g(1), state.bias_g(2));
        printf("(%.3f, %.3f, %.3f) \r\n", state.bias_a(0), state.bias_a(1), state.bias_a(2));
    }

    Eigen::Matrix3d rot_end;                                 // the estimated attitude (rotation matrix) at the end lidar point
    Eigen::Vector3d pos_end;                                 // the estimated position at the end lidar point (world frame)
    Eigen::Vector3d vel_end;                                 // the estimated velocity at the end lidar point (world frame)
    Eigen::Vector3d bias_g;                                  // gyroscope bias
    Eigen::Vector3d bias_a;                                  // accelerator bias
    Eigen::Vector3d gravity;                                 // the estimated gravity acceleration
    Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> cov; // states covariance
    double last_update_time = 0;
    long long int id;
      // Transformation between the IMU and the
    Eigen::Matrix3d R_imu_cam0;
    Eigen::Vector3d t_cam0_imu;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename T>
T rad2deg(T radians)
{
    return radians * 180.0 / PI_M;
}

template <typename T>
T deg2rad(T degrees)
{
    return degrees * PI_M / 180.0;
}

template <typename T>
auto set_pose6d(const double t, const Eigen::Matrix<T, 3, 1> &a, const Eigen::Matrix<T, 3, 1> &g,
                const Eigen::Matrix<T, 3, 1> &v, const Eigen::Matrix<T, 3, 1> &p, const Eigen::Matrix<T, 3, 3> &R)
{
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++)
    {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++)
            rot_kp.rot[i * 3 + j] = R(i, j);
    }
    // Eigen::Map<Eigen::Matrix3d>(rot_kp.rot, 3,3) = R;
    return std::move(rot_kp);
}

/**
 *  @brief 反对称矩阵
 *  @note Performs the operation:
 *  w   ->  [  0 -w3  w2]
 *          [ w3   0 -w1]
 *          [-w2  w1   0]
 */
inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &w)
{
    Eigen::Matrix3d w_hat;
    w_hat(0, 0) = 0;
    w_hat(0, 1) = -w(2);
    w_hat(0, 2) = w(1);
    w_hat(1, 0) = w(2);
    w_hat(1, 1) = 0;
    w_hat(1, 2) = -w(0);
    w_hat(2, 0) = -w(1);
    w_hat(2, 1) = w(0);
    w_hat(2, 2) = 0;
    return w_hat;
}

/**
 * @brief 标准化四元数
 */
inline void quaternionNormalize(Eigen::Vector4d &q)
{
    double norm = q.norm();
    q = q / norm;
}

/**
 * @brief 四元数转旋转矩阵 jpl
 * Convert a quaternion to the corresponding rotation matrix
 * @note Pay attention to the convention used. The function follows the
 *    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for Quaternion Algebra", Equation (62).
 *
 *    The input quaternion should be in the form
 *      [q1, q2, q3, q4(scalar)]^T
 */
inline Eigen::Matrix3d quaternionToRotation(
    const Eigen::Vector4d &q)
{
    const Eigen::Vector3d &q_vec = q.block(0, 0, 3, 1);
    const double &q4 = q(3);
    Eigen::Matrix3d R =
        (2 * q4 * q4 - 1) * Eigen::Matrix3d::Identity() -
        2 * q4 * skewSymmetric(q_vec) +
        2 * q_vec * q_vec.transpose();
    // TODO: Is it necessary to use the approximation equation
    //     (Equation (70)) when the rotation angle is small?
    return R;
}

/**
 * @brief 旋转矩阵转四元数 没在论文里找到，这里不用看，直接用！
 * Convert a rotation matrix to a quaternion.
 * @note Pay attention to the convention used. The function follows the
 *    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for Quaternion Algebra", Equation (78).
 *
 *    The input quaternion should be in the form
 *      [q1, q2, q3, q4(scalar)]^T
 */
inline Eigen::Vector4d rotationToQuaternion(
    const Eigen::Matrix3d &R)
{
    Eigen::Vector4d score;
    score(0) = R(0, 0);
    score(1) = R(1, 1);
    score(2) = R(2, 2);
    score(3) = R.trace();

    int max_row = 0, max_col = 0;
    score.maxCoeff(&max_row, &max_col);

    Eigen::Vector4d q = Eigen::Vector4d::Zero();
    if (max_row == 0)
    {
        q(0) = std::sqrt(1 + 2 * R(0, 0) - R.trace()) / 2.0;
        q(1) = (R(0, 1) + R(1, 0)) / (4 * q(0));
        q(2) = (R(0, 2) + R(2, 0)) / (4 * q(0));
        q(3) = (R(1, 2) - R(2, 1)) / (4 * q(0));
    }
    else if (max_row == 1)
    {
        q(1) = std::sqrt(1 + 2 * R(1, 1) - R.trace()) / 2.0;
        q(0) = (R(0, 1) + R(1, 0)) / (4 * q(1));
        q(2) = (R(1, 2) + R(2, 1)) / (4 * q(1));
        q(3) = (R(2, 0) - R(0, 2)) / (4 * q(1));
    }
    else if (max_row == 2)
    {
        q(2) = std::sqrt(1 + 2 * R(2, 2) - R.trace()) / 2.0;
        q(0) = (R(0, 2) + R(2, 0)) / (4 * q(2));
        q(1) = (R(1, 2) + R(2, 1)) / (4 * q(2));
        q(3) = (R(0, 1) - R(1, 0)) / (4 * q(2));
    }
    else
    {
        q(3) = std::sqrt(1 + R.trace()) / 2.0;
        q(0) = (R(1, 2) - R(2, 1)) / (4 * q(3));
        q(1) = (R(2, 0) - R(0, 2)) / (4 * q(3));
        q(2) = (R(0, 1) - R(1, 0)) / (4 * q(3));
    }

    if (q(3) < 0)
        q = -q;
    quaternionNormalize(q);
    return q;
}

/**
 * @brief 李代数转四元数，小量
 * Convert the vector part of a quaternion to a
 *    full quaternion.
 * @note This function is useful to convert delta quaternion
 *    which is usually a 3x1 vector to a full quaternion.
 *    For more details, check Section 3.2 "Kalman Filter Update" in
 *    "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for quaternion Algebra".
 */
inline Eigen::Vector4d smallAngleQuaternion(
    const Eigen::Vector3d &dtheta)
{
    // δq ~= (1/2δθ, 1)
    Eigen::Vector3d dq = dtheta / 2.0;
    Eigen::Vector4d q;
    double dq_square_norm = dq.squaredNorm();

    // 这么做就是为了符合四元数的定义
    // q(3)的平方+q.head<3>()的平方和的和是1
    if (dq_square_norm <= 1)
    {
        q.head<3>() = dq;
        q(3) = std::sqrt(1 - dq_square_norm);
    }
    else
    {
        q.head<3>() = dq;
        q(3) = 1;
        q = q / std::sqrt(1 + dq_square_norm);
    }

    return q;
}

/**
 * @brief Perform q1 * q2
 * Indirect Kalman Filter for 3D Attitude Estimation 公式8
 * jpl四元数乘法
 */
inline Eigen::Vector4d quaternionMultiplication(
    const Eigen::Vector4d &q1,
    const Eigen::Vector4d &q2)
{
    Eigen::Matrix4d L;
    L(0, 0) = q1(3);
    L(0, 1) = q1(2);
    L(0, 2) = -q1(1);
    L(0, 3) = q1(0);
    L(1, 0) = -q1(2);
    L(1, 1) = q1(3);
    L(1, 2) = q1(0);
    L(1, 3) = q1(1);
    L(2, 0) = q1(1);
    L(2, 1) = -q1(0);
    L(2, 2) = q1(3);
    L(2, 3) = q1(2);
    L(3, 0) = -q1(0);
    L(3, 1) = -q1(1);
    L(3, 2) = -q1(2);
    L(3, 3) = q1(3);

    Eigen::Vector4d q = L * q2;
    quaternionNormalize(q);
    return q;
}

#endif
