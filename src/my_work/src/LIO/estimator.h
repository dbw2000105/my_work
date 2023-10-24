#pragma once
#include "fast_lio.hpp"
#include "parameters.h"

#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include "common_lib.h"

/*
    Estimator 状态估计器
    IMU预积分，图像IMU融合的初始化和状态估计，重定位
*/
class Estimator
{
public:
    Estimator();
    Fast_lio *m_fast_lio_instance = nullptr;
    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header, const StatesGroup &lio_state = StatesGroup());

    void solve_image_pose(const std_msgs::Header &header);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization_LM();
    void vector2double();
    void double2vector();
    bool failureDetection();
    int refine_vio_system(eigen_q q_extrinsic, vec_3 t_extrinsic);

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };
    
    SolverFlag solver_flag;
    // MarginalizationFlag marginalization_flag;
    Vector3d m_gravity;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    // 窗口中的[P,V,R,Ba,Bg]
    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    StatesGroup m_lio_state_prediction_vec[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    // 窗口中的dt,a,v
    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    // FeatureManager f_manager;
    // MotionEstimator m_estimator;
    // InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;

    double m_para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double m_para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double m_para_Feature[NUM_OF_F][SIZE_FEATURE];
    double m_para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double m_para_Retrive_Pose[SIZE_POSE];
    double m_para_Td[1][1];
    double m_para_Tr[1][1];

    int loop_window_index;
    struct ImageFrame{
      ImageFrame() = default;
      ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t):t{_t},is_key_frame{false}
        {
            points = _points;
        };
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;
        double t{};
        Matrix3d R;
        Vector3d T;
        bool is_key_frame{};
        StatesGroup m_state_prior;
        IntegrationBase *pre_integration{};
      
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    vector<double *> m_last_marginalization_parameter_blocks;
    // vio_marginalization *m_vio_margin_ptr = nullptr;

    // kay是时间戳，val是图像帧
    // 图像帧中保存了图像帧的特征点、时间戳、位姿Rt，预积分对象pre_integration，是否是关键帧。
    map<double, ImageFrame> m_all_image_frame;
    IntegrationBase *tmp_pre_integration;

    // relocalization variable
    // 重定位所需的变量
    // bool relocalization_info;
    // double relo_frame_stamp;
    // double relo_frame_index;
    // int relo_frame_local_index;
    // vector<Vector3d> match_points;
    // double relo_Pose[SIZE_POSE];
    // Matrix3d drift_correct_r;
    // Vector3d drift_correct_t;
    // Vector3d prev_relo_t;
    // Matrix3d prev_relo_r;
    // Vector3d relo_relative_t;
    // Quaterniond relo_relative_q;
    // double relo_relative_yaw;
};

struct StateServer
    {
        StatesGroup imu_state;
        // 别看他长，其实就是一个map类
        // key是 StateIDType 由 long long int typedef而来，把它当作int看就行
        // value是CAMState
        CamStateServer cam_states;

        // State covariance matrix
        Eigen::MatrixXd state_cov; //动态大小的矩阵
        Eigen::Matrix<double, 12, 12> continuous_noise_cov;
    };
//msckf
// Measurement update
void stateAugmentation(const double &time, StateServer &state_server);
void addFeatureObservations(const sensor_msgs::PointCloudConstPtr &msg);
void removeLostFeatures();
void pruneCamStateBuffer();