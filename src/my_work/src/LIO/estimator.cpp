#include "estimator.h"
#include "Eigen/src/Core/Matrix.h"
#include "common_lib.h"
#include "fast_lio.hpp"
#include <iomanip>
#include <iostream>

Estimator::Estimator(){
    ROS_INFO("init begins");
    clearState();
}
void Estimator::clearState(){
    for (int i = 0; i < WINDOW_SIZE + 1; i++){
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++){
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : m_all_image_frame){
        if (it.second.pre_integration != nullptr){
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    m_all_image_frame.clear();
    td = TD;


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;

    tmp_pre_integration = nullptr;
}

void Estimator::setParameter(){
  // 用于判断状态是否发散
    // nh.param<double>("position_std_threshold", position_std_threshold, 8.0);
}

void stateAugmentation(const double &time, StateServer &state_server){
  // 1. 取出当前更新好的imu状态量
  // 1.1 取出imu与camera的外参 R_w_i 表示从w系到I系
  const Matrix3d &R_i_c = READ_RIC[0].transpose(); //IMU到相机的旋转矩阵 //!注意：READ_RIC[0]中存储的是从相机到IMU的旋转矩阵，因此需要转置
  const Vector3d &t_c_i = READ_TIC[0]; //IMU到相机的平移向量 //!同理

  // 1.2 取出imu旋转平移，按照外参，将这个时刻cam0的位姿算出来
  Matrix3d R_w_i = state_server.imu_state.rot_end; //w_i表示IMU到世界系的旋转矩阵
  Matrix3d R_w_c = R_i_c * R_w_i;
  Vector3d t_c_w = state_server.imu_state.pos_end +
                      R_w_i.transpose() * t_c_i; //TODO 这里state_server.imu_state.pos_end是哪个坐标系之间的转换

  // 2. 注册新的相机状态到状态库中
  // 嗯。。。说人话就是找个记录的，不然咋更新
  state_server.cam_states[state_server.imu_state.id] =
      CAMState(state_server.imu_state.id);
  CAMState &cam_state = state_server.cam_states[state_server.imu_state.id];


  // 严格上讲这个时间不对，但是几乎没影响
  cam_state.time = time;
  cam_state.orientation = R_w_c; 
  cam_state.position = t_c_w;

  // 记录第一次被估计的数据，不能被改变，因为改变了就破坏了之前的0空间
  cam_state.orientation_null = cam_state.orientation;
  cam_state.position_null = cam_state.position;

  // 3. 这个雅可比可以认为是cam0位姿相对于imu的状态量的求偏导
  // 此时我们首先要知道相机位姿是 Rcw  twc
  // Rcw = Rci * Riw   twc = twi + Rwi * tic
  //原版MSCKF的IMU误差状态量：Q bg V ba P R_I_C P_I_C
  //Fast-LIO的IMU误差状态量：Q V P bg ba
  Matrix<double, 6, 21> J = Matrix<double, 6, 21>::Zero();
  // Rcw对Riw的左扰动导数
  J.block<3, 3>(0, 0) = R_i_c;
  // Rcw对Rci的左扰动导数
  J.block<3, 3>(0, 15) = Matrix3d::Identity(); //* Rcw对旋转外参的导数 //TODO 为什么是单位阵？
  J.block<3, 3>(3, 0) = skewSymmetric(R_w_i.transpose() * t_c_i);
  // twc对twi的左扰动导数
  J.block<3, 3>(3, 12) = Matrix3d::Identity(); 
  // twc对tic的左扰动导数
  J.block<3, 3>(3, 18) = R_w_i.transpose(); //* Pcw对平移外参的导数
  // 4. 增广协方差矩阵
  // 4.1 扩展矩阵大小 conservativeResize函数不改变原矩阵对应位置的数值

  state_server.state_cov.conservativeResize(21, 21); //原本的协方差矩阵没有考虑外参，是18×18的，现在要扩展到21×21
  size_t old_rows = state_server.state_cov.rows();
  size_t old_cols = state_server.state_cov.cols();
  state_server.state_cov.conservativeResize(old_rows + 6, old_cols + 6);
  // std::cout << "state_server.state_cov.rows() = " << state_server.state_cov.size() << std::endl;
  // Rename some matrix blocks for convenience.
  // imu的协方差矩阵 21×21
  const Matrix<double, 21, 21> &P11 =
      state_server.state_cov.block<21, 21>(0, 0);

  // imu相对于各个相机状态量的协方差矩阵（不包括最新的）
  const MatrixXd &P12 =
      state_server.state_cov.block(0, 21, 21,  old_cols - 21);

  // Fill in the augmented state covariance.
  // 4.2 计算协方差矩阵
  // 左下角
  state_server.state_cov.block(old_rows, 0, 6, old_cols) << J * P11, J * P12;

  // 右上角
  state_server.state_cov.block(0, old_cols, old_rows, 6) =
      state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();

  // 右下角，关于相机部分的J都是0所以省略了
  state_server.state_cov.block<6, 6>(old_rows, old_cols) =
      J * P11 * J.transpose();

  // Fix the covariance to be symmetric
  // 强制对称
  MatrixXd state_cov_fixed = (state_server.state_cov +
                              state_server.state_cov.transpose()) /
                              2.0;
  state_server.state_cov = state_cov_fixed;
}

// void Estimator::addFeatureObservations(const sensor_msgs::PointCloudConstPtr &msg){

// }
// void Estimator::removeLostFeatures(){

// }
// void Estimator::pruneCamStateBuffer(){

// }