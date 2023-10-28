#pragma once
#include "fast_lio.hpp"
#include "parameters.h"

#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include "feature.hpp"

#include <opencv2/core/eigen.hpp>
#include "common_lib.h"

// #include "Eigen/SPQRSupport"
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

    // internal
    void clearState();

    // msckf
  // Measurement update
  void stateAugmentation(const double &time);
  void addFeatureObservations(const sensor_msgs::PointCloudConstPtr &o_msg);
  void removeLostFeatures();
  void pruneCamStateBuffer();
  void featureJacobian(const FeatureIDType &feature_id, const std::vector<StateIDType> &cam_state_ids, Eigen::MatrixXd &H_x, Eigen::VectorXd &r);
  void measurementUpdate(const Eigen::MatrixXd &H, const Eigen::VectorXd &r);
  bool gatingTest(const Eigen::MatrixXd &H, const Eigen::VectorXd &r, const int &dof);
  void measurementJacobian(const StateIDType &cam_state_id,
      const FeatureIDType &feature_id,
      Eigen::Matrix<double, 2, 6> &H_x,
      Eigen::Matrix<double, 2, 3> &H_f,
      Eigen::Vector2d &r);

  void findRedundantCamStates(vector<StateIDType> &rm_cam_state_ids);

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };
    
    SolverFlag solver_flag;

    //MSCKF config
    double td; //cam与lio时间差
    StatesGroup imu_state; //imu状态
    CamStateServer cam_states; //相机状态
    //特征管理类
    MapServer map_server; //存储特征点的map key是id，value是Feature

    MatrixXd state_cov; //动态大小的矩阵

// 用于判断状态是否发散
  int max_cam_state_size;
  //旋转角度的阈值
  double rotation_threshold;
  //平移距离的阈值
  double translation_threshold;
  //跟踪率的阈值
  double tracking_rate_threshold;
  //跟踪率
  double tracking_rate;

};

