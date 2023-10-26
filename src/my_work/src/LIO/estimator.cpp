#include "estimator.h"
#include "Eigen/src/Core/Matrix.h"
#include <Eigen/SPQRSupport>
#include "common_lib.h"
MapServer map_server; //存储特征点的map key是id，value是Feature
double tracking_rate;
Feature::OptimizationConfig Feature::optimization_config;
double Feature::observation_noise = 0.01;

//最大相机个数(滑窗的大小)
int max_cam_state_size = 20;
//旋转角度的阈值
double rotation_threshold = 0.2618;
//平移距离的阈值
double translation_threshold = 0.4;
//跟踪率的阈值
double tracking_rate_threshold = 0.5;

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

  size_t old_rows = state_server.state_cov.rows();
  size_t old_cols = state_server.state_cov.cols();
  state_server.state_cov.conservativeResize(old_rows + 6, old_cols + 6);
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
  // std::cout << "state_server: " << state_server.state_cov.size() << std::endl;
  // std::cout << "IMU state id: " << state_server.imu_state.id << std::endl;

}

/**
 * @brief 添加特征点观测
 * @param  msg 前端发来的特征点信息，里面包含了时间，左右目上的角点及其id（严格意义上不能说是特征点）
 */
void addFeatureObservations(const sensor_msgs::PointCloudConstPtr &o_msg, StateServer &state_server){
  // 这是个long long int 嗯。。。。直接当作int理解吧
    // 这个id会在 batchImuProcessing 更新
    StateIDType state_id = state_server.imu_state.id;

    // 1. 获取当前窗口内特征点数量
    int curr_feature_num = map_server.size();
    int tracked_feature_num = 0;

    // Add new observations for existing features or new
    // features in the map server.
    // 2. 添加新来的点，做的花里胡哨，其实就是在现有的特征管理里面找，
    // id已存在说明是跟踪的点，在已有的上面更新
    // id不存在说明新来的点，那么就新添加一个
    for ( int i = 0; i < o_msg->points.size(); ++i){
      int v = o_msg->channels[0].values[i] + 0.5;
      int feature_id = v / NUM_OF_CAM;
      int camera_id = v % NUM_OF_CAM;
      double p_u = o_msg->channels[1].values[i];
      double p_v = o_msg->channels[2].values[i];
        if (map_server.find(feature_id) == map_server.end()){
            // This is a new feature.
            map_server[feature_id] = Feature(feature_id);
            map_server[feature_id].observations[state_id] =
                Vector2d(p_u, p_v);
        }
        else{
            // This is an old feature.
            map_server[feature_id].observations[state_id] =
                Vector2d(p_u,p_v);
            ++tracked_feature_num;
        }
    }
    // 这个东西计算了当前进来的跟踪的点中在总数里面的占比（进来的点有可能是新提的）
    tracking_rate =
        static_cast<double>(tracked_feature_num) /
        static_cast<double>(curr_feature_num);
    // std::cout << "tracking_rate: " << tracking_rate << std::endl;
}

/**
 * @brief 使用不再跟踪上的点进行状态更新
 * @param state_server 状态库
 */
void removeLostFeatures(StateServer &state_server){
  // Remove the features that lost track.
  // BTW, find the size the final Jacobian matrix and residual vector.
  int jacobian_row_size = 0;
  // FeatureIDType 这是个long long int 嗯。。。。直接当作int理解吧
  vector<FeatureIDType> invalid_feature_ids(0);  // 无效点，最后要删的
  vector<FeatureIDType> processed_feature_ids(0);  // 待参与更新的点，用完也被无情的删掉
      // 遍历所有特征管理里面的点，包括新进来的
    for (auto iter = map_server.begin();
            iter != map_server.end(); ++iter)
    {
        
        // Rename the feature to be checked.
        // 引用，改变feature相当于改变iter->second，类似于指针的效果
        auto &feature = iter->second;
        // Pass the features that are still being tracked.
        // 1. 这个点被当前状态观测到，说明这个点后面还有可能被跟踪
        // 跳过这些点
        if (feature.observations.find(state_server.imu_state.id) !=
            feature.observations.end()) //使用find()，如果找到了，返回指向该元素的迭代器；如果没找到，返回指向map尾部的迭代器
            continue;

        // 2. 跟踪小于3帧的点，认为是质量不高的点
        // 也好理解，三角化起码要两个观测，但是只有两个没有其他观测来验证
        if (feature.observations.size() < 3)
        {
            invalid_feature_ids.push_back(feature.id);
            continue;
        }

        // Check if the feature can be initialized if it
        // has not been.
        // 3. 如果这个特征没有被初始化，尝试去初始化
        // 初始化就是三角化
        
        if (!feature.is_initialized)
        {
            // 3.1 看看运动是否足够，没有足够视差或者平移小旋转多这种不符合三角化
            // 所以就不要这些点了
            if (!feature.checkMotion(state_server.cam_states))
            {
                invalid_feature_ids.push_back(feature.id);
                continue;
            }
            else
            {
                // 3.3 尝试三角化，失败也不要了
                if (!feature.initializePosition(state_server.cam_states))
                {
                    invalid_feature_ids.push_back(feature.id);
                    continue;
                }
            }
        }

        // 4. 到这里表示这个点能用于更新，所以准备下一步计算
        // 一个观测代表一帧，一帧有左右两个观测
        // 也就是算重投影误差时维度将会是4 * feature.observations.size()
        // 这里为什么减3下面会提到
        jacobian_row_size += 4 * feature.observations.size() - 3;
        // 接下来要参与优化的点加入到这个变量中
        processed_feature_ids.push_back(feature.id);
    }
    // Remove the features that do not have enough measurements.
    // 5. 删掉非法点
    for (const auto &feature_id : invalid_feature_ids)
        map_server.erase(feature_id);

    // Return if there is no lost feature to be processed.
    if (processed_feature_ids.size() == 0)
        return;

    // 准备好误差相对于状态量的雅可比
    MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
                                    21 + 6 * state_server.cam_states.size());
    VectorXd r = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    // Process the features which lose track.
    // 6. 处理特征点
    for (const auto &feature_id : processed_feature_ids)
    {
        auto &feature = map_server[feature_id];

        vector<StateIDType> cam_state_ids(0);
        for (const auto &measurement : feature.observations)
            cam_state_ids.push_back(measurement.first);

        MatrixXd H_xj;
        VectorXd r_j;
        // 6.1 计算雅可比，计算重投影误差
        featureJacobian(feature.id, cam_state_ids, H_xj, r_j, state_server);

        // 6.2 卡方检验，剔除错误点，并不是所有点都用
        if (gatingTest(H_xj, r_j, cam_state_ids.size() - 1, state_server))
        {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += H_xj.rows();
        }
        {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += H_xj.rows();
        }

        // Put an upper bound on the row size of measurement Jacobian,
        // which helps guarantee the executation time.
        // 限制最大更新量
        if (stack_cntr > 1500)
            break;
    }

    // resize成实际大小
    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);

    // Perform the measurement update step.
    // 7. 使用误差及雅可比更新状态
    measurementUpdate(H_x, r, state_server);

    // Remove all processed features from the map.
    // 8. 删除用完的点
    for (const auto &feature_id : processed_feature_ids)
        map_server.erase(feature_id);
}

void featureJacobian(const FeatureIDType &feature_id, const std::vector<StateIDType> &cam_state_ids, Eigen::MatrixXd &H_x, Eigen::VectorXd &r, StateServer &state_server){
    // 取出特征
    const auto &feature = map_server[feature_id];

    // Check how many camera states in the provided camera
    // id camera has actually seen this feature.
    // 1. 统计有效观测的相机状态，因为对应的个别状态有可能被滑走了
    vector<StateIDType> valid_cam_state_ids(0);
    for (const auto &cam_id : cam_state_ids)
    {
        if (feature.observations.find(cam_id) ==
            feature.observations.end())
            continue;

        valid_cam_state_ids.push_back(cam_id);
    }

    int jacobian_row_size = 0;
    // 行数等于4*观测数量，一个观测在双目上都有，所以是2*2
    // 此时还没有0空间投影
    jacobian_row_size = 4 * valid_cam_state_ids.size();

    // 误差相对于状态量的雅可比，没有约束列数，因为列数一直是最新的
    MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size,
                                    21 + state_server.cam_states.size() * 6);
    // 误差相对于三维点的雅可比
    MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
    // 误差
    VectorXd r_j = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    // 2. 计算每一个观测（同一帧左右目这里被叫成一个观测）的雅可比与误差
    for (const auto &cam_id : valid_cam_state_ids)
    {

        Matrix<double, 4, 6> H_xi = Matrix<double, 4, 6>::Zero();
        Matrix<double, 4, 3> H_fi = Matrix<double, 4, 3>::Zero();
        Vector4d r_i = Vector4d::Zero();
        // 2.1 计算一个左右目观测的雅可比
        measurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i, state_server);

        // 计算这个cam_id在整个矩阵的列数，因为要在大矩阵里面放
        auto cam_state_iter = state_server.cam_states.find(cam_id);
        int cam_state_cntr = std::distance(
            state_server.cam_states.begin(), cam_state_iter);

        // Stack the Jacobians.
        H_xj.block<4, 6>(stack_cntr, 21 + 6 * cam_state_cntr) = H_xi;
        H_fj.block<4, 3>(stack_cntr, 0) = H_fi;
        r_j.segment<4>(stack_cntr) = r_i;
        stack_cntr += 4;
    }

    // Project the residual and Jacobians onto the nullspace
    // of H_fj.
    // 零空间投影
    JacobiSVD<MatrixXd> svd_helper(H_fj, ComputeFullU | ComputeThinV);
    MatrixXd A = svd_helper.matrixU().rightCols(
        jacobian_row_size - 3);

    // 上面的效果跟QR分解一样，下面的代码可以测试打印对比
    // Eigen::ColPivHouseholderQR<MatrixXd> qr(H_fj);
	// MatrixXd Q = qr.matrixQ();
    // std::cout << "spqr_helper.matrixQ(): " << std::endl << Q << std::endl << std::endl;
    // std::cout << "A: " << std::endl << A << std::endl;

    // 0空间投影
    H_x = A.transpose() * H_xj;
    r = A.transpose() * r_j;

}
/**
 * @brief 计算一个路标点的雅可比
 * @param  cam_state_id 有效的相机状态id
 * @param  feature_id 路标点id
 * @param  H_x 误差相对于位姿的雅可比
 * @param  H_f 误差相对于三维点的雅可比
 * @param  r 误差
 */
void measurementJacobian(
    const StateIDType &cam_state_id,
    const FeatureIDType &feature_id,
    Matrix<double, 4, 6> &H_x, Matrix<double, 4, 3> &H_f, Vector4d &r, StateServer &state_server)
{

    // // Prepare all the required data.
    // // 1. 取出相机状态与特征
    // const CAMState &cam_state = state_server.cam_states[cam_state_id];
    // const Feature &feature = map_server[feature_id];

    // // 2. 取出左目位姿，根据外参计算右目位姿
    // // Cam0 pose.
    // Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);
    // const Vector3d &t_c0_w = cam_state.position;

    // // Cam1 pose.
    // Matrix3d R_c0_c1 = CAMState::T_cam0_cam1.linear();
    // Matrix3d R_w_c1 = CAMState::T_cam0_cam1.linear() * R_w_c0;
    // Vector3d t_c1_w = t_c0_w - R_w_c1.transpose() * CAMState::T_cam0_cam1.translation();

    // // 3. 取出三维点坐标与归一化的坐标点，因为前端发来的是归一化坐标的
    // // 3d feature position in the world frame.
    // // And its observation with the stereo cameras.
    // const Vector3d &p_w = feature.position;
    // const Vector4d &z = feature.observations.find(cam_state_id)->second;

    // // 4. 转到左右目相机坐标系下
    // // Convert the feature position from the world frame to
    // // the cam0 and cam1 frame.
    // Vector3d p_c0 = R_w_c0 * (p_w - t_c0_w);
    // Vector3d p_c1 = R_w_c1 * (p_w - t_c1_w);
    // // p_c1 = R_c0_c1 * R_w_c0 * (p_w - t_c0_w + R_w_c1.transpose() * t_cam0_cam1)
    // //      = R_c0_c1 * (p_c0 + R_w_c0 * R_w_c1.transpose() * t_cam0_cam1)
    // //      = R_c0_c1 * (p_c0 + R_c0_c1 * t_cam0_cam1)

    // // Compute the Jacobians.
    // // 5. 计算雅可比
    // // 左相机归一化坐标点相对于左相机坐标系下的点的雅可比
    // // (x, y) = (X / Z, Y / Z)
    // //下面两行即误差对投影点的导数，原始误差对投影的雅可比矩阵是2*3的，这里是4*3的，是因为左右目都有
    // Matrix<double, 4, 3> dz_dpc0 = Matrix<double, 4, 3>::Zero();
    // dz_dpc0(0, 0) = 1 / p_c0(2);
    // dz_dpc0(1, 1) = 1 / p_c0(2);
    // dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2) * p_c0(2));
    // dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2) * p_c0(2));

    // // 与上同理
    // Matrix<double, 4, 3> dz_dpc1 = Matrix<double, 4, 3>::Zero();
    // dz_dpc1(2, 0) = 1 / p_c1(2);
    // dz_dpc1(3, 1) = 1 / p_c1(2);
    // dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2) * p_c1(2));
    // dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2) * p_c1(2));

    // // 左相机坐标系下的三维点相对于左相机位姿的雅可比 先r后t
    // Matrix<double, 3, 6> dpc0_dxc = Matrix<double, 3, 6>::Zero();
    // dpc0_dxc.leftCols(3) = skewSymmetric(p_c0);
    // dpc0_dxc.rightCols(3) = -R_w_c0;

    // // 右相机坐标系下的三维点相对于左相机位姿的雅可比 先r后t
    // Matrix<double, 3, 6> dpc1_dxc = Matrix<double, 3, 6>::Zero();
    // dpc1_dxc.leftCols(3) = R_c0_c1 * skewSymmetric(p_c0);
    // dpc1_dxc.rightCols(3) = -R_w_c1;

    // // Vector3d p_c0 = R_w_c0 * (p_w - t_c0_w);
    // // Vector3d p_c1 = R_w_c1 * (p_w - t_c1_w);
    // // p_c0 对 p_w
    // Matrix3d dpc0_dpg = R_w_c0;
    // // p_c1 对 p_w
    // Matrix3d dpc1_dpg = R_w_c1;

    // // 两个雅可比
    // H_x = dz_dpc0 * dpc0_dxc + dz_dpc1 * dpc1_dxc;
    // H_f = dz_dpc0 * dpc0_dpg + dz_dpc1 * dpc1_dpg;

    // // Modifty the measurement Jacobian to ensure
    // // observability constrain.
    // // 6. OC
    // Matrix<double, 4, 6> A = H_x;
    // Matrix<double, 6, 1> u = Matrix<double, 6, 1>::Zero();
    // u.block<3, 1>(0, 0) = 
    //     quaternionToRotation(cam_state.orientation_null) * IMUState::gravity;
    // u.block<3, 1>(3, 0) =
    //     skewSymmetric(p_w - cam_state.position_null) * IMUState::gravity;
    // H_x = A - A * u * (u.transpose() * u).inverse() * u.transpose();
    // H_f = -H_x.block<4, 3>(0, 3);

    // // Compute the residual.
    // // 7. 计算归一化平面坐标误差
    // r = z - Vector4d(p_c0(0) / p_c0(2), p_c0(1) / p_c0(2),
    //                     p_c1(0) / p_c1(2), p_c1(1) / p_c1(2));
}
void measurementUpdate(const Eigen::MatrixXd &H, const Eigen::VectorXd &r, StateServer &state_server){
  if (H.rows() == 0 || r.rows() == 0)
        return;

    // Decompose the final Jacobian matrix to reduce computational
    // complexity as in Equation (28), (29).
    MatrixXd H_thin;
    VectorXd r_thin;

    if (H.rows() > H.cols())
    {
        // Convert H to a sparse matrix.
        SparseMatrix<double> H_sparse = H.sparseView();//将H转换为稀疏矩阵

        // Perform QR decompostion on H_sparse.
        // 利用H矩阵稀疏性，QR分解
        // 这段结合零空间投影一起理解，主要作用就是降低计算量
        SPQR<SparseMatrix<double>> spqr_helper;
        spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
        spqr_helper.compute(H_sparse);
        MatrixXd H_temp;
        VectorXd r_temp;
        (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
        (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

        H_thin = H_temp.topRows(21 + state_server.cam_states.size() * 6);
        r_thin = r_temp.head(21 + state_server.cam_states.size() * 6);
    }
    else
    {
        H_thin = H;
        r_thin = r;
    }

    // 2. 标准的卡尔曼计算过程
    // Compute the Kalman gain.
    const MatrixXd &P = state_server.state_cov;
    MatrixXd S = H_thin * P * H_thin.transpose() +
                    Feature::observation_noise * MatrixXd::Identity(
                                                    H_thin.rows(), H_thin.rows());
    // MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
    MatrixXd K_transpose = S.ldlt().solve(H_thin * P);
    MatrixXd K = K_transpose.transpose();

    // Compute the error of the state.
    VectorXd delta_x = K * r_thin;

    // Update the IMU state.
    const VectorXd &delta_x_imu = delta_x.head<21>();

    if ( // delta_x_imu.segment<3>(0).norm() > 0.15 ||
            // delta_x_imu.segment<3>(3).norm() > 0.15 ||
        delta_x_imu.segment<3>(6).norm() > 0.5 ||
        // delta_x_imu.segment<3>(9).norm() > 0.5 ||
        delta_x_imu.segment<3>(12).norm() > 1.0)
    {
        printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
        printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
        ROS_WARN("Update change is too large.");
        // return;
    }

    // 3. 更新到imu状态量
    const Vector4d dq_imu =
        smallAngleQuaternion(delta_x_imu.head<3>());
    Vector4d tmp_rot = rotationToQuaternion(state_server.imu_state.rot_end);
    // 相当于左乘dq_imu
    tmp_rot = quaternionMultiplication(dq_imu, tmp_rot);
    state_server.imu_state.rot_end = quaternionToRotation(tmp_rot);
    state_server.imu_state.bias_g += delta_x_imu.segment<3>(3);
    state_server.imu_state.vel_end += delta_x_imu.segment<3>(6);
    state_server.imu_state.bias_a += delta_x_imu.segment<3>(9);
    state_server.imu_state.pos_end += delta_x_imu.segment<3>(12);

    // 外参
    const Vector4d dq_extrinsic =
        smallAngleQuaternion(delta_x_imu.segment<3>(15));
    READ_RIC[0] =
        quaternionToRotation(dq_extrinsic) * state_server.imu_state.R_imu_cam0;
    READ_TIC[0] += delta_x_imu.segment<3>(18);

    // Update the camera states.
    // 更新相机姿态
    auto cam_state_iter = state_server.cam_states.begin();
    for (int i = 0; i < state_server.cam_states.size(); ++i, ++cam_state_iter)
    {
        const VectorXd &delta_x_cam = delta_x.segment<6>(21 + i * 6);
        const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
        Vector4d tmp_rot = rotationToQuaternion(cam_state_iter->second.orientation); //转换为旋转矩阵
        tmp_rot = quaternionMultiplication(
            dq_cam, tmp_rot);
        cam_state_iter->second.orientation = quaternionToRotation(tmp_rot); //转换为四元数
        cam_state_iter->second.position += delta_x_cam.tail<3>();
    }

    // Update state covariance.
    // 4. 更新协方差
    MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K * H_thin;
    // state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
    //   K*K.transpose()*Feature::observation_noise;
    state_server.state_cov = I_KH * state_server.state_cov;

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov +
                                state_server.state_cov.transpose()) /
                                2.0;
    state_server.state_cov = state_cov_fixed;
}
bool gatingTest(const Eigen::MatrixXd &H, const Eigen::VectorXd &r, const int &dof, StateServer &state_server){
  // 输入的dof的值是所有相机观测，且没有去掉滑窗的
    // 而且按照维度这个卡方的维度也不对
    // 
    MatrixXd P1 = H * state_server.state_cov * H.transpose();
    MatrixXd P2 = Feature::observation_noise *
                    MatrixXd::Identity(H.rows(), H.rows());
    double gamma = r.transpose() * (P1 + P2).ldlt().solve(r);

    // cout << dof << " " << gamma << " " <<
    //   chi_squared_test_table[dof] << " ";

    if (gamma < chi_squared_test_table[dof]){
        // cout << "passed" << endl;
        return true;
    }
    else{
        // cout << "failed" << endl;
        return false;
    }
}

void pruneCamStateBuffer(StateServer &state_server){
  // 数量还不到该删的程度，配置文件里面是20个
  std::cout << "state_server.cam_states.size: " << state_server.cam_states.size() << std::endl;
    if (state_server.cam_states.size() < max_cam_state_size)
        return;

    // Find two camera states to be removed.
    // 1. 找出该删的相机状态的id，两个
    vector<StateIDType> rm_cam_state_ids(0);
    findRedundantCamStates(rm_cam_state_ids, state_server);
    // Find the size of the Jacobian matrix.
    // 2. 找到删减帧涉及的观测雅可比大小
    int jacobian_row_size = 0;
    for (auto &item : map_server){
        auto &feature = item.second;
        // Check how many camera states to be removed are associated
        // with this feature.
        // 2.1 在待删去的帧中统计能观测到这个特征的帧
        vector<StateIDType> involved_cam_state_ids(0);
        for (const auto &cam_id : rm_cam_state_ids){
            if (feature.observations.find(cam_id) !=
                feature.observations.end()) //找到了对应的ID
                involved_cam_state_ids.push_back(cam_id);
        }
        if (involved_cam_state_ids.size() == 0)
            continue;
        // 2.2 这个点只在一个里面有观测那就直接删
        // 只用一个观测更新不了状态
        if (involved_cam_state_ids.size() == 1)
        {
            feature.observations.erase(involved_cam_state_ids[0]);
            continue;
        }
        // 程序到这里的时候说明找到了一个特征，先不说他一共被几帧观测到
        // 到这里说明被两帧或两帧以上待删减的帧观测到
        // 2.3 如果没有做过三角化，做一下三角化，如果失败直接删
        if (!feature.is_initialized)
        {
            // Check if the feature can be initialize.
            if (!feature.checkMotion(state_server.cam_states))
            {
                // If the feature cannot be initialized, just remove
                // the observations associated with the camera states
                // to be removed.
                for (const auto &cam_id : involved_cam_state_ids)
                    feature.observations.erase(cam_id);
                continue;
            }
            else
            {
                if (!feature.initializePosition(state_server.cam_states))
                {
                    for (const auto &cam_id : involved_cam_state_ids)
                        feature.observations.erase(cam_id);
                    continue;
                }
            }
        }

        // 2.4 最后的最后得出了行数
        // 意味着有involved_cam_state_ids.size() 数量的观测要被删去
        // 但是因为待删去的帧间有共同观测的关系，直接删会损失这部分信息
        // 所以临删前做最后一次更新
        jacobian_row_size += 4 * involved_cam_state_ids.size() - 3;
    }
    // Compute the Jacobian and residual.
    // 3. 计算待删掉的这部分观测的雅可比与误差
    // 预设大小
    MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
                                    21 + 6 * state_server.cam_states.size());
    VectorXd r = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;

    // 又做了一遍类似上面的遍历，只不过该三角化的已经三角化，该删的已经删了
    for (auto &item : map_server)
    {
        auto &feature = item.second;
        // Check how many camera states to be removed are associated
        // with this feature.
        // 这段就是判断一下这个点是否都在待删除帧中有观测
        vector<StateIDType> involved_cam_state_ids(0);
        for (const auto &cam_id : rm_cam_state_ids)
        {
            if (feature.observations.find(cam_id) !=
                feature.observations.end())
                involved_cam_state_ids.push_back(cam_id);
        }

        // 一个的情况已经被删掉了
        if (involved_cam_state_ids.empty())
            continue;

        // 计算出待删去的这部分的雅可比
        // 这个点假如有多个观测，但本次只用待删除帧上的观测
        MatrixXd H_xj;
        VectorXd r_j;
        featureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j, state_server);

        if (gatingTest(H_xj, r_j, involved_cam_state_ids.size(), state_server))
        {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += H_xj.rows();
        }

        // 删去观测
        for (const auto &cam_id : involved_cam_state_ids)
            feature.observations.erase(cam_id);
    }

    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);

    // Perform measurement update.
    // 4. 用待删去的这些观测更新一下
    measurementUpdate(H_x, r, state_server);

    // 5. 直接删掉对应的行列，直接干掉
    // 为啥没有做类似于边缘化的操作？
    // 个人认为是上面做最后的更新了，所以信息已经更新到了各个地方
    for (const auto &cam_id : rm_cam_state_ids)
    {
        int cam_sequence = std::distance(
            state_server.cam_states.begin(), state_server.cam_states.find(cam_id));
        int cam_state_start = 21 + 6 * cam_sequence;
        int cam_state_end = cam_state_start + 6;

        // Remove the corresponding rows and columns in the state
        // covariance matrix.
        if (cam_state_end < state_server.state_cov.rows())
        {
            state_server.state_cov.block(cam_state_start, 0,
                                         state_server.state_cov.rows() - cam_state_end,
                                         state_server.state_cov.cols()) =
                state_server.state_cov.block(cam_state_end, 0,
                                             state_server.state_cov.rows() - cam_state_end,
                                             state_server.state_cov.cols());

            state_server.state_cov.block(0, cam_state_start,
                                            state_server.state_cov.rows(),
                                            state_server.state_cov.cols() - cam_state_end) =
                state_server.state_cov.block(0, cam_state_end,
                                                state_server.state_cov.rows(),
                                                state_server.state_cov.cols() - cam_state_end);

            state_server.state_cov.conservativeResize(
                state_server.state_cov.rows() - 6, state_server.state_cov.cols() - 6);
        }
        else
        {
            state_server.state_cov.conservativeResize(
                state_server.state_cov.rows() - 6, state_server.state_cov.cols() - 6);
        }

        // Remove this camera state in the state vector.
        state_server.cam_states.erase(cam_id);
    }

}
/**
 * @brief 找出该删的相机状态的id
 * @param  rm_cam_state_ids 要删除的相机状态id
 */
void findRedundantCamStates(vector<StateIDType> &rm_cam_state_ids,
                            StateServer &state_server){
    // Move the iterator to the key position.
    // 1. 找到倒数第四个相机状态，作为关键状态
    auto key_cam_state_iter = state_server.cam_states.end();
    for (int i = 0; i < 4; ++i)
        --key_cam_state_iter;

    // 倒数第三个相机状态
    auto cam_state_iter = key_cam_state_iter;
    ++cam_state_iter;

    // 序列中，第一个相机状态
    auto first_cam_state_iter = state_server.cam_states.begin();

    // Pose of the key camera state.
    // 2. 关键状态的位姿
    const Vector3d key_position =
        key_cam_state_iter->second.position;
    const Matrix3d key_rotation = 
        key_cam_state_iter->second.orientation;

    // Mark the camera states to be removed based on the
    // motion between states.
    // 3. 遍历两次，必然删掉两个状态，有可能是相对新的，有可能是最旧的
    // 但是永远删不到最新的（先遍历倒数第三帧的，再遍历倒数第二帧的）
    for (int i = 0; i < 2; ++i){
        // 从倒数第三个开始
        const Vector3d position =
            cam_state_iter->second.position;
        const Matrix3d rotation = 
            cam_state_iter->second.orientation;

        // 计算相对于关键相机状态的平移与旋转
        double distance = (position - key_position).norm();
        //R3*R4^T表示从R4到R3的旋转（R4*△R=R3）
        double angle = AngleAxisd(rotation * key_rotation.transpose()).angle();
        // 判断大小以及跟踪率，就是cam_state_iter这个状态与关键相机状态的相似度，
        // 且当前的点跟踪率很高
        // 删去这个帧，否则删掉最老的
        if (angle < rotation_threshold &&
            distance < translation_threshold &&
            tracking_rate > tracking_rate_threshold){
            rm_cam_state_ids.push_back(cam_state_iter->first);
            ++cam_state_iter;
        }
        else{
            rm_cam_state_ids.push_back(first_cam_state_iter->first);
            ++first_cam_state_iter;
        }
    }

    // Sort the elements in the output vector.
    // 4. 排序
    sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());
}