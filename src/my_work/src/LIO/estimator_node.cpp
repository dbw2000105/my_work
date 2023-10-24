#include <iostream>
#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "common_lib.h"
#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"
#define CAM_MEASUREMENT_COV 1e-3

//; 全局变量，Camera_Lidar_queue是一个结构体，存储camera_lidar数据
Camera_Lidar_queue g_camera_lidar_queue;
MeasureGroup Measures;    //; 一帧lidar和IMU 数据，这个是不是不应该定义？
StatesGroup g_lio_state;  //; lio的状态，里面包括 Q P V bg ba G  一共3*6=18维
Estimator estimator;
StateServer state_server; //存储IMU和cam的数据

std::condition_variable con; // 条件变量
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

// 互斥量
std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;

// IMU项[P,Q,B,Ba,Bg,a,g]
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;

eigen_q diff_vins_lio_q = eigen_q::Identity();
vec_3 diff_vins_lio_t = vec_3::Zero();

bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = -1;

// imu回调函数，将imu_msg保存到imu_buf，IMU状态递推并发布[P,Q,V,header]
void imu_callback(const sensor_msgs::ImuConstPtr &_imu_msg){
    sensor_msgs::ImuPtr imu_msg = boost::make_shared<sensor_msgs::Imu>();
    *imu_msg = *_imu_msg;

    //; 对于livox内置IMU，以G为单位进行了归一化，这里就是去归一化得到真实值
    //; m_if_acc_mul_G ： acc 是否 multipy G 
    if (g_camera_lidar_queue.m_if_acc_mul_G) {// For LiVOX Avia built-in IMU
        imu_msg->linear_acceleration.x *= 9.805;
        imu_msg->linear_acceleration.y *= 9.805;
        imu_msg->linear_acceleration.z *= 9.805;
    }
    //?-------- 增加结束 ----------

    // 判断时间间隔是否为正
    if (imu_msg->header.stamp.toSec() <= last_imu_t){
        ROS_WARN("imu message in disorder!");
        return;
    }

    g_camera_lidar_queue.imu_in(imu_msg->header.stamp.toSec());

    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg); // 将imu_msg放入imu_buf，存的是指针
    m_buf.unlock();
    con.notify_one(); // 唤醒作用于process线程中的获取观测值数据的函数
    last_imu_t = imu_msg->header.stamp.toSec();{
        // 构造互斥锁m_state，析构时解锁
        std::lock_guard<std::mutex> lg(m_state);
        //; 仅仅使用IMU数据中值积分得到最新的PVQ，目的是为了实现高频的IMU里程计
        // predict(imu_msg); // 递推得到IMU的PQV
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";

        // 发布最新的由IMU直接递推得到的PQV
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

// feature回调函数，将feature_msg放入feature_buf
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg){
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }

    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

// restart回调函数，收到restart时清空feature_buf和imu_buf，估计器重置，时间重置
void restart_callback(const std_msgs::BoolConstPtr &restart_msg){
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while (!feature_buf.empty())
            feature_buf.pop();

        while (!imu_buf.empty())
            imu_buf.pop();

        m_buf.unlock();
        m_estimator.lock();
        // estimator.clearState();
        // estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }

    return;
}
//给vio上锁
void lock_lio(Estimator &estimator){
    if (estimator.m_fast_lio_instance)
    {
        estimator.m_fast_lio_instance->m_mutex_lio_process.lock();
    }
}
// thread: visual-inertial odometry
void unlock_lio(Estimator &estimator)
{
    if (estimator.m_fast_lio_instance)
    {
        estimator.m_fast_lio_instance->m_mutex_lio_process.unlock();
    }
}
/*
    对imu和图像数据进行对齐并组合
*/
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> getMeasurements(){
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
    std::unique_lock<std::mutex> lk(m_buf);
    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;
        
        //? 也就是第一帧图像特征数据的时间要夹在这一组imu数据之间
        // 对齐标准：IMU最后一个数据的时间要大于第一个图像特征数据的时间
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            sum_of_wait++;
            return measurements;
        }

        // 对齐标准：IMU第一个数据的时间要小于第一个图像特征数据的时间
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning [%.5f | %.5f] ", imu_buf.front()->header.stamp.toSec(), feature_buf.front()->header.stamp.toSec());
            feature_buf.pop();
            continue;
        }

        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;

        // 图像数据(img_msg)，对应多组在时间戳内的imu数据,然后塞入measurements
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            // emplace_back相比push_back能更好地避免内存的拷贝与移动
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }

        // 这里把下一个imu_msg也放进去了,但没有pop，因此当前图像帧和下一图像帧会共用这个imu_msg
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");

        measurements.emplace_back(IMUs, img_msg);
        state_server.imu_state.id++;
    }

    return measurements;
}
//将全局状态赋值给msckf状态，不改变原先id的值
inline void system_state_transform(){
  state_server.imu_state.rot_end = g_lio_state.rot_end;
  state_server.imu_state.pos_end = g_lio_state.pos_end;
  state_server.imu_state.vel_end = g_lio_state.vel_end;
  state_server.imu_state.bias_a = g_lio_state.bias_a;
  state_server.imu_state.bias_g = g_lio_state.bias_g;
  // state_server.imu_state.cov = g_lio_state.cov;
  state_server.state_cov.block<18, 18>(0, 0) = g_lio_state.cov;
  state_server.imu_state.last_update_time = g_lio_state.last_update_time;
  state_server.imu_state.gravity = g_lio_state.gravity;
}

void process(){
  Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> G, H_T_H, I_STATE; //G = K * H
  G.setZero();
  H_T_H.setZero();
  I_STATE.setIdentity();
  std::shared_ptr<ImuProcess> p_imu(new ImuProcess());
  state_server.state_cov = Eigen::MatrixXd::Identity(DIM_OF_STATES, DIM_OF_STATES);
  // StateServer state_server; //存储IMU和cam的数据
  g_camera_lidar_queue.m_if_lidar_can_start =
      g_camera_lidar_queue.m_if_lidar_start_first; //雷达首先启动

  while(true){
        //每个vector中有一个pair，pair中的first是IMU数据，second是图像数据，另外每一个pair中first是一个vector，里面存储的是IMU数据，second是一个PointCloudConstPtr
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        measurements = getMeasurements(); //每次始终拿到一个测量，包括一帧IMU数据和一帧图像数据
        if (measurements.empty()){
            continue;
        }
        // std::cout << "angular_velocity: " << measurements[0].first.back()->angular_velocity<< std::endl; 
        m_estimator.lock();
        //; 每次循环都会给设置一个负无穷大的数
        g_camera_lidar_queue.m_last_visual_time = -3e8;

        // TicToc t_s;
        //; 遍历测量到的所有camera和imu数据
        for (auto &measurement : measurements){
            // 对应这段的img data
            auto img_msg = measurement.second;
      
            //?-------- 增加开始 ----------
            int if_camera_can_update = 1;
            //; cam_update_tim ：当前帧的图像时间
            double cam_update_tim = img_msg->header.stamp.toSec() + estimator.td;
            
            // ANCHOR - determine if update of not.
            //; 此时已经开始了LIO线程，所以这个一定成立
            if (estimator.m_fast_lio_instance != nullptr){
                g_camera_lidar_queue.m_camera_imu_td = estimator.td; //todo这里表示当前相机帧和IMU帧的时间差？
                //; 更新最新的图片时间
                g_camera_lidar_queue.m_last_visual_time = img_msg->header.stamp.toSec();
                //; 判断是否能够处理这一帧的camera数据，跟lidar的判断是一样的
                while (g_camera_lidar_queue.if_camera_can_process() == false){
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                //! 注意：这里锁住了lio，也就是vio运行的时候lio是被锁住的！这样保证vio运行的时候， lio无法运行
                lock_lio(estimator);
                
                // t_s.tic();

                *p_imu = *(estimator.m_fast_lio_instance->m_imu_process);  //; IMU处理的类，相当于拿到了LIO中处理好的IMU的数据
                // std::cout << "p_imu->angvel_last: " << p_imu->angvel_last << std::endl;
            }
          //相机状态扩增  
          system_state_transform();
          stateAugmentation(measurements.back().second->header.stamp.toSec(), state_server);
          unlock_lio(estimator);
        }
      m_estimator.unlock();
    }
}

int main(int argc, char **argv){
    // ROS初始化，设置句柄n
    //; 注意这里的vins_estimator在运行的时候被launch文件改成了r2live，
    //; 所以实际上这个节点才是真正运行的r2live后端节点
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    // 读取参数，设置估计器参数
    readParameters(nh);
    estimator.setParameter();

    //?-------- 增加开始 ----------
    get_ros_parameter(nh, "/lidar_drag_cam_tim", g_camera_lidar_queue.m_lidar_drag_cam_tim, 1.0);  //; 配置文件中是10
    get_ros_parameter(nh, "/acc_mul_G", g_camera_lidar_queue.m_if_acc_mul_G, 0);  //; 加速度是否以G为单位进行了归一化
    //; 是否雷达先启动的标志，在配置文件中写的是1
    get_ros_parameter(nh, "/if_lidar_start_first", g_camera_lidar_queue.m_if_lidar_start_first, 1.0);
    get_ros_parameter<int>(nh, "/if_write_to_bag", g_camera_lidar_queue.m_if_write_res_to_bag, false);
    get_ros_parameter<int>(nh, "/if_dump_log", g_camera_lidar_queue.m_if_dump_log, 0);
    get_ros_parameter<std::string>(nh, "/record_bag_name", g_camera_lidar_queue.m_bag_file_name, "./");
    if (g_camera_lidar_queue.m_if_write_res_to_bag)
    {
        // 初始化写入rosbag包
        g_camera_lidar_queue.init_rosbag_for_recording();
    }
    // ANCHOR - Start lio process
    //; 锚 —— 开启LIO线程
    g_camera_lidar_queue.m_if_lidar_can_start = true;  //; 首先置位雷达不能启动标志
    if (estimator.m_fast_lio_instance == nullptr)
    {
        estimator.m_fast_lio_instance = new Fast_lio();   //; 在类构造函数中就开启了LIO线程
    }
    //?-------- 增加结束 ----------

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    // 用于RVIZ显示的Topic
    registerPub(nh);

    // 订阅IMU、feature、restart、match_points的topic,执行各自回调函数
    ros::Subscriber sub_imu = nh.subscribe(IMU_TOPIC, 20000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = nh.subscribe("/feature_tracker/feature", 20000, feature_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_restart = nh.subscribe("/feature_tracker/restart", 20000, restart_callback, ros::TransportHints().tcpNoDelay());
    

    // 创建VIO主线程`
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}