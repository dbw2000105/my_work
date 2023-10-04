#include <cv_bridge/cv_bridge.h>
#include <iomanip>
#include <message_filters/subscriber.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Bool.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;
//这里是发布三个话题，pub_img是特征点，pub_match是带特征点的图片，pub_restart则是看看特征追踪是否出错
ros::Publisher pub_img, pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData
    [NUM_OF_CAM]; // 图像的追踪信息，包括prev、cur、forw帧的特征以及特征的移动速度等
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

void img_callback(const sensor_msgs::ImageConstPtr &img_msg) {
  // 读取第一帧图片，如果是第一帧图片
  if (first_image_flag) {
    first_image_flag = false;
    first_image_time = img_msg->header.stamp.toSec();
    last_image_time = img_msg->header.stamp.toSec();
    return;
  }
  /*******************************************************************************************/
  // detect unstable camera stream
  // 错误读取的处理
  // 如果两帧时间差过长或者有问题，就重新开始并发布重新开始的布尔量
  if (img_msg->header.stamp.toSec() - last_image_time > 1.0 ||
      img_msg->header.stamp.toSec() < last_image_time) {
    ROS_WARN("image discontinue! reset the feature tracker!");
    first_image_flag = true;
    last_image_time = 0;
    pub_count = 1;
    std_msgs::Bool restart_flag;
    restart_flag.data = true;
    pub_restart.publish(restart_flag);
    return;
  }
  /*******************************************************************************************/
  // 通过预设的freq量来进行发布的频率控制
  last_image_time = img_msg->header.stamp.toSec();
  /**
  节点feature_tracker提取到的特征点会发布到话题/feature_tracker/feature上,发布的频率由配置文件Vins-Mono/config/euroc/euroc_config.yaml中的配置项freq
  指定.数据集发布相机图像话题/cam0/image_raw的频率未必与配置文件指定的发布频率相同(数据集发布图像的频率一般会高于节点feature_tracker发布特征点的
  频率),因此需要进行频率控制,抽帧发布特征点追踪结果.
   */
  // 这里通过控制pub_count / (img_msg->header.stamp.toSec() -
  // first_image_time))这个量，其中分子代表已发布的次数，分母则为当前帧与初始帧的时间差，
  // 通过控制该比例即可较好地控制发布的频率。当然这种频率控制方法并不聪明
  if (round(1.0 * pub_count /
            (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ) {
    PUB_THIS_FRAME = true;
    // reset the frequency control
    /**
    系统长时间运行后,从数值上来说,实际频率计算式的分子pub_count和分母img_msg->header.stamp.toSec()-first_image_time都较大,
    这样系统对**瞬时发布速率变化不敏感**,容易造成瞬时数据拥堵(连续几帧都发布或连续几帧都不发布).为避免瞬时数据拥堵,
    需要周期性重置计数器pub_count和first_image_time.实践上选择当实际频率十分接近给定频率时,重置计数器.
     */
    if (abs(1.0 * pub_count /
                (img_msg->header.stamp.toSec() - first_image_time) -
            FREQ) < 0.01 * FREQ) {
      first_image_time = img_msg->header.stamp.toSec();
      pub_count = 0;
    }
  } else
    PUB_THIS_FRAME = false;
  /*******************************************************************************************/
  /*******************************************************************************************/
  //下边才是对图像的处理，需要注意的是即使不发布也是正常做光流追踪的！光流对图像的变化要求尽可能小
  /*******************************************************************************************/
  //将消息转换为cv图像
  cv_bridge::CvImageConstPtr ptr;
  if (img_msg->encoding == "8UC1") {
    sensor_msgs::Image img;
    img.header = img_msg->header;
    img.height = img_msg->height;
    img.width = img_msg->width;
    img.is_bigendian = img_msg->is_bigendian;
    img.step = img_msg->step;
    img.data = img_msg->data;
    img.encoding = "mono8";
    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  } else
    ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

  cv::Mat show_img = ptr->image;
  /*******************************************************************************************/
  TicToc t_r;
  // 特征提取，如果是单目则提取特征并更新追踪信息，双目好像只更新了图像
  for (int i = 0; i < NUM_OF_CAM; i++) {
    ROS_DEBUG("processing camera %d", i);
    if (i != 1 || !STEREO_TRACK) // 单目，实际上也只有单目
      trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)),
                               img_msg->header.stamp.toSec());
    else {
      if (EQUALIZE) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)),
                     trackerData[i].cur_img);
      } else
        trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
    }

#if SHOW_UNDISTORTION
    trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
  }
  /*******************************************************************************************/
  // 更新id,这个特征点的id我仍然没有搞懂是怎么给出来的
  for (unsigned int i = 0;; i++) {
    bool completed = false;
    for (int j = 0; j < NUM_OF_CAM; j++)
      if (j != 1 || !STEREO_TRACK)
        completed |= trackerData[j].updateID(i);
    if (!completed)
      break;
  }
  /*******************************************************************************************/
  //发布
  // 一帧图像所包含的所有数据都在这个feature_points中，其包括points项和channels项。
  // points项传递cur帧所有特征点的归一化坐标，channels项则包含许多。
  // channels[0]传递特征点的id序列，channels[1]和channels[2]传递cur帧特征点的像素坐标序列，
  // channels[3]和channels[4]传递cur帧相对prev帧特征点沿x,y方向的像素移动速度
  if (PUB_THIS_FRAME) {
    pub_count++;
    sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
    sensor_msgs::ChannelFloat32 id_of_point;
    sensor_msgs::ChannelFloat32 u_of_point;
    sensor_msgs::ChannelFloat32 v_of_point;
    sensor_msgs::ChannelFloat32 velocity_x_of_point;
    sensor_msgs::ChannelFloat32 velocity_y_of_point;

    feature_points->header = img_msg->header;
    feature_points->header.frame_id = "world";

    vector<set<int>> hash_ids(NUM_OF_CAM);
    for (int i = 0; i < NUM_OF_CAM; i++) {
      auto &un_pts = trackerData[i].cur_un_pts;
      auto &cur_pts = trackerData[i].cur_pts;
      auto &ids = trackerData[i].ids;
      auto &pts_velocity = trackerData[i].pts_velocity;
      for (unsigned int j = 0; j < ids.size(); j++) {
        if (trackerData[i].track_cnt[j] > 1) {
          int p_id = ids[j];
          hash_ids[i].insert(p_id);
          geometry_msgs::Point32 p;
          p.x = un_pts[j].x;
          p.y = un_pts[j].y;
          p.z = 1;

          feature_points->points.push_back(p);
          id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
          u_of_point.values.push_back(cur_pts[j].x);
          v_of_point.values.push_back(cur_pts[j].y);
          velocity_x_of_point.values.push_back(pts_velocity[j].x);
          velocity_y_of_point.values.push_back(pts_velocity[j].y);
        }
      }
    }
    //
    feature_points->channels.push_back(id_of_point);
    feature_points->channels.push_back(u_of_point);
    feature_points->channels.push_back(v_of_point);
    feature_points->channels.push_back(velocity_x_of_point);
    feature_points->channels.push_back(velocity_y_of_point);
    ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(),
              ros::Time::now().toSec());
    // skip the first image; since no optical speed on frist image
    if (!init_pub) {
      init_pub = 1;
    } else
      pub_img.publish(feature_points);
    //特征的可视化
    if (SHOW_TRACK) {
      ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
      // cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
      cv::Mat stereo_img = ptr->image;

      for (int i = 0; i < NUM_OF_CAM; i++) {
        cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
        cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

        for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++) {
          double len =
              std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
          cv::circle(tmp_img, trackerData[i].cur_pts[j], 5,
                     cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        }
      }
      // cv::imshow("vis", stereo_img);
      // cv::waitKey(5);
      pub_match.publish(ptr->toImageMsg());
    }
  }
  ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "feature_tracker");
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                 ros::console::levels::Info);
  readParameters(n);

  for (int i = 0; i < NUM_OF_CAM; i++)
    trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

  if (FISHEYE) {
    for (int i = 0; i < NUM_OF_CAM; i++) {
      trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
      if (!trackerData[i].fisheye_mask.data) {
        ROS_INFO("load mask fail");
        ROS_BREAK();
      } else
        ROS_INFO("load mask success");
    }
  }

  ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);

  pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
  pub_match = n.advertise<sensor_msgs::Image>("feature_img", 1000);
  pub_restart = n.advertise<std_msgs::Bool>("restart", 1000);
  /*
  if (SHOW_TRACK)
      cv::namedWindow("vis", cv::WINDOW_NORMAL);
  */
  ros::spin();
  return 0;
}

// new points velocity is 0, pub or not?
// track cnt > 1 pub?