#ifndef CAMERA_MANAGER_HPP
#define CAMERA_MANAGER_HPP
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include "tf2_ros/transform_broadcaster.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h> // may not be needed
#include <g2o/core/block_solver.h>       // may not be needed
#include <g2o/core/optimization_algorithm_levenberg.h>  
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/core/sparse_optimizer.h>
#include <Eigen/Geometry>
#include <mutex>
#include <queue>
#include <vector>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <map>


using Header = std_msgs::msg::Header;
using CameraInfo = sensor_msgs::msg::CameraInfo;
using Path = nav_msgs::msg::Path;
using PoseStamped = geometry_msgs::msg::PoseStamped;
using ImageMsg = sensor_msgs::msg::Image;
using MarkerArray = visualization_msgs::msg::MarkerArray;
using CameraInfoMsg = sensor_msgs::msg::CameraInfo;
using StereoSyncPolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, CameraInfoMsg, ImageMsg, CameraInfoMsg>;
using MonoSyncPolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, CameraInfoMsg>;


geometry_msgs::msg::Quaternion e2q(double roll, double pitch, double yaw) {
  geometry_msgs::msg::Quaternion q;
  q.x = sin(roll / 2) * cos(pitch / 2) * cos(yaw / 2) - cos(roll / 2) * sin(pitch / 2) * sin(yaw / 2);
  q.y = cos(roll / 2) * sin(pitch / 2) * cos(yaw / 2) + sin(roll / 2) * cos(pitch / 2) * sin(yaw / 2);
  q.z = cos(roll / 2) * cos(pitch / 2) * sin(yaw / 2) - sin(roll / 2) * sin(pitch / 2) * cos(yaw / 2);
  q.w = cos(roll / 2) * cos(pitch / 2) * cos(yaw / 2) + sin(roll / 2) * sin(pitch / 2) * sin(yaw / 2);
  // Normalize the quaternion to ensure it is a valid rotation
  double norm = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (norm > 0) {
    q.x /= norm;
    q.y /= norm;
    q.z /= norm;
    q.w /= norm;
  } else {
    // If norm is zero, return a default quaternion
    q.x = 0.0;
    q.y = 0.0;
    q.z = 0.0;
    q.w = 1.0; // Default quaternion representing no rotation
  }
  return q;
}

struct CameraIntrinsics {
  cv::Mat K; // Camera intrinsic matrix: focal length, principal point..
  cv::Mat D; // Camera distortion coefficients
};

struct Frame {
  int frameID;
  rclcpp::Time stamp;
  cv::Mat image;
  std::vector<cv::KeyPoint> keypoints;
  CameraIntrinsics intrinsics; // Camera intrinsics
  cv::Mat descriptors; // SIFT/ORB descriptors
};

struct Feature {
  int frameID;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors; // SIFT/ORB descriptors
};

struct StereoFeature {
  int frameID;
  std::vector<cv::KeyPoint> leftKeypoints;
  std::vector<cv::KeyPoint> rightKeypoints;
  cv::Mat leftDescriptors; // SIFT/ORB descriptors for left image
  cv::Mat rightDescriptors; // SIFT/ORB descriptors for right image
};

struct Edge {
  int fromID;
  int toID;
  cv::Mat relativePose; // 4x4 SE(3) Transformation matrix T_from_to
  cv::Mat covariance; // Covariance of the relative pose
};



class CameraManager : public rclcpp::Node {
public:
  CameraManager();
  ~CameraManager() = default;

  Feature MonocularFeatureExtractor(const Frame& frame);
  StereoFeature StereoFeatureExtractor(const Frame& leftFrame, const Frame& rightFrame);
  Edge MonocularCameraPoseEstimation(const Feature& newFeature);
  Edge StereoCameraPoseEstimation(const StereoFeature& newFeature);
  void UpdateCameraPoseVisualization();
  void initializePoseGraph();
  void addEdge(const Edge& edge);
  void addNode(int frameID, const cv::Mat& currentPose);
  void optimizePoseGraph();
  Edge LoopClosureDetector();
  void visulizePoseGraph();

private:
  // Timer for camera image processing
  rclcpp::TimerBase::SharedPtr timer;

  // Synchronized subscribers for stereo and monocular cameras
  std::shared_ptr<message_filters::Subscriber<ImageMsg>> leftCameraSub;
  std::shared_ptr<message_filters::Subscriber<CameraInfoMsg>> leftCameraInfoSub;
  std::shared_ptr<message_filters::Subscriber<ImageMsg>> rightCameraSub;
  std::shared_ptr<message_filters::Subscriber<CameraInfoMsg>> rightCameraInfoSub;
  std::shared_ptr<message_filters::Subscriber<ImageMsg>> cameraSub;
  std::shared_ptr<message_filters::Subscriber<CameraInfoMsg>> cameraInfoSub;
  std::shared_ptr<message_filters::Synchronizer<StereoSyncPolicy>> stereoSync;
  std::shared_ptr<message_filters::Synchronizer<MonoSyncPolicy>> monoSync;

  // Publisher for estimated path of the camera pose
  rclcpp::Publisher<Path>::SharedPtr cameraEstimatePathPub;
  Path cameraPoseEstimatePath;

  // Publisher for images with features
  rclcpp::Publisher<ImageMsg>::SharedPtr featureImagePub;

  // Publisher for pose-graph visualization markers
  rclcpp::Publisher<MarkerArray>::SharedPtr poseGraphVizPub;

  // Transform Broadcaster for publishing camera poses
  std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;

  // Timer Callback for pose-graph management and visualization
  void timerCallback();

  // Synchronized callback for sterero cameras
  void synchronizedStereoCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr left_image_msg,
    const CameraInfo::ConstSharedPtr left_info_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr right_image_msg,
    const CameraInfo::ConstSharedPtr right_info_msg);

  // Synchronized callback for monocular cameras
  void synchronizedMonocularCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr image_msg,
    const CameraInfo::ConstSharedPtr info_msg);

  // Camera Intrinsics Info
  CameraIntrinsics cameraIntrinsics;
  CameraIntrinsics leftCameraIntrinsics;
  CameraIntrinsics rightCameraIntrinsics;

  // Configuration Parameters
  float timerFreq; // Timer frequency [Hz]
  bool useStereoCamera; // Use stereo camera or monocular camera
  double stereoBaseline; // Baseline distance between stereo cameras [m]
  std::string featureExtractionMethod; // Feature extraction method ["SIFT", "ORB", "SURF"]

  // Internal State (previous frames and features)
  Frame previousFrame; // Previous frame for pose estimation
  StereoFeature previousStereoFeature; // Previous stereo feature for pose estimation
  Feature previousFeature; // Previous feature for pose estimation

  // Pose Estimate
  cv::Mat currentPose; // Current camera pose estimate [SE(3)]
  std::mutex poseMutex; // Mutex for currentPose estimate

  // Graph Parameters.
  std::unique_ptr<g2o::SparseOptimizer> optimizer; // The main Pose-Graph
  std::mutex poseGraphMutex; // Mutex for pose-graph operations

};
#endif // !CAMERA_MANAGER_HPP