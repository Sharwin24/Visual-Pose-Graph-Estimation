#ifndef CAMERA_MANAGER_HPP
#define CAMERA_MANAGER_HPP
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>

#include <mutex>
#include <queue>
#include <vector>

using Header = std_msgs::msg::Header;
using Image = sensor_msgs::msg::Image;
using CameraInfo = sensor_msgs::msg::CameraInfo;

struct CameraIntrinsics {
  cv::Mat K; // Camera intrinsic matrix
  cv::Mat D; // Camera distortion coefficients
};

struct Frame {
  int frameID;
  rclcpp::Time stamp;
  cv::Mat image;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors; // SIFT/ORB descriptors
  cv::Mat K; // Camera intrinsic matrix
  cv::Mat D; // Camera distortion coefficients
};

struct Edge {
  int fromID;
  int toID;
  cv::Mat relativePose; // 4x4 SE(3) Transformation matrix T_from_to
  cv::Mat covariance; // Covariance of the relative pose
  bool isLoopClosure; // Flag for loop closure edge
};

struct FeatureMap {
  int frameID;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors; // SIFT/ORB descriptors
};

class CameraManager : public rclcpp::Node {
public:
  CameraManager();
  ~CameraManager() = default;

  FeatureMap FeatureExtractor(const Frame& frame);
  Edge CameraPoseEstimation(const FeatureMap& featureMap);
  Edge LoopClosureDetector();
  void GraphBuilder(const Edge& estimatedPose, const Edge& loopConstraints);
  void VisualizeGraph();

private:
  // Timer for camera image processing
  rclcpp::TimerBase::SharedPtr timer;
  // Subscriber to camera image
  rclcpp::Subscription<Image>::SharedPtr cameraSub;
  // Subscriber to camera info
  rclcpp::Subscription<CameraInfo>::SharedPtr cameraInfoSub;

  void timerCallback();
  // Timer Frequency
  float timerFreq; // [Hz]

  // Camera Intrinsics Info
  CameraIntrinsics cameraIntrinsics;
  bool collectedCameraInfo = false;

  // Mutex for safely accessing the frame queue
  std::mutex frameMutex;

  // Container for frames
  std::queue<Frame> frameQueue;

  // Container for edges
  std::vector<Edge> allEdges;

};
#endif // !CAMERA_MANAGER_HPP