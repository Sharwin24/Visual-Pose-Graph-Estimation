#include "camera_manager.hpp"

CameraManager::CameraManager() : Node("camera_manager") {
  RCLCPP_INFO(this->get_logger(), "Camera Manager Node Initialized");

  // Declare parameters
  this->timerFreq = this->declare_parameter("timer_frequency", 10.0f); // [Hz]
  // Get params from config file
  this->timerFreq = this->get_parameter("timer_frequency").as_double();

  // Create subscriber to camera info
  this->cameraInfoSub = this->create_subscription<CameraInfo>(
    "camera/camera/color/camera_info", 10,
    [this](const CameraInfo::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(this->frameMutex);
    this->cameraIntrinsics.K = cv::Mat(3, 3, CV_64F, const_cast<double*>(msg->k.data())).clone();
    this->cameraIntrinsics.D = cv::Mat(msg->d.size(), 1, CV_64F, const_cast<double*>(msg->d.data())).clone();
    this->collectedCameraInfo = true;
    RCLCPP_INFO(this->get_logger(), "Camera Intrinsics Collected");
  }
  );

  // Create subscriber to camera image and convert to OpenCV Image
  this->cameraSub = this->create_subscription<Image>(
    "camera/camera/color/image_raw", 10,
    [this](const Image::SharedPtr msg) {
    static int frameID = 0;
    if (!this->collectedCameraInfo) {
      RCLCPP_WARN(this->get_logger(), "Camera Info not collected yet");
      return;
    }
    // Convert ROS Image to OpenCV Image
    cv::Mat image = cv_bridge::toCvCopy(msg, "bgr8")->image;
    Frame f;
    f.stamp = this->now();
    f.frameID = frameID++;
    f.image = image;
    // Create a scope to lock the mutex before accessing the frame queue
    {
      std::lock_guard<std::mutex> lock(this->frameMutex);
      f.K = this->cameraIntrinsics.K.clone();
      f.D = this->cameraIntrinsics.D.clone();
      this->frameQueue.push(std::move(f));
    }
  }
  );

  // Timer for processing camera images
  this->timer = this->create_wall_timer(
    std::chrono::milliseconds(static_cast<int>(1000.0f / this->timerFreq)),
    std::bind(&CameraManager::timerCallback, this)
  );
}

void CameraManager::timerCallback() {
  Frame currentFrame;
  {
    std::lock_guard<std::mutex> lock(this->frameMutex);
    if (!this->frameQueue.empty()) {
      currentFrame = std::move(this->frameQueue.front());
      this->frameQueue.pop();
    }
  }
  FeatureMap features = this->FeatureExtractor(currentFrame);
  Edge odomEdge = this->CameraPoseEstimation(features);
  Edge loopConstraints = this->LoopClosureDetector();
  this->GraphBuilder(odomEdge, loopConstraints);
  this->VisualizeGraph();
}

FeatureMap CameraManager::FeatureExtractor(const Frame& frame) {
  // Implement feature extraction logic using SIFT and ORB.
  cv::Ptr<cv::Feature2D> extractor = cv::SIFT::create();
  FeatureMap featureMap;
  featureMap.frameID = frame.frameID;
  extractor->detectAndCompute(
    frame.image, cv::noArray(), featureMap.keypoints, featureMap.descriptors
  );
  return featureMap;
}

Edge CameraManager::CameraPoseEstimation(const FeatureMap& featureMap) {
  // Implement camera pose estimation logic.
  Edge edge;
  edge.fromID = featureMap.frameID;
  edge.toID = featureMap.frameID + 1; // Example increment
  edge.relativePose = cv::Mat::eye(4, 4, CV_64F); // Placeholder
  return edge;
}

Edge CameraManager::LoopClosureDetector() {
  // Implement loop closure detection logic.
  Edge edge;
  edge.fromID = -1; // Placeholder
  edge.toID = -1; // Placeholder
  return edge;
}

void CameraManager::GraphBuilder(const Edge& estimatedPose, const Edge& loopConstraints) {
  // Implement graph building logic.
}

void CameraManager::VisualizeGraph() {
  // Implement graph visualization logic.
}

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraManager>());
  rclcpp::shutdown();
  return 0;
}