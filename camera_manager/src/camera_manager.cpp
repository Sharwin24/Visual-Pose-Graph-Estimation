#include "camera_manager.hpp"

CameraManager::CameraManager() : Node("camera_manager") {
  RCLCPP_INFO(this->get_logger(), "Camera Manager Node Initialized");

  // Declare parameters.
  this->timerFreq = this->declare_parameter("timer_frequency", 10.0f); // [Hz]
  // Get params from config file.
  this->timerFreq = this->get_parameter("timer_frequency").as_double();


  pathPub = this->create_publisher<nav_msgs::msg::Path>("camera_trajectory", 10);
  pathMsg.header.frame_id = "camera_link"; // or "odom" or other fixed frame

  // Create subscriber to camera info
  this->cameraInfoSub = this->create_subscription<CameraInfo>(
    "camera/camera/color/camera_info", 10,
    [this](const CameraInfo::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(this->frameMutex);
    this->cameraIntrinsics.K = cv::Mat(3, 3, CV_64F, const_cast<double*>(msg->k.data())).clone();
    this->cameraIntrinsics.D = cv::Mat(msg->d.size(), 1, CV_64F, const_cast<double*>(msg->d.data())).clone();
    this->collectedCameraInfo = true;
  }
  );

  // Create subscriber to left camera info.
  this->leftCameraInfoSub = this->create_subscription<CameraInfo>(
    "camera/camera/infra1/camera_info", 10,
    [this](const CameraInfo::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(this->leftFrameMutex);
    this->leftCameraIntrinsics.K = cv::Mat(3, 3, CV_64F, const_cast<double*>(msg->k.data())).clone();
    this->leftCameraIntrinsics.D = cv::Mat(msg->d.size(), 1, CV_64F, const_cast<double*>(msg->d.data())).clone();
    this->collectedLeftCameraInfo = true;
  }
  );

  // Create subscriber to right camera info.
  this->rightCameraInfoSub = this->create_subscription<CameraInfo>(
    "camera/camera/infra2/camera_info", 10,
    [this](const CameraInfo::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(this->rightFrameMutex);
    this->rightCameraIntrinsics.K = cv::Mat(3, 3, CV_64F, const_cast<double*>(msg->k.data())).clone();
    this->rightCameraIntrinsics.D = cv::Mat(msg->d.size(), 1, CV_64F, const_cast<double*>(msg->d.data())).clone();
    this->collectedRightCameraInfo = true;
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

  this->leftCameraSub = this->create_subscription<Image>(
    "camera/camera/infra1/image_rect_raw", 10,
    [this](const Image::SharedPtr msg) {
    static int frameID = 0;
    if (!this->collectedLeftCameraInfo) {
      RCLCPP_WARN(this->get_logger(), "Camera Info not collected yet");
      return;
    }
    // Convert ROS Image to OpenCV Image
    cv::Mat image = cv_bridge::toCvCopy(msg, "bgr8")->image;
    Frame f;
    f.stamp = this->now();
    f.frameID = frameID++;
    f.image = image;
    // Create a scope to lock the mutex before accessing the left frame queue
    {
      std::lock_guard<std::mutex> lock(this->leftFrameMutex);
      f.K = this->cameraIntrinsics.K.clone();
      f.D = this->cameraIntrinsics.D.clone();
      this->leftFrameQueue.push(std::move(f));
    }
  });

  this->rightCameraSub = this->create_subscription<Image>(
    "camera/camera/infra2/image_rect_raw", 10,
    [this](const Image::SharedPtr msg) {
    static int frameID = 0;
    if (!this->collectedRightCameraInfo) {
      RCLCPP_WARN(this->get_logger(), "Camera Info not collected yet");
      return;
    }
    // Convert ROS Image to OpenCV Image
    cv::Mat image = cv_bridge::toCvCopy(msg, "bgr8")->image;
    Frame f;
    f.stamp = this->now();
    f.frameID = frameID++;
    f.image = image;
    // Create a scope to lock the mutex before accessing the right frame queue
    {
      std::lock_guard<std::mutex> lock(this->rightFrameMutex);
      f.K = this->cameraIntrinsics.K.clone();
      f.D = this->cameraIntrinsics.D.clone();
      this->rightFrameQueue.push(std::move(f));
    }
  });



  // Timer for processing camera images
  this->timer = this->create_wall_timer(
    std::chrono::milliseconds(static_cast<int>(1000.0f / this->timerFreq)),
    std::bind(&CameraManager::timerCallback, this)
  );
}

void CameraManager::timerCallback() {
  Frame currentFrame;
  Frame currentFrameLeft;
  Frame currentFrameRight;
  // TODO: remove the extra brackets? and see if this is the correct way to impl.
  {
    std::lock_guard<std::mutex> lock(this->frameMutex);
    if (!this->frameQueue.empty()) {
      currentFrame = std::move(this->frameQueue.front());
      this->frameQueue.pop();
    }
  }
  {
    std::lock_guard<std::mutex> lock(this->leftFrameMutex);
    if (!this->leftFrameQueue.empty()) {
      currentFrameLeft = std::move(this->leftFrameQueue.front());
      this->leftFrameQueue.pop();
    }
  }
  {
    std::lock_guard<std::mutex> lock(this->rightFrameMutex);
    if (!this->rightFrameQueue.empty()) {
      currentFrameRight = std::move(this->rightFrameQueue.front());
      this->rightFrameQueue.pop();
    }
  }

  // TODO: make this a config paramter later.
  bool stereo = true;
  Edge odomEdge;
  if (stereo) {
    StereoFeature stereoFeatures = this->StereoFeatureExtractor(currentFrameLeft, currentFrameRight);
    odomEdge = this->StereoCameraPoseEstimation(stereoFeatures);
  } else {
    Feature features = this->FeatureExtractor(currentFrame);
    odomEdge = this->MonocularCameraPoseEstimation(features);
  }
  // Edge loopConstraints = this->LoopClosureDetector();
  // this->GraphBuilder(odomEdge, loopConstraints);
  // this->VisualizeGraph();
  this->VisulizeTrajectory(odomEdge);
}

void CameraManager::VisulizeTrajectory(Edge odomEdge) {
  if (odomEdge.relativePose.empty() || odomEdge.relativePose.rows != 4 || odomEdge.relativePose.cols != 4) {
    RCLCPP_WARN(this->get_logger(), "[VisulizeTrajectory] Invalid or empty pose matrix, skipping visualization.");
    return;
  }
  geometry_msgs::msg::PoseStamped poseStamped;
  poseStamped.header.stamp = this->now();
  poseStamped.header.frame_id = "camera_link"; // or your chosen frame

  // Extract translation from 4x4 pose matrix
  poseStamped.pose.position.x = odomEdge.relativePose.at<double>(0, 3);
  poseStamped.pose.position.y = odomEdge.relativePose.at<double>(1, 3);
  poseStamped.pose.position.z = odomEdge.relativePose.at<double>(2, 3);

  // Convert rotation matrix to quaternion
  cv::Mat R = odomEdge.relativePose(cv::Rect(0, 0, 3, 3));
  cv::Mat rvec;
  cv::Rodrigues(R, rvec);
  tf2::Quaternion q;
  q.setRPY(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
  poseStamped.pose.orientation.x = q.x();
  poseStamped.pose.orientation.y = q.y();
  poseStamped.pose.orientation.z = q.z();
  poseStamped.pose.orientation.w = q.w();

  // Add to path and publish
  pathMsg.poses.push_back(poseStamped);
  pathMsg.header.stamp = this->now();

  rclcpp::sleep_for(std::chrono::milliseconds(400));
  pathPub->publish(pathMsg);
}

Feature CameraManager::FeatureExtractor(const Frame& frame) {
  // Implement feature extraction logic using SIFT and ORB.
  // TODO: SIFT is slow but accurate than ORB, may need to switch to ORB for real-time applications.
  cv::Ptr<cv::Feature2D> extractor = cv::SIFT::create();
  Feature feature;
  feature.frameID = frame.frameID;

  if (frame.image.empty()) {
    RCLCPP_WARN(this->get_logger(), "[FeatureExtractor] Received empty image for frame ID %d", frame.frameID);
    return feature;
  }
  // TODO: Add left and right image support for stereo cameras.
  extractor->detectAndCompute(
    frame.image, cv::noArray(), feature.keypoints, feature.descriptors
  );
  featureMap[feature.frameID] = feature;
  return feature;
}

StereoFeature CameraManager::StereoFeatureExtractor(const Frame& leftFrame, const Frame& rightFrame) {
  // Implement stereo feature extraction logic.
  StereoFeature stereoFeature;
  stereoFeature.frameID = leftFrame.frameID;

  if (leftFrame.image.empty() || rightFrame.image.empty()) {
    RCLCPP_WARN(this->get_logger(), "[StereoFeatureExtractor] Received empty images for frame ID %d", leftFrame.frameID);
    return stereoFeature;
  }

  cv::Ptr<cv::Feature2D> extractor = cv::SIFT::create();
  extractor->detectAndCompute(leftFrame.image, cv::noArray(), stereoFeature.leftKeypoints, stereoFeature.leftDescriptors);
  extractor->detectAndCompute(rightFrame.image, cv::noArray(), stereoFeature.rightKeypoints, stereoFeature.rightDescriptors);

  stereoFeatureMap[stereoFeature.frameID] = stereoFeature;
  return stereoFeature;
}

Edge CameraManager::MonocularCameraPoseEstimation(const Feature& feature) {
  // Implement camera pose estimation logic.
  if (featureMap.find(feature.frameID) == featureMap.end()) {
    RCLCPP_WARN(this->get_logger(), "[MonocularCameraPoseEstimation] Current frame not found for ID %d", feature.frameID);
    return Edge();
  }

  if (featureMap.find(feature.frameID - 1) == featureMap.end()) {
    RCLCPP_WARN(this->get_logger(), "[MonocularCameraPoseEstimation] Previous frame not found for pose estimation.");
    return Edge();
  }
  Feature prevFeature = featureMap[feature.frameID - 1];

  cv::Mat K = this->cameraIntrinsics.K;
  cv::Mat D = this->cameraIntrinsics.D;

  Edge edge;
  edge.fromID = feature.frameID;
  edge.toID = feature.frameID + 1;

  // Match descriptors
  cv::BFMatcher matcher(cv::NORM_L2);
  std::vector<cv::DMatch> matches;
  matcher.match(prevFeature.descriptors, feature.descriptors, matches);

  // Filter good matches
  double max_dist = 0; double min_dist = 100;
  for (int i = 0; i < matches.size(); i++) {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }
  std::vector<cv::DMatch> good_matches;
  for (int i = 0; i < matches.size(); i++) {
    if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }

  // Extract matched points
  std::vector<cv::Point2f> pts1, pts2;
  for (size_t i = 0; i < good_matches.size(); i++) {
    pts1.push_back(prevFeature.keypoints[good_matches[i].queryIdx].pt);
    pts2.push_back(feature.keypoints[good_matches[i].trainIdx].pt);
  }

  if (pts1.size() < 5 || pts2.size() < 5) {
    edge.relativePose = cv::Mat::eye(4, 4, CV_64F); // Not enough points
    return edge;
  }

  // Estimate essential matrix
  cv::Mat mask;
  cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0, mask);

  // Recover pose
  cv::Mat R, t;
  int inliers = cv::recoverPose(E, pts1, pts2, K, R, t, mask);

  // Build 4x4 transformation matrix
  edge.relativePose = cv::Mat::eye(4, 4, CV_64F);
  R.copyTo(edge.relativePose(cv::Rect(0, 0, 3, 3)));
  t.copyTo(edge.relativePose(cv::Rect(3, 0, 1, 3)));

  return edge;
}

Edge CameraManager::StereoCameraPoseEstimation(const StereoFeature& feature) {

  // Implement stereo camera pose estimation logic.
  if (stereoFeatureMap.find(feature.frameID - 1) == stereoFeatureMap.end()) {
    RCLCPP_WARN(this->get_logger(), "[Stereo Pose Estimation] Previous frame not found.");
    return Edge();
  }
  if (stereoFeatureMap.find(feature.frameID) == stereoFeatureMap.end()) {
    RCLCPP_WARN(this->get_logger(), "[Stereo Pose Estimation] Current frame not found.");
    return Edge();
  }
  StereoFeature prevFeature = stereoFeatureMap[feature.frameID - 1];
  cv::Mat K = this->leftCameraIntrinsics.K; // 3x3 camera matrix
  float stereoBaseline = 0.05;  // distance between left and right cameras [in meters]
  double baseline = stereoBaseline;

  Edge edge;
  edge.fromID = prevFeature.frameID;
  edge.toID = feature.frameID;

  // Match descriptors between previous and current left images
  cv::BFMatcher matcher(cv::NORM_L2);
  std::vector<cv::DMatch> matches;
  matcher.match(prevFeature.leftDescriptors, feature.leftDescriptors, matches);

  // Filter good matches 
  double min_dist = 100;
  for (const auto& m : matches) min_dist = std::min(min_dist, (double)m.distance);
  std::vector<cv::DMatch> good_matches;
  for (const auto& m : matches) {
    if (m.distance <= std::max(2 * min_dist, 30.0)) good_matches.push_back(m);
  }
  // Triangulate 3D points in previous and current frames
  std::vector<cv::Point3f> pts3d_prev, pts3d_curr;
  std::vector<cv::Point2f> pts2d_curr;
  for (const auto& m : good_matches) {
    // Get left/right keypoints for triangulation
    cv::Point2f kpL_prev = prevFeature.leftKeypoints[m.queryIdx].pt;
    cv::Point2f kpR_prev = prevFeature.rightKeypoints[m.queryIdx].pt;
    cv::Point2f kpL_curr = feature.leftKeypoints[m.trainIdx].pt;
    cv::Point2f kpR_curr = feature.rightKeypoints[m.trainIdx].pt;
    // Compute disparity and check validity
    float disp_prev = kpL_prev.x - kpR_prev.x;
    float disp_curr = kpL_curr.x - kpR_curr.x;
    if (disp_prev > 1.0 && disp_curr > 1.0) {
      // Triangulate previous 3D point
      float Z_prev = K.at<double>(0, 0) * baseline / disp_prev;
      float X_prev = (kpL_prev.x - K.at<double>(0, 2)) * Z_prev / K.at<double>(0, 0);
      float Y_prev = (kpL_prev.y - K.at<double>(1, 2)) * Z_prev / K.at<double>(1, 1);
      pts3d_prev.emplace_back(X_prev, Y_prev, Z_prev);
      // Triangulate current 3D point (for PnP, we only need 2D in current frame)
      pts2d_curr.emplace_back(kpL_curr);
    }
  }
  if (pts3d_prev.size() < 5) {
    edge.relativePose = cv::Mat::eye(4, 4, CV_64F);
    return edge;
  }

  // Solve PnP: to estimate the rotation and translation between the previous and current frames using the 3D-2D correspondences.
  cv::Mat rvec, tvec, inliers;
  cv::solvePnPRansac(pts3d_prev, pts2d_curr, K, cv::Mat(), rvec, tvec, false, 100, 8.0, 0.99, inliers);

  // Convert rvec to rotation matrix
  cv::Mat R;
  cv::Rodrigues(rvec, R);

  // Build 4x4 transformation matrix
  edge.relativePose = cv::Mat::eye(4, 4, CV_64F);
  R.copyTo(edge.relativePose(cv::Rect(0, 0, 3, 3)));
  tvec.copyTo(edge.relativePose(cv::Rect(3, 0, 1, 3)));
  return edge;
}

Edge CameraManager::LoopClosureDetector() {
  // TODO: Implement loop closure detection logic.
  Edge edge;
  edge.fromID = -1; // Placeholder
  edge.toID = -1; // Placeholder
  return edge;
}

void CameraManager::GraphBuilder(const Edge& estimatedPose, const Edge& loopConstraints) {
  // TODO: Implement graph building logic.
}

void CameraManager::VisualizeGraph() {
  // TODO: Implement graph visualization logic.
}

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraManager>());
  rclcpp::shutdown();
  return 0;
}