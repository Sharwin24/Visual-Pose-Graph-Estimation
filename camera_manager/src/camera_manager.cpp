#include "camera_manager.hpp"

CameraManager::CameraManager() : Node("camera_manager") {
  RCLCPP_INFO(this->get_logger(), "Camera Manager Node Initialized");

  // Declare parameters.
  this->timerFreq = this->declare_parameter("timer_frequency", 10.0); // [Hz] Timer for managing pose-graph
  this->useStereoCamera = this->declare_parameter("use_stereo_camera", true); // Use stereo camera by default
  this->stereoBaseline = this->declare_parameter("stereo_baseline", 0.05); // [m] Default baseline distance between stereo cameras
  this->featureExtractionMethod = this->declare_parameter("feature_extraction_method", "SIFT"); // Default feature extraction method
  // Get params from config file.
  this->timerFreq = this->get_parameter("timer_frequency").as_double();
  this->useStereoCamera = this->get_parameter("use_stereo_camera").as_bool();
  this->stereoBaseline = this->get_parameter("stereo_baseline").as_double();
  this->featureExtractionMethod = this->get_parameter("feature_extraction_method").as_string();

  RCLCPP_INFO(this->get_logger(), "Using %s camera", this->useStereoCamera ? "stereo" : "monocular");
  RCLCPP_INFO(this->get_logger(), "Feature extraction method: %s", this->featureExtractionMethod.c_str());

  // Setup Path publisher for camera trajectory
  this->cameraEstimatePathPub = this->create_publisher<Path>("camera_trajectory", 10);
  this->cameraPoseEstimatePath.header.frame_id = "camera_link";

  // Setup Image publisher for feature visualization
  this->featureImagePub = this->create_publisher<ImageMsg>("camera/feature_image", 10);

  // Setup Pose-Graph marker publisher.
  this->poseGraphVizPub = this->create_publisher<MarkerArray>("pose_graph_markers", 10);

  // Setup tf broadcaster
  this->tfBroadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);


  // Create subscribers for stereo and monocular cameras
  if (this->useStereoCamera) {
    // Setup left and right camera subscribers
    this->leftCameraSub = std::make_shared<message_filters::Subscriber<ImageMsg>>(this, "camera/camera/infra1/image_rect_raw");
    this->leftCameraInfoSub = std::make_shared<message_filters::Subscriber<CameraInfoMsg>>(this, "camera/camera/infra1/camera_info");
    this->rightCameraSub = std::make_shared<message_filters::Subscriber<ImageMsg>>(this, "camera/camera/infra2/image_rect_raw");
    this->rightCameraInfoSub = std::make_shared<message_filters::Subscriber<CameraInfoMsg>>(this, "camera/camera/infra2/camera_info");
    // Synchronize stereo camera messages
    this->stereoSync = std::make_shared<message_filters::Synchronizer<StereoSyncPolicy>>(
      StereoSyncPolicy(10), *leftCameraSub, *leftCameraInfoSub, *rightCameraSub, *rightCameraInfoSub
    );
    this->stereoSync->registerCallback(
      std::bind(
        &CameraManager::synchronizedStereoCallback,
        this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4
      )
    );
  } else {
    // Setup monocular camera subscriber
    this->cameraSub = std::make_shared<message_filters::Subscriber<ImageMsg>>(this, "camera/camera/color/image_raw");
    this->cameraInfoSub = std::make_shared<message_filters::Subscriber<CameraInfoMsg>>(this, "camera/camera/color/camera_info");
    // Synchronize monocular camera messages
    this->monoSync = std::make_shared<message_filters::Synchronizer<MonoSyncPolicy>>(
      MonoSyncPolicy(10), *cameraSub, *cameraInfoSub
    );
    this->monoSync->registerCallback(
      std::bind(
        &CameraManager::synchronizedMonocularCallback,
        this, std::placeholders::_1, std::placeholders::_2
      )
    );
  }

  // Initialize current estimate
  this->currentPose = cv::Mat::eye(4, 4, CV_64F); // SE(3) identity matrix

  // Initialize PoseGraph
  this->initializePoseGraph();

  // Publish a static transform from the world to odom frame
  static tf2_ros::StaticTransformBroadcaster staticBroadcaster = tf2_ros::StaticTransformBroadcaster(this);

  // Publish a static transform from world to base_link
  geometry_msgs::msg::TransformStamped baseLinkTransform;
  baseLinkTransform.header.stamp = this->now();
  baseLinkTransform.header.frame_id = "world";
  baseLinkTransform.child_frame_id = "odom";
  baseLinkTransform.transform.translation.x = 0.0;
  baseLinkTransform.transform.translation.y = 0.0;
  baseLinkTransform.transform.translation.z = 0.0;
  tf2::Quaternion q;
  q.setEuler(M_PI / 2.0, 0.0, -M_PI / 2.0);
  baseLinkTransform.transform.rotation.x = q.x();
  baseLinkTransform.transform.rotation.y = q.y();
  baseLinkTransform.transform.rotation.z = q.z();
  baseLinkTransform.transform.rotation.w = q.w();
  staticBroadcaster.sendTransform(baseLinkTransform);

  // Publish a transform at the origin of the world frame
  geometry_msgs::msg::TransformStamped worldTransform;
  worldTransform.header.stamp = this->now();
  worldTransform.header.frame_id = "odom";
  worldTransform.child_frame_id = "camera_link";
  worldTransform.transform.translation.x = 0.0;
  worldTransform.transform.translation.y = 0.0;
  worldTransform.transform.translation.z = 0.0;
  worldTransform.transform.rotation.x = 0.0;
  worldTransform.transform.rotation.y = 0.0;
  worldTransform.transform.rotation.z = 0.0;
  worldTransform.transform.rotation.w = 1.0;
  this->tfBroadcaster->sendTransform(worldTransform);

  // Setup timer for pose-graph management and visualization
  this->timer = this->create_wall_timer(
    std::chrono::milliseconds(static_cast<int>(1000 / this->timerFreq)),
    std::bind(&CameraManager::timerCallback, this)
  );
}

void CameraManager::timerCallback() {

  // Call the pose graph opti - figure the frequency of this timer
  // Call after loop closure detection or after every N frames/
  // optimizePoseGraph()


}

void CameraManager::synchronizedStereoCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr left_image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr left_info_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr right_image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr right_info_msg) {
  static int frameID = 0;
  Frame currentLeftFrame;
  Frame currentRightFrame;

  // Convert ROS Image to OpenCV Image
  currentLeftFrame.image = cv_bridge::toCvCopy(left_image_msg, "bgr8")->image;
  currentRightFrame.image = cv_bridge::toCvCopy(right_image_msg, "bgr8")->image;

  // Extract intrinsics
  currentLeftFrame.intrinsics.K = cv::Mat(3, 3, CV_64F, const_cast<double*>(left_info_msg->k.data())).clone();
  currentLeftFrame.intrinsics.D = cv::Mat(left_info_msg->d.size(), 1, CV_64F, const_cast<double*>(left_info_msg->d.data())).clone();
  currentRightFrame.intrinsics.K = cv::Mat(3, 3, CV_64F, const_cast<double*>(right_info_msg->k.data())).clone();
  currentRightFrame.intrinsics.D = cv::Mat(right_info_msg->d.size(), 1, CV_64F, const_cast<double*>(right_info_msg->d.data())).clone();
  this->leftCameraIntrinsics.K = currentLeftFrame.intrinsics.K.clone();
  this->leftCameraIntrinsics.D = currentLeftFrame.intrinsics.D.clone();
  this->rightCameraIntrinsics.K = currentRightFrame.intrinsics.K.clone();
  this->rightCameraIntrinsics.D = currentRightFrame.intrinsics.D.clone();
  // Assign timestamps and frame IDs
  currentLeftFrame.stamp = left_image_msg->header.stamp;
  currentRightFrame.stamp = right_image_msg->header.stamp;
  currentLeftFrame.frameID = frameID;
  currentRightFrame.frameID = frameID;
  frameID++; // Increment for next set of frames

  if (currentLeftFrame.image.empty() || currentRightFrame.image.empty() || currentLeftFrame.intrinsics.K.empty()) {
    RCLCPP_WARN(this->get_logger(), "Received empty stereo images or camera info.");
  }

  // 1. Feature Extraction
  StereoFeature currentStereoFeature = this->StereoFeatureExtractor(currentLeftFrame, currentRightFrame);

  // Create a copy of the left image with overlaid features for visualization
  cv::Mat features;
  currentLeftFrame.image.copyTo(features);
  if (!currentStereoFeature.leftKeypoints.empty()) {
    cv::drawKeypoints(
      currentLeftFrame.image, currentStereoFeature.leftKeypoints, features,
      cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );
  }
  // Create ROS Image message to publish
  auto header = std_msgs::msg::Header();
  header.stamp = this->now();
  header.frame_id = "camera_link";
  auto featureImageMsg = cv_bridge::CvImage(
    header, "bgr8", features
  ).toImageMsg();
  // Publish the feature image
  this->featureImagePub->publish(*featureImageMsg);

  // Update previous features if we have not processed any frames yet
  if (this->previousStereoFeature.frameID == 0 && currentStereoFeature.frameID > 0) {
    this->previousStereoFeature = currentStereoFeature;
    return;
  }

  // 2. Odometry (Pose) Estimation 
  Edge odomEdge = this->StereoCameraPoseEstimation(currentStereoFeature);
  this->previousStereoFeature = currentStereoFeature; // Update previous feature for next iteration

  // 3. Pose-Graph Management
  if (!odomEdge.relativePose.empty()) {
    // Update current pose estimate
    {
      std::lock_guard<std::mutex> lock(this->poseMutex);
      this->currentPose = odomEdge.relativePose * this->currentPose;
      // TODO: Test if this is fine.
      // frame id for left and right should be the same: may need to make a non-deep copy.
      this->addNode(currentLeftFrame.frameID, this->currentPose);
    }
    this->addEdge(odomEdge);
    // Visualize the current camera pose
    this->UpdateCameraPoseVisualization();
    // Visualize the pose graph
    this->visulizePoseGraph();
  }

  // 4. Loop-Closure Detection
  // TODO: This should be managed carefully, in the timer or separate thread
  // For now this can be left out as we implement the rest of the pipeline
}

void CameraManager::synchronizedMonocularCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr image_msg,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr info_msg) {
  static int frameID = 0;
  Frame currentFrame;
  // Convert ROS Image to OpenCV Image
  currentFrame.image = cv_bridge::toCvCopy(image_msg, "bgr8")->image;

  // Extract intrinsics
  currentFrame.intrinsics.K = cv::Mat(3, 3, CV_64F, const_cast<double*>(info_msg->k.data())).clone();
  currentFrame.intrinsics.D = cv::Mat(info_msg->d.size(), 1, CV_64F, const_cast<double*>(info_msg->d.data())).clone();
  this->cameraIntrinsics.K = currentFrame.intrinsics.K.clone();
  this->cameraIntrinsics.D = currentFrame.intrinsics.D.clone();
  currentFrame.stamp = image_msg->header.stamp;
  currentFrame.frameID = frameID++; // Increment frame ID
  if (currentFrame.image.empty() || currentFrame.intrinsics.K.empty()) {
    RCLCPP_WARN(this->get_logger(), "Received empty monocular image or camera info.");
  }

  // 1. Feature Extraction
  Feature currentFeature = this->MonocularFeatureExtractor(currentFrame);

  // Create a copy of the image with overlaid features for visualization
  cv::Mat features;
  currentFrame.image.copyTo(features);
  if (!currentFeature.keypoints.empty()) {
    cv::drawKeypoints(
      currentFrame.image, currentFeature.keypoints, features,
      cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );
  }
  // Create ROS Image message to publish
  auto header = std_msgs::msg::Header();
  header.stamp = this->now();
  header.frame_id = "camera_link";
  auto featureImageMsg = cv_bridge::CvImage(
    header, "bgr8", features
  ).toImageMsg();
  // Publish the feature image
  this->featureImagePub->publish(*featureImageMsg);

  // Update previous features if we have not processed any frames yet
  if (this->previousFeature.frameID == 0 && currentFeature.frameID > 0) {
    this->previousFeature = currentFeature;
    return;
  }

  // 2. Odometry (Pose) Estimation
  Edge odomEdge = this->MonocularCameraPoseEstimation(currentFeature);
  this->previousFeature = currentFeature; // Update previous feature for next iteration

  // 3. Pose-Graph Management
  if (!odomEdge.relativePose.empty()) {
    {
      std::lock_guard<std::mutex> lock(this->poseMutex);
      this->currentPose = odomEdge.relativePose * this->currentPose;
      // TODO: Test if this is fine.
      // frame id for left and right should be the same.
      this->addNode(currentFrame.frameID, this->currentPose);
    }
    // Update current pose estimate
    this->addEdge(odomEdge);
    // Visualize the current camera pose
    this->UpdateCameraPoseVisualization();
    // Visualize the pose graph
    this->visulizePoseGraph();
  }

  // 4. Loop-Closure Detection
  // TODO: This should be managed carefully, in the timer or separate thread
  // For now this can be left out as we implement the rest of the pipeline
}

Feature CameraManager::MonocularFeatureExtractor(const Frame& frame) {
  Feature feature;
  feature.frameID = frame.frameID;
  if (frame.image.empty()) {
    RCLCPP_WARN(this->get_logger(), "[FeatureExtractor] Received empty image for frame ID %d", frame.frameID);
    return feature;
  }
  // cv::Ptr<cv::Feature2D> extractor = cv::SIFT::create();
  if (this->featureExtractionMethod == "ORB") {
    cv::Ptr<cv::Feature2D> extractor = cv::ORB::create();
    extractor->detectAndCompute(frame.image, cv::noArray(), feature.keypoints, feature.descriptors);
  } else if (this->featureExtractionMethod == "SIFT") {
    // Default to SIFT if no valid method is specified
    cv::Ptr<cv::Feature2D> extractor = cv::SIFT::create();
    extractor->detectAndCompute(frame.image, cv::noArray(), feature.keypoints, feature.descriptors);
  } else {
    RCLCPP_WARN(this->get_logger(), "[FeatureExtractor] Unsupported feature extraction method: %s", this->featureExtractionMethod.c_str());
    return feature; // Return empty feature
  }
  return feature;
}

StereoFeature CameraManager::StereoFeatureExtractor(const Frame& leftFrame, const Frame& rightFrame) {
  StereoFeature stereoFeature;
  stereoFeature.frameID = leftFrame.frameID;

  if (leftFrame.image.empty() || rightFrame.image.empty()) {
    RCLCPP_WARN(this->get_logger(), "[StereoFeatureExtractor] Received empty images for frame ID %d", leftFrame.frameID);
    return stereoFeature;
  }
  if (this->featureExtractionMethod == "ORB") {
    cv::Ptr<cv::Feature2D> extractor = cv::ORB::create();
    extractor->detectAndCompute(leftFrame.image, cv::noArray(), stereoFeature.leftKeypoints, stereoFeature.leftDescriptors);
    extractor->detectAndCompute(rightFrame.image, cv::noArray(), stereoFeature.rightKeypoints, stereoFeature.rightDescriptors);
    return stereoFeature;
  } else if (this->featureExtractionMethod == "SIFT") {
    cv::Ptr<cv::Feature2D> extractor = cv::SIFT::create(
      0, // nfeatures
      3, // nOctaveLayers
      0.04, // contrastThreshold
      10, // edgeThreshold
      1.6 // sigma
    );
    extractor->detectAndCompute(leftFrame.image, cv::noArray(), stereoFeature.leftKeypoints, stereoFeature.leftDescriptors);
    extractor->detectAndCompute(rightFrame.image, cv::noArray(), stereoFeature.rightKeypoints, stereoFeature.rightDescriptors);
    return stereoFeature;
  } else {
    RCLCPP_WARN(this->get_logger(), "[StereoFeatureExtractor] Unsupported feature extraction method: %s", this->featureExtractionMethod.c_str());
    return stereoFeature; // Return empty stereo feature
  }
}

Edge CameraManager::MonocularCameraPoseEstimation(const Feature& newFeature) {
  // This function will only be called if there's a valid previous feature
  cv::Mat K = this->cameraIntrinsics.K;
  cv::Mat D = this->cameraIntrinsics.D;

  Edge edge;
  edge.fromID = newFeature.frameID;
  edge.toID = newFeature.frameID + 1;

  // Match descriptors
  cv::BFMatcher matcher(cv::NORM_L2);
  std::vector<cv::DMatch> matches;
  matcher.match(previousFeature.descriptors, newFeature.descriptors, matches);

  // Filter good matches
  double max_dist = 0;
  double min_dist = 100;
  for (unsigned int i = 0; i < matches.size(); i++) {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }
  std::vector<cv::DMatch> good_matches;
  for (unsigned int i = 0; i < matches.size(); i++) {
    if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }

  // Extract matched points
  std::vector<cv::Point2f> pts1, pts2;
  for (size_t i = 0; i < good_matches.size(); i++) {
    pts1.push_back(previousFeature.keypoints[good_matches[i].queryIdx].pt);
    pts2.push_back(newFeature.keypoints[good_matches[i].trainIdx].pt);
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

Edge CameraManager::StereoCameraPoseEstimation(const StereoFeature& newFeature) {
  // This function will only be called if there's a valid previous stereo feature

  // Create the edge between the previous and current stereo features
  Edge edge;
  edge.fromID = previousStereoFeature.frameID;
  edge.toID = newFeature.frameID;

  // Match descriptors between previous and current left images
  cv::BFMatcher matcher(cv::NORM_L2);
  std::vector<cv::DMatch> matches;
  matcher.match(previousStereoFeature.leftDescriptors, newFeature.leftDescriptors, matches);

  // Filter good matches 
  double min_dist = DBL_MAX;
  for (const auto& m : matches) {
    min_dist = std::min(min_dist, (double)m.distance);
  }
  std::vector<cv::DMatch> good_matches;
  for (const auto& m : matches) {
    if (m.distance <= std::max(2 * min_dist, 30.0)) {
      good_matches.push_back(m);
    }
  }
  // If too few correspondences, bail out with identity
  if (good_matches.size() < 5) {
    edge.relativePose = cv::Mat::eye(4, 4, CV_64F);
    return edge;
  }

  const cv::Mat K = this->leftCameraIntrinsics.K; // 3x3 camera matrix
  const cv::Mat D = this->leftCameraIntrinsics.D; // Distortion coefficients
  const double sbl = this->stereoBaseline; // Stereo baseline
  cv::Mat I3 = cv::Mat::eye(3, 3, CV_64F); // 3x3 identity matrix

  // Build projection matrices for the PREVIOUS stereo pair:
  // Let the left camera pose be the “reference” (identity). Then the right
  // camera is translated along +X by “baseline” in that same coord‐frame.
  //
  //    P_L = K * [I | 0]
  //    P_R = K * [I | t],   where t = (-baseline, 0, 0)^T

  // Left Projection: [I | 0]
  cv::Mat P_L = cv::Mat::zeros(3, 4, CV_64F);
  I3.copyTo(P_L(cv::Rect(0, 0, 3, 3))); // Copy identity matrix
  P_L = K * P_L; // Apply intrinsic matrix
  // Right Projection: [I | t] with t = (-baseline, 0, 0)^T
  cv::Mat P_R = cv::Mat::zeros(3, 4, CV_64F);
  I3.copyTo(P_R(cv::Rect(0, 0, 3, 3))); // Copy identity matrix
  // translation vector t=(-baseline, 0, 0)^T
  P_R.at<double>(0, 3) = -sbl; // Set translation in X direction
  P_R.at<double>(1, 3) = 0;
  P_R.at<double>(2, 3) = 0;
  P_R = K * P_R; // Apply intrinsic matrix

  // Collect 2D points from the previous stereo pair, according to good_matches
  std::vector<cv::Point2f> ptsL_prev, ptsR_prev;
  // Also collect corresponding 2D point in CURRENT left image for PnP:
  std::vector<cv::Point2f> pts2d_curr;
  ptsL_prev.reserve(good_matches.size());
  ptsR_prev.reserve(good_matches.size());
  pts2d_curr.reserve(good_matches.size());

  for (size_t i = 0; i < good_matches.size(); ++i) {
    const cv::DMatch& m = good_matches[i];
    // Previous‐frame LEFT keypoint  (matched by m.queryIdx)
    ptsL_prev.emplace_back(previousStereoFeature.leftKeypoints[m.queryIdx].pt);
    // Previous‐frame RIGHT keypoint (assuming rightKeypoints is index‐aligned
    // with leftKeypoints; i.e.  leftKeypoints[i] ↔ rightKeypoints[i] )
    ptsR_prev.emplace_back(previousStereoFeature.rightKeypoints[m.queryIdx].pt);
    // Current‐frame LEFT keypoint (for 2D side of PnP)
    pts2d_curr.emplace_back(newFeature.leftKeypoints[m.trainIdx].pt);
  }

  // Triangulate all “previous‐frame” correspondences in one batch:
  cv::Mat points4DH; // 4×N, each column = [X, Y, Z, W]^T in homogeneous coords
  cv::triangulatePoints(P_L, P_R, ptsL_prev, ptsR_prev, points4DH);

  // Convert homogeneous -> Euclidean and fill pts3d_prev
  std::vector<cv::Point3f> pts3d_prev;
  pts3d_prev.reserve(points4DH.cols);
  for (int c = 0; c < points4DH.cols; ++c) {
    float X = points4DH.at<float>(0, c);
    float Y = points4DH.at<float>(1, c);
    float Z = points4DH.at<float>(2, c);
    float W = points4DH.at<float>(3, c);
    if (std::abs(W) < 1e-6) { // Avoid division by zero by removing degenerate points
      continue;
    }
    pts3d_prev.emplace_back(float(X / W),
      float(Y / W),
      float(Z / W));
  }

  // If after removing degenerate points we have too few 3D points:
  if (pts3d_prev.size() < 5) {
    edge.relativePose = cv::Mat::eye(4, 4, CV_64F);
    return edge;
  }

  // Solve PnP: to estimate the rotation and translation between the previous and current frames using the 3D-2D correspondences.
  cv::Mat rvec, tvec, inliers;
  const bool useExtrinsicGuess = false;
  const int iterationsCount = 100; // Number of RANSAC iterations
  const float reprojectionError = 8.0; // Maximum reprojection error
  const double confidence = 0.99; // Confidence level for RANSAC
  const int flags = cv::SOLVEPNP_EPNP;
  bool success = cv::solvePnPRansac(
    pts3d_prev, // 3D points in previous frame
    pts2d_curr, // 2D points in current frame
    K, D, // Camera intrinsic parameters
    rvec, tvec, // Output rotation and translation vector
    useExtrinsicGuess, iterationsCount, reprojectionError, confidence, // RANSAC parameters
    inliers, // Output inliers mask
    flags
  );

  if (!success || inliers.rows < 4) {
    RCLCPP_ERROR(this->get_logger(), "[StereoCameraPoseEstimation] PnP failed for frame ID %d", newFeature.frameID);
    edge.relativePose = cv::Mat::eye(4, 4, CV_64F); // Return identity if PnP fails
    return edge;
  }

  // Convert rvec to rotation matrix
  cv::Mat Rcurr;
  cv::Rodrigues(rvec, Rcurr);

  // Build 4x4 transformation matrix
  edge.relativePose = cv::Mat::eye(4, 4, CV_64F);
  Rcurr.copyTo(edge.relativePose(cv::Rect(0, 0, 3, 3)));
  tvec.copyTo(edge.relativePose(cv::Rect(3, 0, 1, 3)));
  return edge;
}

void CameraManager::UpdateCameraPoseVisualization() {
  // Lock the pose mutex since we we are accessing the current pose estimate
  std::lock_guard<std::mutex> lock(this->poseMutex);

  if (this->currentPose.empty() || this->currentPose.rows != 4 || this->currentPose.cols != 4) {
    RCLCPP_WARN(this->get_logger(), "[VisulizeTrajectory] Invalid or empty pose matrix, skipping visualization.");
    return;
  }

  // Create a PoseStamped msg to visualize the camera pose trajectory
  PoseStamped poseStamped;
  poseStamped.header.stamp = this->now();
  poseStamped.header.frame_id = "camera_link";

  // Extract translation from 4x4 pose matrix
  poseStamped.pose.position.x = this->currentPose.at<double>(0, 3);
  poseStamped.pose.position.y = this->currentPose.at<double>(1, 3);
  poseStamped.pose.position.z = this->currentPose.at<double>(2, 3);

  // TODO: Use Eigen to do this conversion more robustly (cv::cv2eigen function)
  // Convert rotation matrix to quaternion
  cv::Mat R = this->currentPose(cv::Rect(0, 0, 3, 3));
  cv::Mat rvec;
  cv::Rodrigues(R, rvec);
  tf2::Quaternion q;
  const double xRot = rvec.at<double>(0);
  const double yRot = rvec.at<double>(1);
  const double zRot = rvec.at<double>(2);
  q.setRPY(zRot, yRot, xRot); // Note: tf2 uses z-y-x order for RPY
  q.normalize();
  poseStamped.pose.orientation.x = q.x();
  poseStamped.pose.orientation.y = q.y();
  poseStamped.pose.orientation.z = q.z();
  poseStamped.pose.orientation.w = q.w();

  // Add to path and publish
  this->cameraPoseEstimatePath.poses.push_back(poseStamped);
  this->cameraPoseEstimatePath.header.stamp = this->now();
  this->cameraEstimatePathPub->publish(cameraPoseEstimatePath);

  // Broadcast the camera pose as a TF transform
  geometry_msgs::msg::TransformStamped transformStamped;
  transformStamped.header.stamp = this->now();
  transformStamped.header.frame_id = "odom";
  transformStamped.child_frame_id = "camera_link";
  transformStamped.transform.translation.x = poseStamped.pose.position.x;
  transformStamped.transform.translation.y = poseStamped.pose.position.y;
  transformStamped.transform.translation.z = poseStamped.pose.position.z;
  transformStamped.transform.rotation.x = poseStamped.pose.orientation.x;
  transformStamped.transform.rotation.y = poseStamped.pose.orientation.y;
  transformStamped.transform.rotation.z = poseStamped.pose.orientation.z;
  transformStamped.transform.rotation.w = poseStamped.pose.orientation.w;
  this->tfBroadcaster->sendTransform(transformStamped);
}

void CameraManager::initializePoseGraph() {
  // TODO: Can work with other options g20/GTSAM/Ceres.
  // Nodes are camera poses, edges are relative transformations between them.
  RCLCPP_INFO(this->get_logger(), "Initializing Pose Graph");
  optimizer = std::make_unique<g2o::SparseOptimizer>();
  // Set up the linear solver and block solver
  auto linearSolver = std::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>>();
  auto blockSolver = std::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
  auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
  optimizer->setAlgorithm(solver);
  optimizer->setVerbose(false); // Set true for debug output
}

void CameraManager::addNode(int frameID, const cv::Mat& currentPose) {
  std::lock_guard<std::mutex> lock(this->poseGraphMutex);

  // Check if node already exists
  if (optimizer->vertex(frameID) != nullptr) return;

  // Create a new SE3 vertex
  auto* v = new g2o::VertexSE3();
  v->setId(frameID);

  // Convert cv::Mat (4x4) to Eigen::Isometry3d for initial estimate
  Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
  if (!currentPose.empty() && currentPose.rows == 4 && currentPose.cols == 4) {
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        mat(r, c) = currentPose.at<double>(r, c);
  }
  Eigen::Isometry3d pose(mat);
  v->setEstimate(pose);

  // Fix the first node to anchor the graph
  if (frameID == 0) v->setFixed(true);

  optimizer->addVertex(v);
}

void CameraManager::addEdge(const Edge& edge) {
  if (edge.relativePose.empty() || edge.fromID == edge.toID) {
    RCLCPP_WARN(this->get_logger(), "Invalid edge, skipping addition to pose graph.");
    return;
  }

  std::lock_guard<std::mutex> lock(this->poseGraphMutex);

  // Create SE3 edge
  auto* e = new g2o::EdgeSE3();
  e->setVertex(0, optimizer->vertex(edge.fromID));
  e->setVertex(1, optimizer->vertex(edge.toID));

  // Convert cv::Mat (4x4) to Eigen::Isometry3d
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      T(r, c) = edge.relativePose.at<double>(r, c);
  Eigen::Isometry3d relPose(T);

  e->setMeasurement(relPose);

  // TODO: Set information matrix (identity for now, tune as needed)
  e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());

  optimizer->addEdge(e);

  // RCLCPP_INFO(
  //   this->get_logger(),
  //   "Added edge to pose graph: from %d to %d",
  //   edge.fromID, edge.toID
  // );
}

void CameraManager::optimizePoseGraph() {
  // TODO: Need to call this in timer.
  // This function optimizes the graph to refine the camera poses in least squares sense.
  // And updates the current pose estimate based on the optimized graph.: this can be optional, config variable based.

  std::lock_guard<std::mutex> lock(this->poseGraphMutex);
  RCLCPP_INFO(this->get_logger(), "Optimizing Pose Graph");

  int maxIterations = 20;
  optimizer->initializeOptimization();
  int result = optimizer->optimize(maxIterations);

  if (result > 0) {
    RCLCPP_INFO(this->get_logger(), "Pose graph optimization finished successfully.");
  } else {
    RCLCPP_WARN(this->get_logger(), "Pose graph optimization failed or did not converge.");
  }

  // Optionally update currentPose from the last vertex
  if (!optimizer->vertices().empty()) {
    int lastId = optimizer->vertices().begin()->first;
    auto* v = dynamic_cast<g2o::VertexSE3*>(optimizer->vertex(lastId));
    if (v) {
      Eigen::Isometry3d est = v->estimate();
      Eigen::Matrix4d mat = est.matrix();
      cv::Mat cvPose(4, 4, CV_64F);
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          cvPose.at<double>(r, c) = mat(r, c);
      {
        std::lock_guard<std::mutex> poseLock(this->poseMutex);
        this->currentPose = cvPose.clone();
      }
    }
  }
  RCLCPP_INFO(this->get_logger(), "Pose graph optimization is not yet implemented.");
}

void CameraManager::visulizePoseGraph() {
  // Use the Global Pose Graph [optimizer], the nodes as point markers and edge as line markers.
  // Publish all the markers in a single marker array.
  MarkerArray poseGraphMarkers;
  poseGraphMarkers.markers.clear();
  std::lock_guard<std::mutex> lock(this->poseGraphMutex);
  for (const auto& vertexPair : this->optimizer->vertices()) {
    int id = vertexPair.first;
    auto* v = dynamic_cast<g2o::VertexSE3*>(vertexPair.second);
    if (!v) continue;

    // Create a marker for the vertex
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "odom";
    marker.header.stamp = this->now();
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = v->estimate().translation().x();
    marker.pose.position.y = v->estimate().translation().y();
    marker.pose.position.z = v->estimate().translation().z();
    Eigen::Quaterniond q(v->estimate().rotation());
    marker.pose.orientation.x = q.x();
    marker.pose.orientation.y = q.y();
    marker.pose.orientation.z = q.z();
    marker.pose.orientation.w = q.w();
    marker.scale.x = 0.06; // Sphere radius
    marker.scale.y = 0.06;
    marker.scale.z = 0.06;
    marker.color.r = 0.0f;
    marker.color.g = 1.0f; // Green for vertices
    marker.color.b = 0.0f;
    marker.color.a = 1.0f; // Fully opaque
    poseGraphMarkers.markers.push_back(marker);
  }
  // Create edges between this vertex and all connected vertices
  for (const auto& edgePtr : this->optimizer->edges()) {
    auto* edge = dynamic_cast<g2o::EdgeSE3*>(edgePtr);
    if (!edge) continue;

    auto* v1 = dynamic_cast<g2o::VertexSE3*>(edge->vertices()[0]);
    auto* v2 = dynamic_cast<g2o::VertexSE3*>(edge->vertices()[1]);
    if (!v1 || !v2) continue;

    geometry_msgs::msg::Point startPoint, endPoint;
    startPoint.x = v1->estimate().translation().x();
    startPoint.y = v1->estimate().translation().y();
    startPoint.z = v1->estimate().translation().z();

    endPoint.x = v2->estimate().translation().x();
    endPoint.y = v2->estimate().translation().y();
    endPoint.z = v2->estimate().translation().z();

    visualization_msgs::msg::Marker lineMarker;
    lineMarker.header.frame_id = "odom";
    lineMarker.header.stamp = this->now();
    lineMarker.id = v1->id() * 10000 + v2->id(); // Unique ID for the edge
    lineMarker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    lineMarker.action = visualization_msgs::msg::Marker::ADD;
    lineMarker.pose.orientation.w = 1.0;
    lineMarker.scale.x = 0.02;
    lineMarker.color.r = 1.0f;
    lineMarker.color.g = 0.0f;
    lineMarker.color.b = 0.0f;
    lineMarker.color.a = 1.0f;
    lineMarker.points.push_back(startPoint);
    lineMarker.points.push_back(endPoint);

    poseGraphMarkers.markers.push_back(lineMarker);
  }

  this->poseGraphVizPub->publish(poseGraphMarkers);
}

Edge CameraManager::LoopClosureDetector() {
  // TODO: Implement loop closure detection logic.
  Edge edge;
  return edge;
}

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraManager>());
  rclcpp::shutdown();
  return 0;
}