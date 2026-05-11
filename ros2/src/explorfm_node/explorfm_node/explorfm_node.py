import time
import torch
import torch.nn.functional as F

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from explorfm_msgs.srv import ScoreVisualFrontiers, ScoreTraversability

from explorfm.explorfm_model import ExploRFMInference

bridge = CvBridge()

CAMERA_TOPIC = "/boxi/alphasense/front_center/image_raw/compressed"
CAMERA_INFO_TOPIC = "/boxi/alphasense/front_center/camera_info"

VISUAL_FRONTIER_CONFIDENCE_THRESHOLD = 0.6

class ExplorfmNode(Node):
    """ROS2 Node for ExplorfM visual frontier and traversability scoring services."""

    def __init__(self):
        super().__init__('explorfm_node')
        self.get_logger().info('ExplorfmNode initialized')

        # Create service servers
        self.score_visual_frontiers_srv = self.create_service(
            ScoreVisualFrontiers,
            'score_visual_frontiers',
            self.score_visual_frontiers_callback
        )

        self.score_traversability_srv = self.create_service(
            ScoreTraversability,
            'score_traversability',
            self.score_traversability_callback
        )

        self.get_logger().info('Service servers created')

        # Initialize camera-related instance variables
        self.camera_info = None
        self.calibration_matrix = None
        self.distortion = None
        self.latest_frame = None

        # Create QoS profile for best effort subscriptions with keep_last=1
        best_effort_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        # Subscribe to camera info and image topics
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            qos_profile=best_effort_qos
        )

        self.image_sub = self.create_subscription(
            CompressedImage,
            CAMERA_TOPIC,
            self.image_callback,
            qos_profile=best_effort_qos
        )

        self.get_logger().info('Camera subscriptions created')

        print("Initializing ExploRFM model...")
        self.explorfm_model: ExploRFMInference = ExploRFMInference(
            frontier_ckpt="model_ckpts/frontier_head.ckpt",
            traversability_ckpt="model_ckpts/trav_head.ckpt",
            model_version="c-radio_v3-b",
            adaptor_version=None, # We aren't doing object similarity checks so we don't need siglip.
            radio_dim=768,
            static_scale_factor=1.0,
            model_precision="FP32", # TODO: Can maybe use FP16 once deployed on the AGX Orin.
        )
        self.get_logger().info(f"Model loaded on device: {self.explorfm_model.device}")

    def camera_info_callback(self, msg):
        """
        Callback for camera info messages.
        Stores the camera calibration matrix and distortion coefficients.
        Unsubscribes from camera info topic after first message since calibration is static.

        Args:
            msg: sensor_msgs/CameraInfo message
        """
        self.camera_info = msg
        self.calibration_matrix = msg.k  # Intrinsic camera matrix (3x3 flattened to 9 elements)
        self.distortion = msg.d  # Distortion coefficients
        self.get_logger().info('Camera info received, unsubscribing from camera_info topic')
        
        # Unsubscribe from camera info since it's static
        self.destroy_subscription(self.camera_info_sub)

    def image_callback(self, msg):
        """
        Callback for compressed image messages.
        Stores only the latest frame (older frames are discarded due to QoS settings).

        Args:
            msg: sensor_msgs/CompressedImage message
        """

        rgb = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.latest_frame = rgb

        self.get_logger().debug('Latest frame received')

    def score_visual_frontiers_callback(self, request, response):
        """
        Service callback for score_visual_frontiers.

        Args:
            request: ScoreVisualFrontiers.Request with array of geometry_msgs/Point
            response: ScoreVisualFrontiers.Response to populate with bool array

        Returns:
            ScoreVisualFrontiers.Response with scores array
        """

        self.get_logger().info(f'Received {len(request.points)} points for visual frontier scoring')
        
        current_frame = self.latest_frame
        if current_frame is None:
            self.get_logger().warning('No camera frame received yet, cannot score visual frontiers')
            response.scores = []
            return response

        # Assume points are already filtered by traversability.
        _, raw_frontier_scores = self.run_inference(current_frame)
        frontier_scores = self.normalize_output(raw_frontier_scores)
        frontiers = self.threshold_frontiers(frontier_scores)

        frontier_bools = self.map_points_to_frontiers(request.points, frontiers)

        response.scores = frontier_bools
        
        return response

    def score_traversability_callback(self, request, response):
        """
        Service callback for score_traversability.

        Args:
            request: ScoreTraversability.Request with array of geometry_msgs/Point
            response: ScoreTraversability.Response to populate with int32 array

        Returns:
            ScoreTraversability.Response with scores array
        """

        self.get_logger().info(f'Received {len(request.points)} points for traversability scoring')

        # Get latest frame and convert to rgb numpy array for model inference.
        current_frame = self.latest_frame
        if current_frame is None:
            self.get_logger().warning('No camera frame received yet, cannot run inference')
            response.scores = []
            return response
        
        start_time = time.time()

        raw_traversability, _ = self.run_inference(current_frame)
        traversability_class_map = self.raw_traversability_to_class_map(raw_traversability)
        point_classes = self.map_points_to_traversability(request.points, traversability_class_map)

        inference_time = time.time() - start_time
        self.get_logger().info(f'Traversability scoring completed in {inference_time:.2f} seconds')

        response.scores = point_classes.tolist()
        
        return response

    def run_inference(self, image_rgb):
        self.get_logger().info("Running inference...")
        inference_start_time = time.time()
        with torch.no_grad():
            traversability, frontiers, _ = self.explorfm_model.forward_on_numpy(image_rgb)
            inference_time = time.time() - inference_start_time
        
        self.get_logger().info(f"Inference completed in {inference_time:.2f} seconds")
        self.get_logger().info(f"Frontiers shape: {frontiers.shape}")
        self.get_logger().info(f"Traversability shape: {traversability.shape}")

        return traversability, frontiers

    def raw_traversability_to_class_map(self, traversability):
        """
        Convert raw traversability output to discrete class map.
        Class 0 = traversable, Class 1 = mild, Class 2 = Untraversable, Class -1 = unknown (this happens if the point projects outside the image bounds)

        Args:
            traversability: numpy array of raw traversability scores (e.g. from 0 to 1)

        Returns:
            class_map: numpy array of int8 with values in {-1, 0, 1, 2}
        """

        # Apply softmax to get probabilities
        probs = F.softmax(traversability, dim=1)  # Shape: (1, 3, H, W)
        probs = probs.squeeze(0).detach().cpu().numpy()  # Shape: (3, H, W)
    
        # Get class predictions (argmax across channels)
        class_map = np.argmax(probs, axis=0).astype(np.uint8)  # Shape: (H, W), values 0-2
    
        return class_map

    def map_points_to_traversability(self, points, traversability_class_map):
        
        img_h, img_w = traversability_class_map.shape

        uv, valid = self.project_points_to_image(points, img_h, img_w)

        # Lookup class for each valid point; default -1 for out-of-bounds.
        point_classes = np.full(len(uv), -1, dtype=np.int8)
        point_classes[valid] = traversability_class_map[uv[valid, 1], uv[valid, 0]]  # class_map[row, col] = class_map[v, u]

        return point_classes 
    
    def map_points_to_frontiers(self, points, frontiers):
        
        img_h, img_w = frontiers.shape

        uv, valid = self.project_points_to_image(points, img_h, img_w)

        # Lookup class for each valid point; default -1 for out-of-bounds.
        frontier_bools = np.full(len(uv), -1, dtype=np.int8)
        frontier_bools[valid] = frontiers[uv[valid, 1], uv[valid, 0]]  # frontiers[row, col] = frontiers[v, u]

        return frontier_bools 
    
    def project_points_to_image(self, points, img_h: int, img_w: int):

        # Assumes points are already in the camera frame.
        pts = points.reshape(-1, 1, 3).astype(np.float64)
        pixels, _ = cv2.projectPoints(objectPoints=pts, cameraMatrix=self.calibration_matrix, dist=self.distortion)
        pixels = pixels.reshape(-1, 2)   # (N, 2)  [u, v]

        # Round to nearest integer pixel.
        uv = np.round(pixels).astype(np.int32)  # (N, 2)

        # Mask out points that project outside the image bounds.
        valid = (
            (uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
        )

        return uv, valid

    def threshold_frontiers(self, tensor: torch.Tensor, threshold: float = VISUAL_FRONTIER_CONFIDENCE_THRESHOLD) -> np.ndarray:
        """Convert single-channel tensor output to a thresholded float array in range [0, 1]."""
        output = tensor.squeeze().detach().cpu().numpy()  # (1, 1, H, W) → (H, W)
        return output > threshold
    
def main(args=None):
    rclpy.init(args=args)
    node = ExplorfmNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
