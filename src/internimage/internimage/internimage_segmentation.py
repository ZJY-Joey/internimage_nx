import os
import sys
import time
import numpy as np

try:
    import tensorrt as trt
except Exception:
    trt = None

from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import QoSPresetProfiles, QoSProfile, QoSReliabilityPolicy

# Use non-interactive backend for matplotlib when saving images
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# import tensorrt sample common module
sys.path.append('/usr/src/tensorrt/samples/python')
import common
import cv2



class ModelData(object):
    # INPUT_SHAPE = (3, 300, 480)
    DTYPE = trt.float16 if trt is not None else None


def build_engine_onnx(model_file, save_path, logger):
    # Load cached engine if exists
    if os.path.exists(save_path):
        logger.info(f"Found existing engine file: {save_path}. Loading it...")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(save_path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    logger.info("Engine file not found. Building new engine from ONNX...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # set workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(4))

    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            logger.error("Failed to parse the ONNX file.")
            for i in range(parser.num_errors):
                logger.error(parser.get_error(i))
            return None

    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine:
        with open(save_path, "wb") as f:
            f.write(serialized_engine)
        logger.info(f"Saved TensorRT engine to {save_path}")

    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(serialized_engine)


def build_cv2_legend_panel(unique_ids, class_names, palette, font_scale=0.5, padding=8, swatch=16,
                            text_color=(0, 0, 0), bg_color=(255, 255, 255)):
    """Build a legend panel using OpenCV. Returns BGR image or None."""
    filtered = [i for i in unique_ids if 0 <= i < len(class_names)]
    if not filtered:
        return None
    entries = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    max_text_w = 0
    text_h = 0
    for cid in filtered:
        label = str(class_names[cid])
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        max_text_w = max(max_text_w, tw)
        text_h = max(text_h, th)
        color_rgb = palette[cid] if (palette is not None and cid < len(palette)) else (0, 0, 0)
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        entries.append((label, color_bgr))
    row_h = max(swatch + 2, text_h + 4)
    panel_h = padding * 2 + row_h * len(entries)
    panel_w = padding * 3 + swatch + max_text_w
    panel = np.full((panel_h, panel_w, 3), bg_color, dtype=np.uint8)
    for idx, (label, color_bgr) in enumerate(entries):
        y_top = padding + idx * row_h
        cv2.rectangle(panel, (padding, y_top), (padding + swatch, y_top + swatch), color_bgr, thickness=-1)
        x_text = padding * 2 + swatch
        y_text = y_top + min(swatch, row_h) - 4
        cv2.putText(panel, label, (x_text, y_text), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return panel


def colorize_segmentation(segmentation, palette=None):
    """Map segmentation ids to a BGR uint8 image using a fast vectorized path."""
    seg = segmentation.astype(np.int32, copy=False)
    h, w = seg.shape
    if palette is None or len(palette) == 0:
        cmap = plt.get_cmap('tab20')
        max_id = int(seg.max()) if seg.size else 0
        palette = [[int(255 * c) for c in cmap(i / max(1, max_id + 1))[:3]] for i in range(max_id + 1)]
    P = np.asarray(palette, dtype=np.uint8)
    if P.ndim != 2 or P.shape[1] != 3:
        P = np.zeros((max(1, len(palette)), 3), dtype=np.uint8)
    idx = np.clip(seg, 0, len(P) - 1)
    rgb = P[idx]  # HxWx3 RGB
    bgr = rgb[..., ::-1].copy()
    return bgr

def segmentation_to_color_image(segmentation, palette=None):
    """Map segmentation ids to a BGR uint8 image using a fast vectorized path."""
    seg = segmentation.astype(np.int32, copy=False)
    h, w = seg.shape
    if palette is None or len(palette) == 0:
        cmap = plt.get_cmap('tab20')
        max_id = int(seg.max()) if seg.size else 0
        palette = [[int(255 * c) for c in cmap(i / max(1, max_id + 1))[:3]] for i in range(max_id + 1)]
    P = np.asarray(palette, dtype=np.uint8)
    if P.ndim != 2 or P.shape[1] != 3:
        P = np.zeros((max(1, len(palette)), 3), dtype=np.uint8)
    idx = np.clip(seg, 0, len(P) - 1)
    rgb = P[idx]  # HxWx3 RGB
    # print("rgb shape:", rgb.shape)
    # bgr = rgb[..., ::-1].copy()
    # print("bgr shape:", bgr.shape)
    return rgb


def overlay_segmentation_and_encode(segmentation, original_bgr, palette=None, class_names=None,
                                    alpha=0.5, merge_legend=False, legend_placement='right',
                                    encode_format='jpeg', jpeg_quality=80):
    """Overlay colorized segmentation onto the original image and return PNG bytes and BGR image.

    Args:
        segmentation: HxW (int) segmentation ids.
        original_bgr: HxWx3 uint8 original image (BGR).
        palette: List[List[int]] of RGB colors.
        class_names: Optional list of class names for legend.
        alpha: Blend factor for overlay (color mask alpha).
        merge_legend: Whether to append a legend panel.
        legend_placement: 'right' or 'bottom'.

    Returns:
        (png_bytes, out_bgr)
    """
    oh, ow = original_bgr.shape[:2]
    sh, sw = segmentation.shape[:2]
    if (sh, sw) != (oh, ow):
        seg_resized = cv2.resize(segmentation.astype(np.int32), (ow, oh), interpolation=cv2.INTER_NEAREST)
    else:
        seg_resized = segmentation

    color_mask = colorize_segmentation(seg_resized, palette=palette)
    # Ensure uint8
    if color_mask.dtype != np.uint8:
        color_mask = color_mask.astype(np.uint8, copy=False)
    overlay = cv2.addWeighted(original_bgr, 1.0 - alpha, color_mask, alpha, 0.0)

    # print("color_mask.shape", color_mask.shape)

    out_img = overlay
    if merge_legend and (class_names is not None):
        panel = build_cv2_legend_panel(np.unique(seg_resized), class_names, palette)
        if panel is not None:
            ih, iw = out_img.shape[:2]
            ph, pw = panel.shape[:2]
            if legend_placement == 'right':
                new_pw = max(1, int(pw * (ih / ph)))
                panel_resized = cv2.resize(panel, (new_pw, ih), interpolation=cv2.INTER_AREA)
                out_img = np.concatenate([out_img, panel_resized], axis=1)
            else:
                new_ph = max(1, int(ph * (iw / pw)))
                panel_resized = cv2.resize(panel, (iw, new_ph), interpolation=cv2.INTER_AREA)
                out_img = np.concatenate([out_img, panel_resized], axis=0)

    # Encode with configurable format; prefer JPEG for speed/size
    if encode_format.lower() in ('jpg', 'jpeg'):
        ok, buf = cv2.imencode('.jpg', out_img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    else:
        # Low compression for speed
        ok, buf = cv2.imencode('.png', out_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    if not ok:
        raise RuntimeError('Failed to encode overlay image')
    return buf.tobytes(), out_img

class InternImageNode(Node):
    def __init__(self):
        super().__init__('internimage')

        # Parameters
        self.declare_parameter("image_topic","/zed/zed_node/rgb/color/rect/image/compressed")
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.declare_parameter("result_topic","/zed/zed_node/rgb/color/rect/image/compressed/internimage/segmented")
        self.result_topic = self.get_parameter("result_topic").get_parameter_value().string_value
        self.declare_parameter("model_name","upernet_internimage_s_300x480_int8fp16")
        self.model_name = self.get_parameter("model_name").get_parameter_value().string_value
        self.declare_parameter("default_data_dir","/home/jetson/workspaces/segmentation_ws/models/internimage_s/")
        default_data_dir = self.get_parameter('default_data_dir').get_parameter_value().string_value            
        # publish internimage result topic
        self.internimage_pub = self.create_publisher(CompressedImage, self.result_topic, 10)  
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        # Use RELIABLE QoS for segmentation image publishers to be compatible with subscribers
        reliable_qos = QoSProfile(depth=10)
        reliable_qos.reliability = QoSReliabilityPolicy.RELIABLE
        self.image_sub = self.create_subscription(
            CompressedImage, self.image_topic, self._image_callback, sensor_qos
        )
        self.declare_parameter("color_segmentation_topic", "/internimage/color_segmentation_mask")
        self.declare_parameter("id_segmentation_topic", "/internimage/id_segmentation_mask")
        self.declare_parameter("combined_segmentation_topic", "/internimage/combined_segmentation_mask")
        self.color_segmentation_topic = self.get_parameter("color_segmentation_topic").get_parameter_value().string_value
        self.id_segmentation_topic = self.get_parameter("id_segmentation_topic").get_parameter_value().string_value
        self.combined_segmentation_topic = self.get_parameter("combined_segmentation_topic").get_parameter_value().string_value
        self.color_seg_pub = self.create_publisher(Image, self.color_segmentation_topic, reliable_qos)
        self.id_seg_pub = self.create_publisher(Image, self.id_segmentation_topic, reliable_qos)
        # Combined publishes a raw RGBA Image (not CompressedImage)
        self.combined_seg_pub = self.create_publisher(Image, self.combined_segmentation_topic, reliable_qos)

        # Declare both spellings so either YAML key works.
        self.declare_parameter("color_palette_flat", [120,120,120, 180,120,120, 6,230,230, 80,50,50, 4,200,3, 120,120,80, 140,140,140, 204,5,255, 230,230,230, 4,250,7, 224,5,255, 235,255,7, 150,5,61, 120,120,70, 8,255,51, 255,6,82, 143,255,140, 204,255,4, 255,51,7, 204,70,3, 0,102,200, 61,230,250, 255,6,51, 11,102,255, 255,7,71, 255,9,224, 9,7,230, 220,220,220, 255,9,92, 112,9,255, 8,255,214, 7,255,224, 255,184,6, 10,255,71, 255,41,10, 7,255,255, 224,255,8, 102,8,255, 255,61,6, 255,194,7, 255,122,8, 0,255,20, 255,8,41, 255,5,153, 6,51,255, 235,12,255, 160,150,20, 0,163,255, 140,140,140, 250,10,15, 20,255,0, 31,255,0, 255,31,0, 255,224,0, 153,255,0, 0,0,255, 255,71,0, 0,235,255, 0,173,255, 31,0,255, 11,200,200, 255,82,0, 0,255,245, 0,61,255, 0,255,112, 0,255,133, 255,0,0, 255,163,0, 255,102,0, 194,255,0, 0,143,255, 51,255,0, 0,82,255, 0,255,41, 0,255,173, 10,0,255, 173,255,0, 0,255,153, 255,92,0, 255,0,255, 255,0,245, 255,0,102, 255,173,0, 255,0,20, 255,184,184, 0,31,255, 0,255,61, 0,71,255, 255,0,204, 0,255,194, 0,255,82, 0,10,255, 0,112,255, 51,0,255, 0,194,255, 0,122,255, 0,255,163, 255,153,0, 0,255,10, 255,112,0, 143,255,0, 82,0,255, 163,255,0, 255,235,0, 8,184,170, 133,0,255, 0,255,92, 184,0,255, 255,0,31, 0,184,255, 0,214,255, 255,0,112, 92,255,0, 0,224,255, 112,224,255, 70,184,160, 163,0,255, 153,0,255, 71,255,0, 255,0,163, 255,204,0, 255,0,143, 0,255,235, 133,255,0, 255,0,235, 245,0,255, 255,0,122, 255,245,0, 10,190,212, 214,255,0, 0,204,255, 20,0,255, 255,255,0, 0,153,255, 0,41,255, 0,255,204, 41,0,255, 41,255,0, 173,0,255, 0,245,255, 71,0,255, 122,0,255, 0,255,184, 0,92,255, 184,255,0, 0,133,255, 255,214,0, 25,194,194, 102,255,0, 92,0,255])  # common misspelling
        # Prefer correctly spelled; fallback to misspelled.
        palette_candidate = self.get_parameter("color_palette_flat").value
        self._palette = []
        try:
            if isinstance(palette_candidate, (list, tuple)) and palette_candidate:
                if len(palette_candidate) % 3 != 0:
                    self.get_logger().warning(
                        f"Palette length {len(palette_candidate)} not multiple of 3; truncating extra values"
                    )
                triplet_count = len(palette_candidate) // 3
                it = iter(int(x) for x in palette_candidate[:triplet_count * 3])
                self._palette = [[next(it), next(it), next(it)] for _ in range(triplet_count)]
                self.get_logger().info(f"Loaded palette with {triplet_count} colors (RGB)")
            else:
                self.get_logger().info("No palette provided; will auto-generate if needed")
        except Exception as e:
            self.get_logger().warning(f"Failed to parse palette: {e}")

        # declare both spellings so either YAML key works.
        self.declare_parameter("classes", ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag'])
        classes_candidate = self.get_parameter("classes").value
        self.class_names = None
        try:
            if isinstance(classes_candidate, (list, tuple)) and classes_candidate:
                self.class_names = [str(x) for x in classes_candidate]
                self.get_logger().info(f"Loaded {len(self.class_names)} class names")
            else:
                self.get_logger().info("No class names provided; legend will be omitted")
        except Exception as e:
            self.get_logger().warning(f"Failed to parse class names: {e}")

        # Runtime visualization/publish tuning parameters
        self.declare_parameter('show_legend', False)
        self.declare_parameter('overlay_alpha', 0.5)
        self.declare_parameter('publish_stride', 1)
        self.declare_parameter('output_format', 'jpeg')  # 'jpeg' or 'png'
        self.declare_parameter('jpeg_quality', 80)
        # Combined matrix publish control (segmentation + color)
        self.declare_parameter('publish_combined', False)
        self.declare_parameter('combined_target_height', 600)
        self.declare_parameter('combined_target_width', 960)
        self.declare_parameter('trt_h', 300)
        self.declare_parameter('trt_w', 480)
        self.declare_parameter('mean', [123.675, 116.28, 103.53])
        self.declare_parameter('std', [58.395, 57.12, 57.375])
        self.declare_parameter('_inv_std', [0.017, 0.0175, 0.0173])  # for reference only; computed internally
        self.declare_parameter('visualize_image_en', False)

        self.show_legend = bool(self.get_parameter('show_legend').value)
        self.overlay_alpha = float(self.get_parameter('overlay_alpha').value)
        self.publish_stride = max(1, int(self.get_parameter('publish_stride').value))
        self.output_format = str(self.get_parameter('output_format').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.publish_combined = bool(self.get_parameter('publish_combined').value)
        self.combined_h = int(self.get_parameter('combined_target_height').value)
        self.combined_w = int(self.get_parameter('combined_target_width').value)
        self.trt_h = int(self.get_parameter('trt_h').value)
        self.trt_w = int(self.get_parameter('trt_w').value)
        self.mean = self.get_parameter('mean').value
        self.std = self.get_parameter('std').value
        self._mean = np.array(self.mean, dtype=np.float32)  # RGB
        self._inv_std = np.array([1.0/s for s in self.std], dtype=np.float32)  # RGB
        self.visualize_image_en = bool(self.get_parameter('visualize_image_en').value)
        # Warn only once if ID range exceeds mono8 capability
        self._warned_id_overflow = False


        # Debug summary (concise)
        if self._palette:
            self.get_logger().debug(f"First palette color: {self._palette[0]}")
        if self.class_names:
            self.get_logger().debug(f"First class name: {self.class_names[0]}")

        # Frame counter for stride-based publishing
        self._frame_idx = 0

        onnx_model_file = default_data_dir + self.model_name + '.onnx'

        self.get_logger().info(f"Using data_dir: {default_data_dir}")

        engine_path = os.path.join(default_data_dir, self.model_name + '.engine')

        # Build or load engine
        if trt is None:
            self.get_logger().error('tensorrt Python module not available')
            raise RuntimeError('tensorrt not available')

        logger = self.get_logger()
        self.get_logger().info('Building/loading TensorRT engine...')
        engine = build_engine_onnx(onnx_model_file, engine_path, logger)
        if engine is None:
            self.get_logger().error('Failed to build/load engine')
            raise RuntimeError('Failed to create TensorRT engine')

        # allocate buffers
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        context = engine.create_execution_context()

        self._engine = engine
        self._inputs = inputs
        self._outputs = outputs
        self._bindings = bindings
        self._stream = stream
        self._context = context
        # Precompute normalization constants for fast per-frame preprocessing
        self._mean = np.array(self._mean, dtype=np.float32)  # RGB
        self._inv_std = np.array(self._inv_std, dtype=np.float32)  # RGB
        self.segmentation_result = None
        self.cv_image = None
        self.image_msg = None



        # CvBridge for converting sensor_msgs/Image to OpenCV
        self.bridge = CvBridge()

        # --- Inference gating state ---
        # Sequence number incremented every time a new image arrives via _image_callback
        self._last_image_seq = 0
        # Sequence number of last image we actually ran inference on
        self._last_inferred_seq = 0
        # Flag set True when a new image has been received & preprocessed but not yet inferred
        self._new_image_available = False
        # Idle logging control to avoid spamming logs while waiting for images
        self._last_wait_log_time = 0.0
        self._idle_log_interval = 5.0  # seconds between idle debug logs


    def _combined_seg_color_msg(self, seg2d, color_rgb, header=None):
        seg = np.asarray(seg2d)
        if seg.ndim != 2:
            raise ValueError(f"segmentation must be 2D, got {seg.shape}")
        seg = np.clip(seg, 0, None)
        color = np.asarray(color_rgb)
        if color.shape[:2] != seg.shape:
            raise ValueError("color_rgb shape mismatch segmentation")
        # rgba8: channels order RGBA. Our color is RGB; map R<-R, G<-G, B<-B, A<-seg

        rgba = np.stack([color_rgb[:, :, 0], color_rgb[:, :, 1], color_rgb[:, :, 2], seg.astype(np.uint8)], axis=-1).astype(np.uint8)
        msg = self.bridge.cv2_to_imgmsg(rgba, encoding='rgba8')
        if header is not None:
            msg.header = header
        return msg
    
    def _image_callback(self, msg):
        t0 = time.time_ns()
        try:
            # Decode incoming ROS image (CompressedImage or Image) into an OpenCV BGR ndarray
            if isinstance(msg, CompressedImage):
                np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if cv_image is None:
                    raise RuntimeError("unable to decode compressed image (data may be corrupted or unsupported encoding)")
            elif isinstance(msg, Image):
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                raise TypeError(f"Unsupported image message type: {type(msg)}, please publish Image or CompressedImage")
            
            t1 = time.time_ns()
            self.get_logger().info(f'Image decoding time: {(t1 - t0) / 1e6:.3f} ms')
            # print("cv_image received, shape:", cv_image.shape) #( h w c)
            # Fast preprocess (BGR -> RGB, resize, normalize) directly into TensorRT host buffer
            self._preprocess_to_trt_input(cv_image)
            t2 = time.time_ns()
            elapsed_ms = (t2 - t1) / 1e6
            self.get_logger().info(f'Preprocessing time: {elapsed_ms:.3f} ms')

        except Exception as e:
            self.get_logger().error(f'Error during imagecallback: {e}')
        self.cv_image = cv_image
        self.image_msg = msg
        # Mark new image available for inference and bump sequence counter
        self._last_image_seq += 1
        self._new_image_available = True
        t3 = time.time_ns()
        # check inference total time
        elapsed_ms = (t3 - t0) / 1e6
        self.get_logger().info(f'toatal image callback time: {elapsed_ms:.3f} ms')


    def _preprocess_to_trt_input(self, bgr_image):
        """Fast path: BGR -> RGB, resize to (H,W), normalize, write into TRTPagelocked buffer.
        Expects ModelData.INPUT_SHAPE=(C,H,W).
        """
        # print("================================ Preprocessing ==================")
        c, h, w = (3, self.trt_h, self.trt_w)
        ih, iw = bgr_image.shape[:2]
        if (ih, iw) != (h, w):
            # INTER_AREA for downscale, INTER_LINEAR for upsample
            interp = cv2.INTER_AREA if ih >= h and iw >= w else cv2.INTER_LINEAR
            bgr = cv2.resize(bgr_image, (w, h), interpolation=interp)
        else:
            bgr = bgr_image

        # Convert BGR->RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # To float32 for math
        rgb = rgb.astype(np.float32, copy=False)
        # Normalize per channel: (x - mean) / std  => (x - mean) * inv_std
        rgb -= self._mean
        rgb *= self._inv_std

        # Write into TensorRT host buffer in CHW order
        host = self._inputs[0].host
        host_view = host.reshape((c, h, w))
        if host_view.dtype != rgb.dtype:
            rgb = rgb.astype(host_view.dtype, copy=False)
        # Split channels without extra copies
        host_view[0, :, :] = rgb[:, :, 0]
        host_view[1, :, :] = rgb[:, :, 1]
        host_view[2, :, :] = rgb[:, :, 2]
        
        # print("================================ End Preprocessing ==================")
    
    def _inference_publish(self, cv_image, msg):
        # Guard: only run inference if a new image has arrived since last inference.
        if cv_image is None or msg is None or (not self._new_image_available) or (self._last_image_seq == self._last_inferred_seq):
            # Throttled idle logging
            now = time.time()
            if (now - self._last_wait_log_time) >= self._idle_log_interval:
                self.get_logger().debug("Idle: no new image; inference paused.")
                self._last_wait_log_time = now
            return
        else:
            start_ns = time.time_ns()
            trt_outputs = common.do_inference(
                self._context,
                engine=self._engine,
                bindings=self._bindings,
                inputs=self._inputs,
                outputs=self._outputs,
                stream=self._stream,
            )
            end_ns = time.time_ns()
            elapsed_ms = (end_ns - start_ns) / 1e6
            self.get_logger().info(f'Inference time: {elapsed_ms:.3f} ms')

            # print("trt_shape:", trt_outputs[0].shape)
            # print("trt_outputs[0] dtype:", trt_outputs[0].dtype)

            self.segmentation_result = trt_outputs[0].reshape((self.trt_h, self.trt_w))  # (H,W) 类别ID整型图 (模型输出分辨率)
            # print("segmentation shape:", self.segmentation_result.shape)
            # print("segmentation[0][0]:", self.segmentation_result[0][0])

            t4 = time.time_ns()
            # Publish segmentation outputs
            # 1) Native resolution (match incoming RGB for compatibility with depth_image_proc)
            in_h, in_w = cv_image.shape[:2]
            if self.segmentation_result.shape != (in_h, in_w):
                seg_native = cv2.resize(self.segmentation_result.astype(np.int32), (in_w, in_h), interpolation=cv2.INTER_NEAREST)
            else:
                seg_native = self.segmentation_result

            # 2) Combined resolution (for optional combined RGBA output)
            target_h = self.combined_h
            target_w = self.combined_w
            if (in_h, in_w) == (target_h, target_w):
                seg_combined = seg_native
            elif self.segmentation_result.shape == (target_h, target_w):
                seg_combined = self.segmentation_result
            else:
                seg_combined = cv2.resize(seg_native.astype(np.int32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            if self.publish_combined:
                try:
                    color_combined = segmentation_to_color_image(seg_combined, palette=self._palette)
                    # self.get_logger().info(f'Color combined shape: {color_combined.shape}')  # (600, 960, 3)
                    combined_msg = self._combined_seg_color_msg(seg_combined, color_combined, header=getattr(msg, 'header', None))
                    self.combined_seg_pub.publish(combined_msg)
                except Exception as ex:
                    self.get_logger().error(f'publish combined array failed: {ex}')
            else:
                try:
                    # Publish color segmentation mask at native resolution for depth_image_proc
                    color_native = segmentation_to_color_image(seg_native, palette=self._palette)
                    # Convert RGB (HxWx3) -> BGRA (HxWx4) with opaque alpha channel and publish as 'bgra8'
                    try:
                        # Preferred: use OpenCV conversion (handles dtype and channel order)
                        bgra_native = cv2.cvtColor(color_native, cv2.COLOR_RGB2BGRA)
                    except Exception:
                        # Fallback: manually build BGRA from RGB
                        alpha = np.full((color_native.shape[0], color_native.shape[1], 1), 255, dtype=np.uint8)
                        # Convert RGB -> BGR then append alpha
                        bgra_native = np.concatenate([color_native[:, :, ::-1], alpha], axis=-1)
                    bgra_native = bgra_native.astype(np.uint8, copy=False)
                    color_msg = self.bridge.cv2_to_imgmsg(bgra_native, encoding='bgra8')
                    if hasattr(msg, 'header'):
                        color_msg.header = msg.header
                    self.color_seg_pub.publish(color_msg)

                    # Publish ID mask as mono8 (warn if overflow)
                    max_id = int(seg_native.max()) if seg_native.size else 0
                    if max_id > 255 and not self._warned_id_overflow:
                        self.get_logger().warning(
                            'Segmentation IDs exceed 255; mono8 will overflow. '
                            'Consider using the combined RGBA topic or a 16UC1 ID image.'
                        )
                        self._warned_id_overflow = True
                    id_u8 = np.clip(seg_native, 0, 255).astype(np.uint8, copy=False)
                    id_msg = self.bridge.cv2_to_imgmsg(id_u8, encoding='mono8')
                    if hasattr(msg, 'header'):
                        id_msg.header = msg.header
                    self.id_seg_pub.publish(id_msg)
                except Exception as ex:
                    self.get_logger().error(f'publish color segmentation failed: {ex}')

            # 3) Visualization overlay (optional) in rgb image space if enabled
            if  self.visualize_image_en:
                # Visualize segmentation projected onto original image and encode for publishing
                try:
                    self._frame_idx += 1
                    if (self._frame_idx % self.publish_stride) == 0:
                        print("Publishing overlay segmentation...")
                        img_bytes, composed = overlay_segmentation_and_encode(
                            self.segmentation_result,
                            original_bgr=cv_image,
                            palette=self._palette if self._palette else None,
                            class_names=self.class_names if self.show_legend else None,
                            alpha=self.overlay_alpha,
                            merge_legend=self.show_legend,
                            legend_placement='right',
                            encode_format=self.output_format,
                            jpeg_quality=self.jpeg_quality
                        )
                        # print("Composed image shape:", composed.shape)
                        # t4 = time.time_ns()
                        # visualize_ms = (t4 - t3) / 1e6
                        # self.get_logger().info(f'Visualization time: {visualize_ms:.3f} ms')

                        # Publish as CompressedImage with original header if available
                        out_msg = CompressedImage()
                        if hasattr(msg, 'header'):
                            out_msg.header = msg.header
                        out_msg.format = 'jpeg' if self.output_format.lower() in ('jpg','jpeg') else 'png'
                        out_msg.data = img_bytes
                        self.internimage_pub.publish(out_msg)

                        # t5 = time.time_ns()
                        # publish_ms = (t5 - t4) / 1e6
                        # self.get_logger().info(f'Publish time: {publish_ms:.3f} ms')
                
                except Exception as ex:
                    self.get_logger().error(f'Failed to visualize/publish overlay segmentation: {ex}')
            t5 = time.time_ns()
            total_ms = (t5 - t4) / 1e6    
            self.get_logger().info(f'publish time: {total_ms:.3f} ms')
            # Update gating state: mark this image as processed
            self._last_inferred_seq = self._last_image_seq
            self._new_image_available = False
    


    def destroy_node(self):
        # Try to free buffers
        try:
            common.free_buffers(self._inputs, self._outputs, self._stream)
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = InternImageNode()
    
    try:
        while(rclpy.ok()):
            # Spin first to allow image callbacks to fill new data
            rclpy.spin_once(node, timeout_sec=0.1)
            # Only perform inference if a new image has arrived
            node._inference_publish(node.cv_image, node.image_msg)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
