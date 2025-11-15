import os
import sys
import time
import io
import random
from tokenize import String

import PIL
import numpy as np

try:
    import tensorrt as trt
except Exception:
    trt = None

from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import QoSPresetProfiles

# Use non-interactive backend for matplotlib when saving images
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
# import tensorrt sample common module
sys.path.append('/usr/src/tensorrt/samples/python')
import common
import cv2
from PIL import Image as PILImage


class ModelData(object):
    INPUT_SHAPE = (3, 300, 480)
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


def _build_cv2_legend_panel(unique_ids, class_names, palette, font_scale=0.5, padding=8, swatch=16,
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

    out_img = overlay
    if merge_legend and (class_names is not None):
        panel = _build_cv2_legend_panel(np.unique(seg_resized), class_names, palette)
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
        self.declare_parameter("result_topic","/internimage/segmentation")
        self.result_topic = self.get_parameter("result_topic").get_parameter_value().string_value
        self.declare_parameter("model_name","upernet_internimage_s_300x480_int8fp16")
        self.model_name = self.get_parameter("model_name").get_parameter_value().string_value
        self.declare_parameter("default_data_dir","/home/jetson/workspaces/segmentation_ws/models/internimage_s/")
        default_data_dir = self.get_parameter('default_data_dir').get_parameter_value().string_value            
        self.declare_parameter("depth_topic","/zed/zed_node/depth/depth_registered")
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value  
        # 发布绘制后的图像
        self.internimage_pub = self.create_publisher(CompressedImage, self.result_topic, 10)
        # 使用传感器数据 QoS 以匹配相机数据特性
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        # 订阅压缩图像
        self.image_sub = self.create_subscription(
            CompressedImage, self.image_topic, self.image_callback, sensor_qos
        )
        # Disabled by default; implement depth_callback before enabling to avoid runtime errors.
        # self.depth_sub = self.create_subscription(
        #     CompressedImage, self.depth_topic, self.depth_callback, sensor_qos
        # )
        # --- Load parameters: palette & classes with fallback spellings ---
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

        self.show_legend = bool(self.get_parameter('show_legend').value)
        self.overlay_alpha = float(self.get_parameter('overlay_alpha').value)
        self.publish_stride = max(1, int(self.get_parameter('publish_stride').value))
        self.output_format = str(self.get_parameter('output_format').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

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
        self._mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # RGB
        self._inv_std = np.array([1.0/58.395, 1.0/57.12, 1.0/57.375], dtype=np.float32)  # RGB



        # CvBridge for converting sensor_msgs/Image to OpenCV
        self.bridge = CvBridge()


    def _preprocess_to_trt_input(self, bgr_image):
        """Fast path: BGR -> RGB, resize to (H,W), normalize, write into TRTPagelocked buffer.
        Expects ModelData.INPUT_SHAPE=(C,H,W).
        """
        # print("================================ Preprocessing ==================")
        c, h, w = ModelData.INPUT_SHAPE
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

    def image_callback(self, msg):
        t0 = time.time_ns()
        try:
            # Decode incoming ROS image (CompressedImage or Image) into an OpenCV BGR ndarray
            if isinstance(msg, CompressedImage):
                np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if cv_image is None:
                    raise RuntimeError("无法解码压缩图像（可能数据损坏或编码不支持）")
            elif isinstance(msg, Image):
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                raise TypeError(f"不支持的图像消息类型: {type(msg)}，请发布 Image 或 CompressedImage")
            
            t1 = time.time_ns()
            print("cv_image received, shape:", cv_image.shape) #( h w c)
            # Fast preprocess (BGR -> RGB, resize, normalize) directly into TensorRT host buffer
            self._preprocess_to_trt_input(cv_image)
            t2 = time.time_ns()
            preprocess_ms = (t2 - t1) / 1e6
            self.get_logger().info(f'Preprocess time: {preprocess_ms:.3f} ms') 


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

            print("trt_shape:", trt_outputs[0].shape)
            print("trt_outputs[0] dtype:", trt_outputs[0].dtype)

            segmentation = trt_outputs[0].reshape((300, 480))  # 600 960
            print("segmentation shape:", segmentation.shape)
            print("segmentation[0][0]:", segmentation[0][0])

            t3 = time.time_ns()

            # Visualize segmentation projected onto original image and encode for publishing
            try:
                self._frame_idx += 1
                if (self._frame_idx % self.publish_stride) == 0:
                    print("Publishing overlay segmentation...")
                    img_bytes, composed = overlay_segmentation_and_encode(
                        segmentation,
                        original_bgr=cv_image,
                        palette=self._palette if self._palette else None,
                        class_names=self.class_names if self.show_legend else None,
                        alpha=self.overlay_alpha,
                        merge_legend=self.show_legend,
                        legend_placement='right',
                        encode_format=self.output_format,
                        jpeg_quality=self.jpeg_quality
                    )
                    print("Composed image shape:", composed.shape)
                    t4 = time.time_ns()
                    visualize_ms = (t4 - t3) / 1e6
                    self.get_logger().info(f'Visualization time: {visualize_ms:.3f} ms')

                    # Publish as CompressedImage with original header if available
                    out_msg = CompressedImage()
                    if hasattr(msg, 'header'):
                        out_msg.header = msg.header
                    out_msg.format = 'jpeg' if self.output_format.lower() in ('jpg','jpeg') else 'png'
                    out_msg.data = img_bytes
                    self.internimage_pub.publish(out_msg)

                    t5 = time.time_ns()
                    publish_ms = (t5 - t4) / 1e6
                    self.get_logger().info(f'Publish time: {publish_ms:.3f} ms')
            

            except Exception as ex:
                self.get_logger().error(f'Failed to visualize/publish overlay segmentation: {ex}')

        except Exception as e:
            self.get_logger().error(f'Error during inference/publish: {e}')
        # check inference total time
        t1 = time.time_ns()
        elapsed_ms = (t1 - t0) / 1e6
        self.get_logger().info(f'Total callback time: {elapsed_ms:.3f} ms')


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
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
