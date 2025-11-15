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


# def load_normalized_test_case(test_image, pagelocked_buffer):
#     # Keep compatibility with previous API (accepts a path string).
#     np.copyto(pagelocked_buffer, normalize_pil_image(Image.open(test_image).convert("RGB")))
#     return test_image


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
    """Map segmentation ids to a BGR uint8 image."""
    seg = segmentation.astype(np.int32)
    h, w = seg.shape
    if palette is None:
        cmap = plt.get_cmap('tab20')
        max_id = int(seg.max()) if seg.size else 0
        palette = [[int(255 * c) for c in cmap(i / max(1, max_id + 1))[:3]] for i in range(max_id + 1)]
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    unique = np.unique(seg)
    for cid in unique:
        if cid < 0:
            continue
        color = palette[cid] if cid < len(palette) else [0, 0, 0]
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        mask = (seg == cid)
        color_img[mask] = color_bgr
    return color_img


def visualize_segmentation_and_encode(segmentation, palette=None, class_names=None, merge_legend=True, legend_placement='right'):
    """Return PNG bytes and the composed BGR image for a segmentation map."""
    color_img = colorize_segmentation(segmentation, palette=palette)
    out_img = color_img
    if merge_legend and (class_names is not None):
        panel = _build_cv2_legend_panel(np.unique(segmentation), class_names, palette)
        if panel is not None:
            ih, iw = color_img.shape[:2]
            ph, pw = panel.shape[:2]
            if legend_placement == 'right':
                new_pw = max(1, int(pw * (ih / ph)))
                panel_resized = cv2.resize(panel, (new_pw, ih), interpolation=cv2.INTER_AREA)
                out_img = np.concatenate([color_img, panel_resized], axis=1)
            else:
                new_ph = max(1, int(ph * (iw / pw)))
                panel_resized = cv2.resize(panel, (iw, new_ph), interpolation=cv2.INTER_AREA)
                out_img = np.concatenate([color_img, panel_resized], axis=0)
    ok, buf = cv2.imencode('.png', out_img)
    if not ok:
        raise RuntimeError('Failed to encode segmentation image')
    return buf.tobytes(), out_img


def normalize_pil_image(image):
    """Normalize a PIL.Image to the model input layout and dtype.

    Returns a flattened array in CHW order matching the TensorRT input buffer.
    """
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    c, h, w = ModelData.INPUT_SHAPE
    image = np.asarray(image.resize((w, h), PILImage.LANCZOS), dtype=np.float32)
    image = (image - mean) / std   # (w h c)
    print("image after normalize shape:", image.shape)

    image_arr = (
        image
        .transpose([2, 0, 1])  # (c h w) to (h w c)
        .astype(trt.nptype(ModelData.DTYPE))
        .ravel()
    )
    return image_arr


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
        # self.declare_parameter("depth_topic")
        # self.depth_topic = self.get_parameter("depth_topic").value
        # self.depthImage_sub = self.create_subscription(
        #     Image, self.depth_topic, self.depth_callback, sensor_qos
        # )
        # 发布绘制后的图像
        self.internimage_pub = self.create_publisher(CompressedImage, self.result_topic, 10)
        # 使用传感器数据 QoS 以匹配相机数据特性
        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        # 订阅压缩图像
        self.image_sub = self.create_subscription(
            CompressedImage, self.image_topic, self.image_callback, sensor_qos
        )
        # Read flattened color palette (RGB triplets flattened)
        # YAML key should be 'color_palette_flat': [r0,g0,b0, r1,g1,b1, ...]
        self.declare_parameter("color_palette_flat", [])
        flat_palette = self.get_parameter("color_palette_flat").value
        self._palette = []
        try:
            if isinstance(flat_palette, (list, tuple)) and (len(flat_palette) % 3 == 0):
                it = iter(int(x) for x in flat_palette)
                self._palette = [[next(it), next(it), next(it)] for _ in range(len(flat_palette)//3)]
                self.get_logger().info(f"Loaded color palette with {len(self._palette)} colors")
            elif flat_palette:
                self.get_logger().warning(f"color_palette_flat length {len(flat_palette)} is not multiple of 3; ignoring palette")
        except Exception as e:
            self.get_logger().warning(f"Failed to parse color_palette_flat: {e}")

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
            # self.get_logger().info(f'Preprocess time: {preprocess_ms:.3f} ms') check


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

            # Visualize segmentation (colorize + optional legend) and encode to PNG

            # TODO: 20ms
            try:
                png_bytes, composed = visualize_segmentation_and_encode(
                    segmentation,
                    palette=self._palette if self._palette else None,
                    class_names=getattr(self, 'class_names', None),
                    merge_legend=True,
                    legend_placement='right'
                )

                # Publish as CompressedImage with PNG bytes
                msg = CompressedImage()
                msg.format = 'png'
                msg.data = png_bytes
                self.internimage_pub.publish(msg)
                print("Published segmentation image.")

            except Exception as ex:
                self.get_logger().error(f'Failed to visualize/publish segmentation: {ex}')

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
