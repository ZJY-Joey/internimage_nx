import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    import numpy as np
except ImportError:
    np = None

class ConfidenceInspector(Node):
    def __init__(self):
        super().__init__('confidence_inspector')
        self.sub = self.create_subscription(
            Image,
            '/zed/zed_node/confidence/confidence_map/decompressedConfidence',
            self.cb,
            10
        )
        self.logged = False

    def cb(self, msg: Image):
        if self.logged:
            return
        self.get_logger().info(f'Confidence image received: width={msg.width} height={msg.height} encoding={msg.encoding} step={msg.step}')
        dtype_map = {
            'mono8': 'uint8', '8UC1': 'uint8',
            'mono16': 'uint16', '16UC1': 'uint16',
            '32FC1': 'float32'
        }
        dtype = dtype_map.get(msg.encoding, 'uint8')
        if np:
            try:
                arr = np.frombuffer(msg.data, dtype=dtype)
                # For multi-channel encodings, a reshape would be needed; assume single channel confidence map.
                self.get_logger().info(f'Pixel stats: min={arr.min()} max={arr.max()} mean={float(arr.mean()):.2f}')
            except Exception as e:
                self.get_logger().warn(f'Failed to compute statistics: {e}')
        else:
            self.get_logger().warn('numpy not installed; skipping statistics.')
        self.get_logger().info('.')
        self.logged = True

def main():
    rclpy.init()
    node = ConfidenceInspector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
