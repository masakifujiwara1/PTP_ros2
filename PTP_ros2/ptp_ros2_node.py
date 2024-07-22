import rclpy
from rclpy.node import Node
import numpy as np
from ptp_msgs.msg import PedestrianArray

class PtpRos2Node(Node):
    def __init__(self):
        super().__init__('ptp_ros2_node')

        self.ped_sub = self.create_subscription(PedestrianArray, 'ped_seq', self.ped_callback, 10)
        self.data_array = None
    
    def ped_callback(self, msg):
        self.data_array = np.array(msg.data, dtype=np.dtype(msg.dtype)).reshape(msg.shape)
        self.get_logger().info(f"Received NumPy array: {self.data_array}")

def main():
    try:
        rclpy.init()
        node = PtpRos2Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("ctrl-C")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()