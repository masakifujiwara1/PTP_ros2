import os
import math
import sys
import torch
import pickle
import argparse
import torch.distributions.multivariate_normal as torchdist
import rclpy
from rclpy.node import Node
import numpy as np
from ptp_msgs.msg import PedestrianArray
from utils import *
from metrics import *
from model_depth_fc_fix import GAT_TimeSeriesLayer

class PtpRos2Node(Node):
    def __init__(self):
        super().__init__('ptp_ros2_node')
        self.ped_sub = self.create_subscription(PedestrianArray, 'ped_seq', self.ped_callback, 10)
        self.data_array = None

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.path = '../checkpoint/social-stgcnn-hotel'
        self.model_path = self.path + '/val_best.pth'
        self.obs_seq_len = 8
        self.pred_seq_len = 12
        self.model = GAT_TimeSeriesLayer(in_features=2, hidden_features=16, out_features=5, obs_seq_len=self.obs_seq_len, pred_seq_len=self.pred_seq_len, num_heads=2).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
    
    def ped_callback(self, msg):
        self.data_array = np.array(msg.data, dtype=np.dtype(msg.dtype)).reshape(msg.shape)
        self.get_logger().info(f"Received NumPy array: {self.data_array}, shape: {self.data_array.shape}")

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