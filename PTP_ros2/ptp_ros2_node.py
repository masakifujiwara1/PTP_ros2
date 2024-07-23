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
        # self.ped_sub = self.create_subscription(PedestrianArray, 'ped_seq', self.ped_callback, 10)
        self.data_array = None

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.path = '../checkpoint/social-stgcnn-eth-fpp'
        self.model_path = self.path + '/val_best.pth'
        self.obs_seq_len = 8
        self.pred_seq_len = 12

        self.get_logger().info(f'Model initialized')
        self.model = GAT_TimeSeriesLayer(in_features=2, hidden_features=16, out_features=5, obs_seq_len=self.obs_seq_len, pred_seq_len=self.pred_seq_len, num_heads=2).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.get_logger().info(f'Model loaded')

        with open('data_array.pkl','rb') as f: 
            self.data_array = pickle.load(f)
            self.get_logger().info(f'load array: {self.data_array}, shape: {self.data_array.shape}')

        self.dset_test = TrajectoryDataset(
                obs_len=self.obs_seq_len,
                pred_len=self.pred_seq_len,
                skip=1,norm_lap_matr=True)

        obs_traj, obs_traj_rel, non_linear_ped, loss_mask, V_obs, A_obs = self.dset_test.processed_data(self.data_array)
        V_obs = V_obs.unsqueeze(0).to(self.device)
        A_obs = A_obs.unsqueeze(0).to(self.device)
        V_pred = self.model(V_obs, A_obs)
        print(f'V_pred: {V_pred}, shape: {V_pred.shape}')
    
    def ped_callback(self, msg):
        self.data_array = np.array(msg.data, dtype=np.dtype(msg.dtype)).reshape(msg.shape)
        # if self.data_array.shape == (8, 4):
        #     with open('data_array.pkl', 'wb') as fp:
        #         pickle.dump(self.data_array, fp)

        self.get_logger().info(f"Received NumPy array: {self.data_array}, shape: {self.data_array.shape}")

        # self.dset_test.processed_data(self.data_array)

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