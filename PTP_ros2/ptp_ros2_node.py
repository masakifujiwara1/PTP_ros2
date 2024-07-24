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
import copy

class PtpRos2Node(Node):
    def __init__(self):
        super().__init__('ptp_ros2_node')
        # self.ped_sub = self.create_subscription(PedestrianArray, 'ped_seq', self.ped_callback, 10)
        self.data_array = None

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.path = '../checkpoint/social-stgcnn-eth'
        self.model_path = self.path + '/val_best.pth'
        self.obs_seq_len = 8
        self.pred_seq_len = 12

        self.get_logger().info(f'Model initialized')
        self.model = GAT_TimeSeriesLayer(in_features=2, hidden_features=16, out_features=5, obs_seq_len=self.obs_seq_len, pred_seq_len=self.pred_seq_len, num_heads=2).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.get_logger().info(f'Model loaded')

        self.debug_flag = True

        if self.debug_flag:
            with open('data_array.pkl','rb') as f: 
                self.data_array = pickle.load(f)
                self.get_logger().info(f'load array: {self.data_array}, shape: {self.data_array.shape}')

            with open('raw_data_dict.pkl','rb') as f: 
                self.raw_data_dict = pickle.load(f)
                self.get_logger().info(f'raw_data_dict: {self.raw_data_dict}')

        self.dset_test = TrajectoryDataset(
                obs_len=self.obs_seq_len,
                pred_len=self.pred_seq_len,
                skip=1,norm_lap_matr=True)

        # obs_traj, obs_traj_rel, non_linear_ped, loss_mask, V_obs, A_obs = self.dset_test.processed_data(self.data_array)
        # V_obs = V_obs.unsqueeze(0).to(self.device)
        # A_obs = A_obs.unsqueeze(0).to(self.device)
        # V_pred = self.model(V_obs, A_obs)
        # print(f'V_pred: {V_pred}, shape: {V_pred.shape}')
        # raw_data_dict = self.test()
        # print(raw_data_dict)
        # with open('raw_data_dict.pkl', 'wb') as fp:
        #     pickle.dump(raw_data_dict, fp)
    
    def ped_callback(self, msg):
        self.data_array = np.array(msg.data, dtype=np.dtype(msg.dtype)).reshape(msg.shape)
        if self.data_array.shape == (8, 4):
            self.debug_flag = True
        #     with open('data_array.pkl', 'wb') as fp:
        #         pickle.dump(self.data_array, fp)

        self.get_logger().info(f"Received NumPy array: {self.data_array}, shape: {self.data_array.shape}")

        # self.dset_test.processed_data(self.data_array)

    def test(self, KSTEPS=1):
        # global loader_test,model
        self.model.eval()
        ade_bigls = []
        fde_bigls = []
        raw_data_dict = {}
        step =0 
        # for batch in loader_test: 
        step+=1
        #Get data
        # batch = [tensor.cuda() for tensor in batch]
        # obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
        # loss_mask,V_obs,A_obs,V_tr,A_tr = batch

        obs_traj, obs_traj_rel, non_linear_ped, loss_mask, V_obs, A_obs = self.dset_test.processed_data(self.data_array)
        V_obs = V_obs.unsqueeze(0).to(self.device)
        A_obs = A_obs.unsqueeze(0).to(self.device)
        obs_traj = obs_traj.unsqueeze(0).to(self.device)
        obs_traj_rel = obs_traj_rel.unsqueeze(0).to(self.device)

        # print(obs_traj_rel.shape)
        # print(V_obs.shape)
        num_of_objs = obs_traj_rel.shape[1]

        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        # V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred = self.model(V_obs,A_obs)
        # print(V_pred.shape)
        # torch.Size([1, 5, 12, 2])
        # torch.Size([12, 2, 5])
        # V_pred = V_pred.permute(0,2,3,1)
        # torch.Size([1, 12, 2, 5])>>seq,node,feat
        # V_pred= torch.rand_like(V_tr).cuda()


        # V_tr = V_tr.squeeze()
        # A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze(0)
        num_of_objs = obs_traj_rel.shape[1]
        #print(V_pred.shape)
        V_pred =  V_pred[:,:num_of_objs,:]
        #For now I have my bi-variate parameters 
        #normx =  V_pred[:,:,0:1]
        #normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr
        
        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).to(self.device)
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]
        
        mvnormal = torchdist.MultivariateNormal(mean,cov)


        ### Rel to abs 
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len 
        
        #Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        # print(obs_traj.shape, obs_traj.data.shape)
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        # print(V_x.shape, V_obs.shape)
        V_obs = V_obs[:, :, :, :2]
        # print(V_x.shape, V_obs.shape)
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze(0).copy(),
                                                V_x[0,:,:].copy())

        # V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        # print(V_y.shape, V_tr.shape)
        # V_tr = V_tr[:, :, :2]
        # V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
        #                                         V_x[-1,:,:].copy())
        
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        # raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):
            ade_ls[n]=[]
            fde_ls[n]=[]

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()
            V_pred = V_pred.unsqueeze(0)

            #V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            # print(V_x.shape, V_pred.shape)
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze(0).copy(),
                                                    V_x[-1,:,:].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))
            
        # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = [] 
                target = []
                obsrvs = [] 
                number_of = []
                pred.append(V_pred_rel_to_abs[:,n:n+1,:])
                # target.append(V_y_rel_to_abs[:,n:n+1,:])
                obsrvs.append(V_x_rel_to_abs[:,n:n+1,:])
                number_of.append(1)

                # ade_ls[n].append(ade(pred,target,number_of))
                # fde_ls[n].append(fde(pred,target,number_of))
        
        # for n in range(num_of_objs):
        #     ade_bigls.append(min(ade_ls[n]))
        #     fde_bigls.append(min(fde_ls[n]))

        # ade_ = sum(ade_bigls)/len(ade_bigls)
        # fde_ = sum(fde_bigls)/len(fde_bigls)
        # return ade_,fde_,raw_data_dict
        return raw_data_dict

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