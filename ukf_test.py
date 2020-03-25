import sys
import numpy as np
import scipy as sp
import math
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.constants import g
# import transforms3d as tf3d
# import quaternion

import ukf_utils as ukf 
import transformations as tf
from sklearn.metrics import mean_squared_error

_EPS = np.finfo(float).eps * 4.0

if len(sys.argv) == 1:
	dataset_num = 1
else:	
	dataset_num = sys.argv[1]


if __name__ == "__main__":

    

    #############################################################load the data
    imu_data = loadmat('imu/imuRaw'+ str(dataset_num) + '.mat')
    vicon_data = loadmat('vicon/viconRot'+ str(dataset_num) + '.mat')
    acc_data = imu_data['vals'][0:3,:]
    gyro_data = imu_data['vals'][3:6,:]
    imu_ts = imu_data['ts']
    rots_truth = vicon_data['rots']
    vicon_ts = vicon_data['ts']
    imu_ts_del = imu_ts[:,1:imu_ts.size] - imu_ts[:,0:imu_ts.size-1]

    acc_data = acc_data.astype(np.float64)
    gyro_data = gyro_data.astype(np.float64)
    gyro_data = np.roll(gyro_data,2,axis=0)
    

    ################################################# Measurement data pre-processing 
    acc_scale = 3300.0/(1023*(300.0/sp.constants.g))
    gyro_scale_xy = (3000.0/(1023*3.33))*(math.pi/180.0)
    gyro_scale_z = gyro_scale_xy*1 
    
    # 370.2,374,374.75
    
    acc_bias = np.array([512, 500, 500], ndmin=2, dtype = np.float64)
    acc_scale = np.array([acc_scale, acc_scale, acc_scale], ndmin=2, dtype = np.float64 )

    # gyro_bias = np.array([373.8, 375.68,  367.2], ndmin=2, dtype = np.float64)
    gyro_bias = np.array([378,376,369], ndmin=2, dtype = np.float64)    
    # 381,376,370, #374,374.75,370.2
    # gyro_bias = np.array([381, 376, 370], ndmin=2)
    gyro_scale = np.array([gyro_scale_xy , gyro_scale_xy , gyro_scale_z], ndmin=2, dtype = np.float64)
    # gyro_bias = (1.23/3.0)*1023

    # print acc_data
    acc_data = np.multiply(np.subtract(acc_data,acc_bias.T),acc_scale.T)
    gyro_data = np.multiply(np.subtract(gyro_data,gyro_bias.T),gyro_scale.T)
    # gyro_data = np.subtract(gyro_data,gyro_bias)*gyro_scale

    acc_data[0:2,:] = -acc_data[0:2,:]

    # gyro_data[0:2,:] = 0.0
    
    # gyro_data[2,:] = (gyro_data[2,:]/gyro_scale)*0.01706089200385464*2.0

    # print acc_data
    # print gyro_data


    ############################################################# filter initialization variables
    X_init = np.array([ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],dtype=np.float64)
    # X_init[0:4] = tf.quaternion_from_matrix(rots_truth[:,:,0])
    # X_init[4:7] = tf.euler_from_matrix(rots_truth[:,:,0])  
    P_init = np.identity(6,dtype=np.float64)*0.000001
    Q = np.identity(6,dtype=np.float64)*0.000001
    Q[3:6] = Q[3:6]*200
    R = np.identity(6,dtype=np.float64)*0.0043
    R[2,2] = R[2,2]* 0.01

    # Q = np.identity(6,dtype=np.float64)*0.000002  
    # R = np.identity(6,dtype=np.float64)*0.0008

    # print Q
    # R[2,:] = R[2,:]*0.01

    # 1.5*10-6, 4.3*10-2
    #0.0003, 0.0009

    Z_data = np.concatenate((gyro_data,acc_data),axis=0)
    euler_truth = np.empty([3,imu_ts_del.shape[1]],dtype = np.float64)
    euler_est = np.empty([3,imu_ts_del.shape[1]],dtype = np.float64)

    
    num_data_points = imu_ts_del.shape[1]

    ###################################################################### filter loop
    for filter_iteration in range(0,num_data_points-500): 

        del_t = imu_ts_del[:,filter_iteration]
        ############################################################### calculate sigma points    
        dimension_cov = P_init.shape[0]
        num_sigma_pts = 2*dimension_cov

        S = np.linalg.cholesky((P_init+Q))
        # u, s, vh = np.linalg.svd((P_init+Q)*0.01, full_matrices=True)
        # S = u*np.sqrt(s)*vh
        S = S*math.sqrt(dimension_cov)
        W_i = np.concatenate((S,-S), axis=1)
        sigma_pts = ukf.vecmat2quatmat(W_i)   ####### convert matrix from vector space to quaternion space
        X_i = np.empty(sigma_pts.shape)
        X_i[0:4,:] = ukf.quatmat_multiply(sigma_pts[0:4,:],X_init[0:4])  ####### adding(multiplying) quaternions
        X_i[4:7,:] = np.add(W_i[3:6,:],np.array([X_init[4:7]]).T)  ####### adding omegas

        # print X_i[0:4,:]
        # break

        ######################################################## Process model update - Transformations of the sigma points

        orientquat_del = tf.quaternion_about_axis(np.linalg.norm(X_init[4:7])*del_t, X_init[4:7])

        Y_i = np.empty(X_i.shape)
        Y_i[0:4,:] = ukf.quatmat_multiply(X_i[0:4,:],orientquat_del)	######### transformation of sigma points
        Y_i[4:7] = X_i[4:7]

        X_hat_minus = np.empty(X_init.shape, dtype=np.float64)
        X_hat_minus[0:4],error_vec = ukf.compute_mean_quaternion(Y_i[0:4,:], X_init[0:4])  ########### mean of transformed sigma points
        X_hat_minus[4:7] = np.mean(Y_i[4:7,:],1)										   ########### Priori State Update

        W_i_dash = np.empty(W_i.shape, dtype=np.float64)

        X_hat_minus_inv = tf.quaternion_inverse(X_hat_minus[0:4])
            
        for i in range(0,num_sigma_pts):
            W_i_dash[0:3,i] = tf.rotvec_from_quaternion(tf.quaternion_multiply(X_hat_minus_inv, Y_i[0:4,i]))
        
        W_i_dash[3:6,:] = np.subtract(Y_i[4:7,:], np.array([X_hat_minus[4:7]]).T)
        P_hat_minus = ukf.compute_covariance(W_i_dash)								############ computing Priori (state vector) covariance 
        

        ################################################################# Measurement model update
        g_vec_quat = np.array([0,0,0,sp.constants.g])	

        g_dash_vec = np.empty([3,Y_i.shape[1]],dtype=np.float64)
        
        for i in range(0,num_sigma_pts):
            g_dash_quat = tf.quaternion_multiply(tf.quaternion_multiply(tf.quaternion_inverse(Y_i[0:4,i]),g_vec_quat),Y_i[0:4,i])
            g_dash_vec[:,i] = tf.rotvec_from_quaternion(g_dash_quat)

        Z_i = np.empty([Y_i.shape[0]-1, Y_i.shape[1]], dtype=np.float64)
        Z_i[0:3] = Y_i[4:7,:]
        Z_i[3:6] = g_dash_vec

        Z_hat_minus = np.mean(Z_i,1) 							########### mean of transformed sigma points

        V_k = Z_data[:,filter_iteration] - Z_hat_minus			########### innovation is difference of actual and estimated measurement 

        P_zz = ukf.compute_covariance(Z_i, Z_hat_minus)			########### computing measurement covariance 
        P_vv = P_zz + R



        ###################################################################### Cross Covariance
        Z_i_centered = np.subtract(Z_i, np.array([Z_hat_minus]).T)
        P_xz = np.matmul(W_i_dash,Z_i_centered.T)/W_i_dash.shape[1]


        ########################################################################### Kalman Gain 
        K = np.matmul(P_xz,np.linalg.inv(P_vv))



        ########################################################################## Posterior State and Covariance Update
        K_mul_V = np.matmul(K,V_k)
        K_mul_V_quat = tf.quaternion_from_rotvec(K_mul_V[0:3])
        X_hat_quat = tf.quaternion_multiply(X_hat_minus[0:4] , K_mul_V_quat) 
        X_hat_omega = X_hat_minus[4:7] + K_mul_V[3:6]
        X_hat = np.concatenate((X_hat_quat,X_hat_omega),axis=0)

        P_hat = P_hat_minus - np.matmul(np.matmul(K,P_vv),K.T)

        X_init = X_hat
        P_init = P_hat

        
        ########################################################################### debug
        # print P_init
        # print filter_iteration
        euler_truth[:,filter_iteration] = tf.euler_from_matrix(rots_truth[:,:,filter_iteration], axes='sxyz')
        euler_est[:,filter_iteration] = tf.euler_from_quaternion(X_hat[0:4],axes='sxyz')
    
    # print X_i[0:4,:]

    # euler_est[2,:] = euler_est[2,:]

    # print mean_squared_error(euler_truth[0,:], euler_est[0,:]),mean_squared_error(euler_truth[1,:], euler_est[1,:]), mean_squared_error(euler_truth[2,:], euler_est[2,:]) 
    plt.subplot(3, 1, 1)
    plt.plot(range(0,num_data_points-500), euler_truth[0,0:num_data_points-500],range(0,num_data_points-500),euler_est[0,0:num_data_points-500]) 
    plt.subplot(3, 1, 2)
    plt.plot(range(0,num_data_points-500), euler_truth[1,0:num_data_points-500],range(0,num_data_points-500),euler_est[1,0:num_data_points-500]) 
    plt.subplot(3, 1, 3)
    plt.plot(range(0,num_data_points-500), euler_truth[2,0:num_data_points-500],range(0,num_data_points-500),euler_est[2,0:num_data_points-500]) 
    
    plt.show() 
    