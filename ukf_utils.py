import numpy as np
import math

import transformations as tf


def compute_mean_quaternion(sigma_pts, init_mean):

	#returns the quaternion mean of the quaternion state in the sigma_pts using gradient descent

	num_pts = sigma_pts.shape[1]

	# error_quat = np.empty(sigma_pts.shape)
	error_vec = np.empty([3,num_pts], dtype=np.float64)
	mean_error_vec = np.ones([3,],dtype=np.float64)
	iteration = 0;
	while(np.linalg.norm(mean_error_vec)>0.001 and iteration<50):
		iteration += 1
		prev_mean = tf.quaternion_inverse(init_mean)

		for i in range(0,num_pts):
			error_quat = tf.quaternion_multiply(sigma_pts[:,i],prev_mean)
			
			error_vec[:,i] = tf.rotvec_from_quaternion(error_quat)
		
		mean_error_vec = np.mean(error_vec,1)

		init_mean =	tf.quaternion_multiply(tf.quaternion_from_rotvec(mean_error_vec),init_mean)
	# print iteration	
	return init_mean,error_vec



def vecmat2quatmat(vecmat):
	# returns whole matrix of quaternions from matrix of vectors (vectorized)

	# vecmat_vecsub = vecmat[0:3,:]
	dim = vecmat.shape[1]
	vecmat_quatsub = np.empty([4, dim])

	for i in range(0, dim):
		vecmat_quatsub[:,i] = tf.quaternion_from_rotvec(vecmat[0:3,i])
	
	quatmat = np.concatenate((vecmat_quatsub,vecmat[3:6,:]),axis=0)	
	return quatmat



def quatmat_multiply(quatmat, quaternion):
	# vectorized multiplication between quaternion matrix and quaternion

	product = np.empty(quatmat.shape)
	product[0,:] = -quatmat[1,:]*quaternion[1] - quatmat[2,:]*quaternion[2] - quatmat[3,:]*quaternion[3] + quatmat[0,:]*quaternion[0] 
	product[1,:] = +quatmat[1,:]*quaternion[0] + quatmat[2,:]*quaternion[3] - quatmat[3,:]*quaternion[2] + quatmat[0,:]*quaternion[1]
	product[2,:] = -quatmat[1,:]*quaternion[3] + quatmat[2,:]*quaternion[0] + quatmat[3,:]*quaternion[1] + quatmat[0,:]*quaternion[2]
	product[3,:] = +quatmat[1,:]*quaternion[2] - quatmat[2,:]*quaternion[1] + quatmat[3,:]*quaternion[0] + quatmat[0,:]*quaternion[3]

	# for i in range(0, quatmat.shape[1]):
	# 	product[:,i] = tf.quaternion_multiply(quatmat[:,i],quaternion)

	return product



def compute_covariance(distribution, mean=[]):
	# vectorized covariance computation from distribution matrix
	
	if mean == []:
		mean = np.zeros(distribution.shape[0])
	centered_distribution = np.subtract(distribution, np.array([mean]).T)
	covariance = np.matmul(centered_distribution,centered_distribution.T)/distribution.shape[1]
	return covariance



