import numpy as np
from robot_math import RobotMath

rob_math = RobotMath()

u = 1410
v = 387
zc = 322

K = np.array([[1.35483044e3, 0.00000000, 9.64733521e2],
              [0.00000000, 1.35530383e3, 5.46814697e2],
              [0.00000000, 0.00000000, 1.00000000]])
img_pixels = zc*np.array([[u],
                          [v],
                          [1]])
        
cam_pose = np.linalg.inv(K)@(img_pixels)
cam_pose = [cam_pose[0].item(), cam_pose[1].item(), cam_pose[2].item()]
print(f"cam position: {cam_pose}")

trans_vec = np.array([[42.25],
                      [30.850],
                      [25.6]])

rot_mat = rob_math.euler_to_rotation_matrix(-180, -25, 90)

T_x = rob_math.trans_mat(rot_mat, trans_vec)

cam_pose_homo = np.append(cam_pose, 1)
position_6_homo = T_x@cam_pose_homo
position_6 = position_6_homo[:3] / position_6_homo[3]
print(f"position_6: {position_6}")

# T_0_6 = np.array([[1, 0, 0, 135.598],
#                 [0, 1, 0, 4.737],
#                 [0, 0, 1, 293.085],
#                 [0, 0, 0, 1]])
trans_vec = np.array([[140.447],
                      [4.905],
                      [257.038+50]])

rot_mat = rob_math.euler_to_rotation_matrix(0, -25, 2)
T_0_6 = rob_math.trans_mat(rot_mat, trans_vec)

# find position relative to robot
position_6_homo = np.append(position_6, 1)
position_0_homo = T_0_6@position_6_homo
position_0 = position_0_homo[:3] / position_0_homo[3]
print(f"position_0: {position_0}")
print("Real Position: (209.6, -76, 25.7)")