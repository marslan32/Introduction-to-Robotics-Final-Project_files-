import numpy as np
import math

class RobotMath:
    def __init__(self, K):
        self.K = K

    def euler_to_rotation_matrix(self, x, y, z):
        # Convert angles to radians
        x = np.radians(x)
        y = np.radians(y)
        z = np.radians(z)

        # Calculate individual rotation matrices
        Rx = np.array([[1, 0, 0],
                        [0, np.cos(x), -np.sin(x)],
                        [0, np.sin(x), np.cos(x)]])

        Ry = np.array([[np.cos(y), 0, np.sin(y)],
                        [0, 1, 0],
                        [-np.sin(y), 0, np.cos(y)]])

        Rz = np.array([[np.cos(z), -np.sin(z), 0],
                        [np.sin(z), np.cos(z), 0],
                        [0, 0, 1]])

        # Combine the individual rotation matrices in XYZ order
        rotation_matrix = Rx@Ry@Rz

        return rotation_matrix



    def trans_mat(self, rot_mat, trans_vec):
        return np.vstack((np.hstack((rot_mat, trans_vec)), np.array([0, 0, 0, 1])))

    def cam_pose(self, u, v, zc):
        img_pixels = zc*np.array([[u],
                                  [v],
                                  [1]])
        cam_pose = np.linalg.inv(self.K)@(img_pixels)
        cam_pose = [cam_pose[0].item(), cam_pose[1].item(), cam_pose[2].item()]
        # print(f"cam position: {cam_pose}")

        return cam_pose
    
    def position_6(self, cam_pose, T_x):
        cam_pose_homo = np.append(cam_pose, 1)
        position_6_homo = T_x@cam_pose_homo
        position_6 = position_6_homo[:3] / position_6_homo[3]
        # print(f"position_6: {position_6}")

        return position_6
    
    def position_0(self, position_6, T_0_6):
        position_6_homo = np.append(position_6, 1)
        position_0_homo = T_0_6@position_6_homo
        position_0 = position_0_homo[:3] / position_0_homo[3]
        # print(f"position_0: {position_0}")

        return position_0
    

if __name__=="__main__":
    K = np.array([[1.35483044e3, 0.00000000, 9.64733521e2],
              [0.00000000, 1.35530383e3, 5.46814697e2],
              [0.00000000, 0.00000000, 1.00000000]])
    rob_math = RobotMath(K)

    u = 1410
    v = 387
    zc = 322
    cam_pose = rob_math.cam_pose(u, v, zc)

    trans_vec = np.array([[42.25],
                      [30.850],
                      [25.6]])
    rot_mat = rob_math.euler_to_rotation_matrix(-180, -25, 90)
    T_x = rob_math.trans_mat(rot_mat, trans_vec)
    position_6 = rob_math.position_6(cam_pose, T_x)

    trans_vec = np.array([[140.447],
                      [4.905],
                      [257.038+50]])

    rot_mat = rob_math.euler_to_rotation_matrix(0, -25, 2)
    T_0_6 = rob_math.trans_mat(rot_mat, trans_vec)
    position_0 = rob_math.position_0(position_6, T_0_6)

