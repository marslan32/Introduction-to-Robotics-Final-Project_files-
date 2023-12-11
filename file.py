import numpy as np
import time

from camera_class import IntelCam
from pixel_branch import Pixel_Branch
from wlkata_mirobot import WlkataMirobot
from robot_math import RobotMath



class Main:
    def __init__(self):
        self.camera = IntelCam()
        
        self.robot = WlkataMirobot()

        self.joint_start = {1:3, 2:-6, 3:-19, 4:0, 5:25, 6:0}
        self.start_position = (173.1, 9, 301.9, 0, -25, 3)
        self.x = 173.1
        self.y = 9
        self.z = 301.9
    
    def robot_home(self):
        self.robot.go_to_zero()
        

    def intialize(self):
        self.camera.initialize()

    def hand_eye(self, u, v, zc, K):
        rob_math = RobotMath(K)
        cam_pose = rob_math.cam_pose(u, v, zc)

        trans_vec = np.array([[42.25],
                              [30.850],
                              [25.6]])
        rot_mat = rob_math.euler_to_rotation_matrix(-180, -25, 90)
        T_x = rob_math.trans_mat(rot_mat, trans_vec)
        position_6 = rob_math.position_6(cam_pose, T_x)

        trans_vec = np.array([[173.1],
                              [9],
                              [301.9]])

        rot_mat = rob_math.euler_to_rotation_matrix(0, -25, 3)
        T_0_6 = rob_math.trans_mat(rot_mat, trans_vec)
        position_0 = rob_math.position_0(position_6, T_0_6)

        return position_0
        
        

    
    def main(self):
        # start
        self.intialize()
        K = self.camera.rgb_intrinsics()
        self.robot.home()

        # move to starting position
        self.robot.set_joint_angle(self.joint_start, wait_ok=True)
        time.sleep(1)
        self.x = self.robot.pose.x
        self.y = self.robot.pose.y
        self.z = self.robot.pose.z
        print(f"home coords: {self.x, self.y, self.z}")
        print("real home coords: (173.1, 9, 301.9)")
        
        # capture image
        self.mask_path = self.camera.capture_image()

        # find centers
        self.pixel = Pixel_Branch(self.mask_path)
        centers = self.pixel.optimize_path()
        # median_number = np.median(centers[:,0])
        # median_number_index = np.where(centers[:,0] == median_number)[0][0]

        u = centers[0][0]
        v = centers[0][1]

        # get depth
        zc = self.camera.depth_val(u, v)
        print(f"(u, v, zc): ({u}, {v}, {zc})")
        print(f"K: {K}")

        # calculate hand-eye calibration
        self.robot_coords = self.hand_eye(u, v, zc, K)

        # move robot to new position
        self.robot.linear_interpolation(self.robot_coords[0], self.robot_coords[1], self.robot_coords[2]+40, wait_ok=True)

        time.sleep(2)

        self.robot.set_joint_angle(self.joint_start, wait_ok=True)
if __name__=="__main__":
    mirobot = Main()
    mirobot.main()
    # mirobot.robot_home()


        
