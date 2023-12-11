from robolink import *    # API to communicate with RoboDK
from robodk import *      # robodk robotics toolbox
import numpy as np

from camera_class import IntelCam
from pixel_branch import Pixel_Branch
from robot_math import RobotMath

class Robot:
    def __init__(self, cam_sim, robot_sim):
        # initialize if we're using simulation and mock sensors
        self.robot_sim = robot_sim
        self.cam_sim = cam_sim

        # initialize other classes
        if not self.cam_sim:
            self.camera = IntelCam()

        # joints
        self.joints_home = [3, -2, -23, 0, 0, 0]
        
        # intialize poses
        self.base_frame = xyzrpw_2_pose([0, 0, 0, 0, 0, 0])    
        self.ground_frame = xyzrpw_2_pose([238.1, -17.49, -50, 0.0, 0.0, 0.0])

        # Tool frame pose (offset the tool flange by applying a tool frame transformation)
        # These measurements are obtained from RoboDK, which may not match the real robot
        self.ee_pose = xyzrpw_2_pose([0, 0, 0, 180, 0, 180])
        self.cam_pose = xyzrpw_2_pose([-42.25, 30.850, -25.6, -90, -0, -25])

##############################################################################################################################
    # This is from RoboDK API to connect to the real robot or simulation

    def initialize(self):
        # Any interaction with RoboDK must be done through RDK:
        RDK = Robolink()

        # Select a robot (popup is displayed if more than one robot is available)
        self.robot = RDK.ItemUserPick('Select a robot', ITEM_TYPE_ROBOT)
        if not self.robot.Valid():
            raise Exception('No robot selected or available')
        # Important: by default, the run mode is RUNMODE_SIMULATE
        # If the program is generated offline manually the runmode will be RUNMODE_MAKE_ROBOTPROG,
        # Therefore, we should not run the program on the robot
        if RDK.RunMode() != RUNMODE_SIMULATE:
            self.robot_sim = True

        if not self.robot_sim:
            # Connect to the robot using default IP
            success = self.robot.Connect() # Try to connect once
            #success robot.ConnectSafe() # Try to connect multiple times
            status, status_msg = self.robot.ConnectedState()
            if status != ROBOTCOM_READY:
                # Stop if the connection did not succeed
                raise Exception("Failed to connect: " + status_msg)
            
            # This will set to run the API programs on the robot and the simulator (online programming)
            RDK.setRunMode(RUNMODE_RUN_ROBOT)
            # Note: This is set automatically when we Connect() to the robot through the API

        self.ground = RDK.Item("Ground")
        self.crack = RDK.Item("Crack")
        self.ground.setPose(self.ground_frame)

        if not self.cam_sim:
            self.camera.initialize()
            self.K = self.camera.rgb_intrinsics()
        else:
            self.K = np.array([[1.35483044e3, 0.00000000, 9.64733521e2],
                               [0.00000000, 1.35530383e3, 5.46814697e2],
                               [0.00000000, 0.00000000, 1.00000000]])
            
            


    def hand_eye(self, u, v, zc, K):
        rob_math = RobotMath(K)
        cam_pose = rob_math.cam_pose(u, v, zc)

        trans_vec = np.array([[42.25],
                              [30.850],
                              [25.6]])
        rot_mat = rob_math.euler_to_rotation_matrix(-180, -25, 90)
        T_x = rob_math.trans_mat(rot_mat, trans_vec)
        position_6 = rob_math.position_6(cam_pose, T_x)

        trans_vec = np.array([[self.base_x],
                              [self.base_y],
                              [self.base_z]])

        rot_mat = rob_math.euler_to_rotation_matrix(0, -25, 3)
        T_0_6 = rob_math.trans_mat(rot_mat, trans_vec)
        position_0 = rob_math.position_0(position_6, T_0_6)

        return position_0
    
    def convert_coords(self, centers):
        robot_coordinates = []

        for i in range(len(centers)):
            # get (u, v) coordinates
            u = centers[i][0]
            v = centers[i][1]

            # get depth
            # zc = self.camera.depth_val(int(1280/2), int(720/2))
            zc = 345
            # zc = self.camera.depth_val(u, v)
            print(f"(u, v, zc): ({u}, {v}, {zc})")

            # calculate hand-eye calibration
            robot_coords = self.hand_eye(u, v, zc, self.K)

            robot_coordinates.append(robot_coords)
        # print(idx, actual_z)
        robot_coordinates = np.array(robot_coordinates)
        

        return robot_coordinates

        

    

    def robot_home(self):
        self.robot.setPoseTool(self.ee_pose)
        self.robot.setPoseFrame(self.base_frame)
        # go home
        self.robot.MoveJ(self.joints_home, blocking=True)
        
    
    def move_robot(self, robot_coords):
        print(robot_coords)
        
        self.crack_frame = xyzrpw_2_pose([robot_coords[0], robot_coords[1], robot_coords[2], 0, 0, 0])
        self.crack.setPose(self.crack_frame)
        self.robot.setPoseTool(self.ee_pose)
        self.robot.setPoseFrame(self.crack_frame)
        # self.robot.MoveL(transl(-20, 3, 175), blocking=True)
        self.robot.setSpeed(10)
        self.robot.MoveL(transl(-20, 3, 150), blocking=True)
        self.robot.setSpeed(100)
        time.sleep(0.5)
        # self.robot.MoveL(transl(-20, 3, 175), blocking=True)
    
    def main(self):
        self.initialize()
        self.robot_home()
        time.sleep(1)

        self.base_x, self.base_y, self.base_z = self.robot.Pose().Pos()
        print(f"Home Coordinates: {self.base_x, self.base_y, self.base_z}")
        
        u = 1410
        v = 387
        zc = 322

        if not self.cam_sim:
            # capture image
            self.mask_path = self.camera.capture_image()

            # find centers
            self.pixel = Pixel_Branch(self.mask_path)
            centers = self.pixel.optimize_path()
            u = centers[0][0]
            v = centers[0][1]
            print(centers)

            # # get depth
            # zc = self.camera.depth_val(u, v)
            # print(f"(u, v, zc): ({u}, {v}, {zc})")

            coords = self.convert_coords(centers)

        
        # move to new spot
        for i in range(len(coords)):
            self.move_robot(coords[i])

        # go back home
        self.robot_home()





if __name__=="__main__":
    robot_sim = False
    cam_sim = False
    robot = Robot(cam_sim, robot_sim)
    robot.main()
