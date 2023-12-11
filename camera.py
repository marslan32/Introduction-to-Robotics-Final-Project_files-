import pyrealsense2 as rs
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import transforms3d as t3d

from pixel_branch import Pixel_Branch

class IntelCam:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.image_path = "C:/Users/Administrator/Documents/GitHub/mirobot_intelrealsense/images"



    def initialize(self):
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(self.config)
        # Set the exposure anytime during the operation
        rgb_sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]
        rgb_sensor.set_option(rs.option.exposure, 1000.000)



    def rgb_intrinsics(self):
        # get camera intrinsics
        profile = self.pipeline.get_active_profile()

        rgb_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        rgb_intrinsics = rgb_profile.get_intrinsics()

        K = np.array([[rgb_intrinsics.fx, 0, rgb_intrinsics.ppx],
                    [0, rgb_intrinsics.fy, rgb_intrinsics.ppy],
                    [0, 0, 1]])
        return K



    def depth_intrinsics(self):
        depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        return depth_intrinsics
    

    def close(self):
        # Stop streaming
        self.pipeline.stop()

    def capture_image(self):
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)
        hole_filling = rs.hole_filling_filter(2)

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = hole_filling.process(depth_frame)

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() 
        aligned_color_frame = aligned_frames.get_color_frame()

        aligned_depth_frame = hole_filling.process(aligned_depth_frame)

        self.aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        
        aligned_color_image = np.asanyarray(aligned_color_frame.get_data())
        
        # Apply colormap to depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.255), cv2.COLORMAP_JET)
        aligned_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.aligned_depth_image, alpha=0.255), cv2.COLORMAP_JET)

        # segment rgb image
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        # Define the lower and upper bounds of the blueish color in HSV
        # lower_red = np.array([0,220,150])   # Lower bounds for redish color
        # upper_red = np.array([15,255,255]) # Upper bounds for redish color
        lower_red = np.array([0,200,150])   # Lower bounds for redish color
        upper_red = np.array([15,255,255]) # Upper bounds for redish color


        # Create a mask that selects the blueish color in the image
        mask = cv2.inRange(hsv_image, lower_red, upper_red)
        mask_path = f"{self.image_path}/mask.jpeg"
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(f"{self.image_path}/rgb.jpeg", color_image)
        cv2.imwrite(f"{self.image_path}/depth.jpeg", aligned_depth_colormap)

        return mask_path
    
    def depth_val(self, u, v):
        return self.aligned_depth_image[v, u]

        




if __name__=="__main__":
    camera = IntelCam()
    camera.initialize()
    K = camera.rgb_intrinsics()
    mask_path = camera.capture_image()
    camera.close()
    