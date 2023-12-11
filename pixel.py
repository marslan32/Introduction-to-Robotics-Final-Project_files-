import cv2
import numpy as np
from skimage.morphology import skeletonize, binary_closing, disk
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os


class Pixel_Branch:
    def __init__(self, image_path):
        self.image_path = image_path
        # Step 1: Read the binary mask image of the crack
        self.binary_mask = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        self.num_clusters = 25

    def find_branch_centers(self):

        # Step 2: Pre-processing
        blurred_mask = cv2.GaussianBlur(self.binary_mask, (5, 5), 0)

        # Step 3: Thresholding
        _, thresholded_mask = cv2.threshold(blurred_mask, 150, 255, cv2.THRESH_BINARY)

        # Step 4: Morphological operations
        closed_mask = binary_closing(thresholded_mask, disk(7))

        # Step 5: Use skimage.morphology.skeletonize to obtain the medial axis
        medial_axis = skeletonize(closed_mask)

        # Convert boolean array to binary image (0 and 255)
        medial_axis = medial_axis.astype(np.uint8) * 255

        # # Step 2: Use skimage.morphology.skeletonize to obtain the medial axis
        # medial_axis = skeletonize(self.binary_mask > 0)
        # # Convert boolean array to binary image (0 and 255)
        # medial_axis = medial_axis.astype(np.uint8) * 255

        # Step 3: Find the branch points on the medial axis (these are the centers of the branching cracks)
        branch_points = cv2.findNonZero(medial_axis)
        branch_centers = [tuple(point[0]) for point in branch_points]

        # Step 4: Store branch centers in a NumPy array
        branch_centers_np = np.array(branch_centers)

        # Step 5: Use k-means clustering to find the centers of branching cracks
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)  # Set k-means criteria
        _, labels, centers = cv2.kmeans(branch_centers_np.astype(np.float32), self.num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert cluster centers to integers
        centers = np.round(centers).astype(int)
        # centers[:, 1] = self.binary_mask.shape[0] - 1 - centers[:, 1]

        # Return the pixel coordinates of the centers of branching cracks as a list of tuples
        # return [tuple(center) for center in centers]
        return centers
    
    def rearrange_centers(self, centers):
        # Calculate the pairwise Euclidean distances between centers
        distance_matrix = cdist(centers, centers, 'euclidean')

        # Apply the linear sum assignment (Traveling Salesman) algorithm
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        # Rearrange the centers according to the optimized path
        optimized_centers = centers[row_ind]

        return optimized_centers
    
    def calculate_total_distance(self, path, distance_matrix):
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += distance_matrix[path[i], path[i + 1]]
        return total_distance

    def two_opt(self, path, distance_matrix):
        best_path = path
        improved = True
        while improved:
            improved = False
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path)):
                    if j - i == 1:
                        continue
                    new_path = path[:]
                    new_path[i:j] = path[j - 1:i - 1:-1]
                    new_distance = self.calculate_total_distance(new_path, distance_matrix)
                    if new_distance < self.calculate_total_distance(best_path, distance_matrix):
                        best_path = new_path
                        improved = True
            path = best_path
        return path
    
    def optimize_path(self):
        branching_crack_centers = self.find_branch_centers()
        distance_matrix = cdist(branching_crack_centers, branching_crack_centers, 'euclidean')
        initial_path = list(range(len(branching_crack_centers)))
        optimized_path = self.two_opt(initial_path, distance_matrix)
        optimized_centers = branching_crack_centers[optimized_path]

        plt.imshow(cv2.cvtColor(cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_BGR2RGB))
        plt.plot(optimized_centers[:,0], optimized_centers[:,1], 'ro', markersize=2)
        plt.axis('off')
        plt.savefig(f"C:/Users/Administrator/Documents/GitHub/mirobot_intelrealsense/images/segmented.jpeg", bbox_inches='tight', pad_inches=0, dpi=96)

        return optimized_centers


if __name__=="__main__":
    # path = "C:/Users/Administrator/Documents/GitHub/mirobot/CFD_031.jpg"
    path = "temp_mask.jpeg"
    pb = Pixel_Branch(path)
    centers = pb.optimize_path()
    # print(centers)
    plt.imshow(cv2.cvtColor(cv2.imread(path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_BGR2RGB))
    plt.plot(centers[:,0], centers[:,1], 'ro', markersize=2)
    plt.axis('off')
    plt.savefig("temp_segmented.jpeg", bbox_inches='tight', pad_inches=0, dpi=96)
