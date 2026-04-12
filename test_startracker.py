#import tkinter as tk
#from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from dataset import *
from helpers import *
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor
from models import ViTForRegressionWithAngles  # your model class
from dataset import *  # replace with your dataset class
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import time


def decode_cos_sin_to_angle_deg(cos_sin):
    """
    cos_sin: Tensor of shape (B, 2), i.e., [cos, sin] pairs
    Returns: Tensor of angles in degrees, shape (B,)
    """
    rad = torch.atan2(cos_sin[:, 1], cos_sin[:, 0])
    deg = torch.rad2deg(rad)
    return (deg + 360) % 360  # Normalize to [0, 360)

def load_model():
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       model = ViTForRegressionWithAngles()
       checkpoint = torch.load("VitRegressionPoly_epoch120.pth", map_location=device)
       # print(checkpoint.keys())
       model.load_state_dict(checkpoint)
       model.to(device)
       model.eval()
       return model
      # Load model
model = load_model()
def evaluate_model(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()

    all_pred_ra, all_true_ra = [], []
    all_pred_dec, all_true_dec = [], []

    with torch.no_grad():
        for images, labels, angles in dataloader:
            images=images.unsqueeze(0)
            angles=angles.unsqueeze(0)
            labels=labels.unsqueeze(0)
            images = images.permute(0, 3, 1, 2).to(device)
            angles = angles.to(device)
            labels = labels.to(device)  # shape: (B, 4)

            # Forward pass
            outputs = model(images, angles)  # shape: (B, 4)

            # Decode RA and DEC
            pred_ra_deg  = decode_cos_sin_to_angle_deg(outputs[:, :2].cpu())
            true_ra_deg  = decode_cos_sin_to_angle_deg(labels[:, :2].cpu())

            pred_dec_deg = decode_cos_sin_to_angle_deg(outputs[:, 2:].cpu())
            true_dec_deg = decode_cos_sin_to_angle_deg(labels[:, 2:].cpu())

            # Collect for later stats or plotting
            all_pred_ra.append(pred_ra_deg)
            all_true_ra.append(true_ra_deg)
            all_pred_dec.append(pred_dec_deg)
            all_true_dec.append(true_dec_deg)

            # Print current batch predictions vs true
            for pra, tra, pdec, tdec in zip(pred_ra_deg, true_ra_deg, pred_dec_deg, true_dec_deg):
                print(f"RA: predicted = {pra:.2f}°, true = {tra:.2f}° | DEC: predicted = {pdec:.2f}°, true = {tdec:.2f}°")

    # Optionally return all results for error analysis
    return torch.cat(all_pred_ra), torch.cat(all_true_ra), torch.cat(all_pred_dec), torch.cat(all_true_dec)
    
    
@torch.no_grad()
def angular_loss(preds, labels):
    # preds, labels: [N, 4] (cos(RA), sin(RA), cos(DEC), sin(DEC))
    pred_ra  = torch.atan2(preds[:, 1], preds[:, 0])
    true_ra  = torch.atan2(labels[:, 1], labels[:, 0])
    ra_loss  = torch.mean(1 - torch.cos(pred_ra - true_ra))

    pred_dec = torch.atan2(preds[:, 3], preds[:, 2])
    true_dec = torch.atan2(labels[:, 3], labels[:, 2])
    dec_loss = torch.mean(1 - torch.cos(pred_dec - true_dec))

    return (ra_loss + dec_loss) / 2

def encode_labels(ra_deg, dec_deg):
    ra_rad  = torch.deg2rad(torch.tensor(ra_deg,  dtype=torch.float32))
    dec_rad = torch.deg2rad(torch.tensor(dec_deg, dtype=torch.float32))
    ra_cos,  ra_sin  = torch.cos(ra_rad),  torch.sin(ra_rad)
    dec_cos, dec_sin = torch.cos(dec_rad), torch.sin(dec_rad)
    return torch.stack([ra_cos, ra_sin, dec_cos, dec_sin])   

def adaptive_crop(img, crop_size=310, threshold_value=190):
    # Convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove non-bright stars by thresholding
    _, thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Create a mask and apply it to the original image
    mask = cv2.bitwise_and(img, img, mask=thresholded)

    # Optionally, enhance the stars
    #enhanced_stars = cv2.convertScaleAbs(mask, alpha=1.5, beta=30)

    # Convert enhanced image to grayscale to find the brightest star
    gray_enhanced = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    y_bright, x_bright = np.unravel_index(np.argmax(gray_enhanced), gray_enhanced.shape)

    # Define the initial crop position
    x_start = x_bright - crop_size // 2
    y_start = y_bright - crop_size // 2

    # Adjust crop position if out of bounds
    x_start = max(0, min(x_start, img.shape[1] - crop_size))
    y_start = max(0, min(y_start, img.shape[0] - crop_size))

    # Ensure the brightest star is within the crop area
    if x_bright < x_start or x_bright > x_start + crop_size or y_bright < y_start or y_bright > y_start + crop_size:
        if x_bright < x_start:
            x_start = max(0, x_bright - crop_size // 2)
        elif x_bright > x_start + crop_size:
            x_start = min(img.shape[1] - crop_size, x_bright - crop_size // 2)

        if y_bright < y_start:
            y_start = max(0, y_bright - crop_size // 2)
        elif y_bright > y_start + crop_size:
            y_start = min(img.shape[0] - crop_size, y_bright - crop_size // 2)

    # Crop the image
    cropped_img = mask[y_start:y_start + crop_size, x_start:x_start + crop_size]

    return cropped_img

def center_crop_around_centroid(image, target_size, threshold_value=190):

    h, w = target_size
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove non-bright stars by thresholding
    _, thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Create a mask and apply it to the original image
    mask = cv2.bitwise_and(img, img, mask=thresholded)

    # Optionally, enhance the stars
    #enhanced_stars = cv2.convertScaleAbs(mask, alpha=1.5, beta=30)

    # Convert enhanced image to grayscale to find the brightest star
    gray_enhanced = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    center_x,center_y=calculate_centroid(gray_enhanced)
    center_x,center_y=int(center_x),int(center_y)
    crop_y1 = max(center_y - h // 2, 0)
    crop_y2 = min(center_y + h // 2, image.shape[0])
    crop_x1 = max(center_x - w // 2, 0)
    crop_x2 = min(center_x + w // 2, image.shape[1])

    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    return cropped_image


def calculate_centroid(image):
    # Convert image to grayscale
    #grayscale_image = image.convert("L")
    np_image = np.array(image)

    # Calculate the coordinates of the centroid
    y_indices, x_indices = np.nonzero(np_image)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    cx = np.mean(x_indices)
    cy = np.mean(y_indices)
    #print(cx,cy)

    return cx, cy
    
def filter_contours(img_gray):
    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for valid contours
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    # Filter contours based on the criteria
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the original image
    filtered_image = cv2.bitwise_and(img_gray, img_gray, mask=mask)



    return filtered_image    
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_centroid(image):
    """
    Calculate the centroid of the image based on brightness values.

    Args:
        image (np.array): The input grayscale image.

    Returns:
        tuple: The (x, y) coordinates of the centroid.
    """
    # Compute the weighted average of pixel coordinates
    total_intensity = np.sum(image)
    y_coords, x_coords = np.indices(image.shape)
    x_centroid = np.sum(x_coords * image) / total_intensity
    y_centroid = np.sum(y_coords * image) / total_intensity
    return (int(x_centroid), int(y_centroid))

def get_brightest_stars(image, num_stars=6):
    """
    Get the coordinates and brightness of the brightest stars.

    Args:
        image (np.array): The input grayscale image.
        num_stars (int): Number of brightest stars to extract.

    Returns:
        list: A list of tuples where each tuple contains (x, y, brightness) of a star.
    """
    # Flatten the image to get intensity values
    intensities = image.flatten()

    # Get the indices of the brightest stars
    top_indices = np.argsort(intensities)[-num_stars:]

    # Get the coordinates of these indices
    y_coords, x_coords = np.unravel_index(top_indices, image.shape)

    # Get the brightness values
    brightest_stars = [(x_coords[i], y_coords[i], intensities[top_indices[i]]) for i in range(num_stars)]

    return brightest_stars

def is_star_in_cropped_image(star_coords, crop_bounds):
    """
    Check if the star coordinates are within the cropped image bounds.

    Args:
        star_coords (list): List of (x, y) coordinates of stars.
        crop_bounds (tuple): Bounds of the cropped image as (x1, x2, y1, y2).

    Returns:
        bool: True if all stars are inside the cropped image, False otherwise.
    """
    x1, x2, y1, y2 = crop_bounds
    return all(x1 <= x <= x2 and y1 <= y <= y2 for x, y, _ in star_coords)

def crop_around_centroid(image, centroid, crop_size=100):
    """
    Crop the image around the given centroid.

    Args:
        image (np.array): The input grayscale image.
        centroid (tuple): The (x, y) coordinates of the centroid.
        crop_size (int): The size of the square crop.

    Returns:
        tuple: The cropped image and the bounds of the cropped image.
    """
    cx, cy = centroid
    half_crop_size = crop_size // 2

    # Define the crop boundaries
    x1 = max(cx - half_crop_size, 0)
    x2 = min(cx + half_crop_size, image.shape[1])
    y1 = max(cy - half_crop_size, 0)
    y2 = min(cy + half_crop_size, image.shape[0])

    # Crop the image
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image, (x1, x2, y1, y2)


import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_angle(point1, point2, point3):
    """
    Calculate the angle between three points: point1, point2, and point3.
    Returns the angle in degrees.
    """
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)

    # Normalize the vectors
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)

    # Calculate the angle using the dot product
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(dot_product) * (180.0 / np.pi)

    return angle

def remove_collinear_points(points):
    """
    Remove points that lie on the same straight line (collinear).
    Only the first and last points of a collinear segment are kept.
    """
    if len(points) < 3:
        return points

    filtered_points = [points[0]]

    for i in range(1, len(points) - 1):
        angle = calculate_angle(points[i - 1], points[i], points[i + 1])

        # If the angle is not 0 (i.e., points are not collinear), keep the point
        if not np.isclose(angle, 0):
            filtered_points.append(points[i])

    filtered_points.append(points[-1])
    return filtered_points

def calculate_angles_between_points(points):
    """
    Calculate angles between consecutive points in a list of points.
    """
    angles = []

    for i in range(1, len(points) - 1):
        angle = calculate_angle(points[i - 1], points[i], points[i + 1])
        angles.append(angle)

    return angles
    
    
def cluster_and_merge_bright_pixels(coords, eps=2):
    coords = np.array(coords)
    # Use DBSCAN to cluster nearby points
    clustering = DBSCAN(eps=eps, min_samples=1).fit(coords)
    merged_coords = []

    for cluster_label in np.unique(clustering.labels_):
        cluster_points = coords[clustering.labels_ == cluster_label]
        # Calculate centroid of the cluster
        centroid = np.mean(cluster_points, axis=0)
        merged_coords.append(tuple(centroid))

    return merged_coords    
    
    
import cv2
import numpy as np
import math
from scipy.spatial import ConvexHull

# Helper function to calculate angle at point 'b' given points 'a', 'b', and 'c'
def calculate_angle(a, b, c):
    ab = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    magnitude_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if magnitude_ab == 0 or magnitude_bc == 0:
        return 0

    cos_angle = dot_product / (magnitude_ab * magnitude_bc)
    try :
     angle = math.degrees(math.acos(cos_angle))
    except:
      angle=0
    return angle

# Function to calculate polygon angles and fill up to 4 or 8 angles
def calculate_polygon_angles(polygon_points, num_angles=4):
    angles = []
    num_points = len(polygon_points)
    for i in range(num_points):
        prev_point = polygon_points[i - 1]
        current_point = polygon_points[i]
        next_point = polygon_points[(i + 1) % num_points]
        angle = calculate_angle(prev_point, current_point, next_point)
        angles.append(angle)

    # Fill the rest of the angles with zeros if not enough points
    while len(angles) < num_angles:
        angles.append(0)

    return angles

# Function to form a convex hull (non-crossing polygon)
def form_convex_hull_polygon(points, num_points_to_find=4):
    if len(points) < 3:
        return points, []  # Not enough points for a polygon

    points = np.array(points)
    hull = ConvexHull(points)

    # Get the convex hull points (indices) and corresponding polygon points
    polygon_points = points[hull.vertices].tolist()

    # If the convex hull has more than the required points, pick the first num_points_to_find points
    polygon_points = polygon_points[:num_points_to_find]

    # Remaining points are those that are not part of the hull
    remaining_points = [point for point in points if list(point) not in polygon_points]

    return polygon_points, remaining_points

# Function to find the closest two points based on x or y distance
def find_closest_by_axis(points):
    min_distance = float('inf')
    closest_pair = (None, None)

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            # Calculate both x and y distance
            x_distance = abs(points[i][0] - points[j][0])
            y_distance = abs(points[i][1] - points[j][1])

            # Choose the minimum of x or y distance
            distance = min(x_distance, y_distance)

            if distance < min_distance:
                min_distance = distance
                closest_pair = (points[i], points[j])

    return closest_pair

# Main function to form polygons based on number of points and calculate angles
def form_polygons_and_calculate_angles(points):
    num_points = len(points)

    if num_points < 2:
        return [], [], [0]*8  # Not enough points for a polygon or line

    # Form polygons based on the number of points
    if num_points == 8:
        polygon1, remaining_points = form_convex_hull_polygon(points, 4)
        polygon2, _ = form_convex_hull_polygon(remaining_points, 4)
    elif num_points == 7:
        polygon1, remaining_points = form_convex_hull_polygon(points, 4)
        polygon2, _ = form_convex_hull_polygon(remaining_points, 3)
    elif num_points == 6:
        # Find the closest two points by x or y and form a line
        closest_pair = find_closest_by_axis(points)
        remaining_points = [p for p in points if p not in closest_pair]

        # Form a polygon with the remaining 4 points
        polygon1, _ = form_convex_hull_polygon(remaining_points, 4)
        polygon2 = closest_pair
    elif num_points == 5:
        polygon1, remaining_points = form_convex_hull_polygon(points, 4)
        polygon2 = remaining_points  # The remaining point is a single point
    elif num_points == 4:
        polygon1, remaining_points = form_convex_hull_polygon(points, 4)
        polygon2 = []
    elif num_points == 3:
        polygon1, remaining_points = form_convex_hull_polygon(points, 3)
        polygon2 = []
    else:  # Case for num_points == 2 (draw line)
        polygon1 = points
        polygon2 = []

    # Calculate angles and merge them into one list of 8
    angles_polygon1 = calculate_polygon_angles(polygon1, 4)
    angles_polygon2 = calculate_polygon_angles(polygon2, 4) if polygon2 else [0] * 4

    all_angles = angles_polygon1 + angles_polygon2
    return polygon1, polygon2, all_angles

# Function to draw polygons on the 224x224 black image
def draw_polygons(black_image_with_stars, polygon1, polygon2):
    # Draw the first polygon in red
    if len(polygon1) >= 3:
        for i in range(len(polygon1)):
            pt1 = tuple(map(int, polygon1[i]))
            pt2 = tuple(map(int, polygon1[(i + 1) % len(polygon1)]))
            cv2.line(black_image_with_stars, pt1, pt2, (0, 0, 255), 1)
    elif len(polygon1) == 2:
        pt1 = tuple(map(int, polygon1[0]))
        pt2 = tuple(map(int, polygon1[1]))
        cv2.line(black_image_with_stars, pt1, pt2, (0, 0, 255), 1)

    # Draw the second polygon in green or the closest pair of points in green
    if polygon2 and len(polygon2) == 2:
        pt1 = tuple(map(int, polygon2[0]))
        pt2 = tuple(map(int, polygon2[1]))
        cv2.line(black_image_with_stars, pt1, pt2, (0, 0, 255), 1)
    elif len(polygon2) >= 3:
        for i in range(len(polygon2)):
            pt1 = tuple(map(int, polygon2[i]))
            pt2 = tuple(map(int, polygon2[(i + 1) % len(polygon2)]))
            cv2.line(black_image_with_stars, pt1, pt2, (0, 0, 255), 1)

    return black_image_with_stars    

def process_star_image(image_path, crop_size=224):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    thresholded = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=11, C=2
    )
    mask = cv2.bitwise_and(image, image, mask=thresholded)
    original_centroid = calculate_centroid(mask)
    cropped_image, _ = crop_around_centroid(mask, original_centroid, crop_size)
    brightest_stars_cropped = get_brightest_stars(cropped_image, num_stars=20)
    brightest_star_coords_cropped = [(x, y) for x, y, _ in brightest_stars_cropped]
    brightest_star_coords_cropped_final = cluster_and_merge_bright_pixels(brightest_star_coords_cropped, eps=2)
    new_centroid = calculate_centroid(cropped_image)

    black_image_with_stars = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 3), dtype=np.uint8)
    for x, y in brightest_star_coords_cropped_final:
        cv2.circle(black_image_with_stars, (int(x), int(y)), 3, (255, 0, 0), -1)

    centroid_x, centroid_y = int(new_centroid[0]), int(new_centroid[1])
    cv2.circle(black_image_with_stars, (centroid_x, centroid_y), 3, (0, 0, 255), -1)

    return black_image_with_stars
def process_inference_data(image,labels,polygone_angles):

        masked_image= torch.tensor(image).to(torch.float32)


        angles = labels # Ensure angles are float
        angles=torch.tensor(angles)
        angles = torch.fmod(angles+ 360, 360)
        angle_label = encode_labels(angles[0], angles[1]) 
        

        angles_between_points = polygone_angles





        # Convert the list of angles to a PyTorch tensor
        angles_between_points = torch.tensor(angles_between_points, dtype=torch.float32)

        angles_between_points = torch.fmod(angles_between_points + 360, 360)

        # Normalize angles using cosine to get values between 0 and 1
        angles_between_points_rad = torch.deg2rad(angles_between_points)
        enc = torch.stack([torch.sin(angles_between_points_rad), torch.cos(angles_between_points_rad)], dim=1).flatten()  # size = 2*N
        # Create a padded tensor with a size of 8 (if needed)
        padded_tensor = torch.zeros(16)
        padded_tensor[:enc.size(0)] = enc
          # size = 2*N
        # Now you can continue with the rest of your training or inference process





        #print(padded_tensor.shape)

        return masked_image, angle_label,padded_tensor    
    
def process_features_image(image_path, crop_size=224):
    # Load the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply adaptive thresholding to remove non-bright stars
    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, blockSize=11, C=2)

    # Create a mask and apply it to the original image
    mask = cv2.bitwise_and(image, image, mask=thresholded)

    # Calculate the original centroid based on brightness values
    original_centroid = calculate_centroid(mask)
    #print("Original Centroid:", original_centroid)

    # Crop the image around the centroid
    cropped_image, crop_bounds = crop_around_centroid(mask, original_centroid, crop_size)


    # Get the brightest stars in the cropped image
    brightest_stars_cropped = get_brightest_stars(cropped_image, num_stars=20)

    # Extract coordinates for clustering
    brightest_star_coords_cropped = [(x, y) for x, y, _ in brightest_stars_cropped]

    # Apply the clustering function to remove adjacent bright pixels
    brightest_star_coords_cropped_final = cluster_and_merge_bright_pixels(brightest_star_coords_cropped, eps=2)

    # Calculate the new centroid in the cropped image
    new_centroid = calculate_centroid(cropped_image)
    #print("New Centroid:", new_centroid)

    # Sort by proximity to the new centroid and keep the closest 8 points
    if len(brightest_star_coords_cropped_final) > 8:
        brightest_star_coords_cropped_final.sort(key=lambda coord: np.linalg.norm(np.array(coord) - np.array(new_centroid)))
        brightest_star_coords_cropped_final = brightest_star_coords_cropped_final[:8]

    # Create a black image with 3 channels (for RGB color)
    black_image_with_stars = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 3), dtype=np.uint8)

    # Draw the brightest stars as red circles
    for x, y in brightest_star_coords_cropped_final:
        x, y = int(x), int(y)
        cv2.circle(black_image_with_stars, (x, y), 3, (255, 0, 0), -1)  # Red circles for stars
    angles = calculate_angles_between_points(brightest_star_coords_cropped_final)

    # Draw the new centroid as a blue circle
    polygon1, polygon2, all_angles = form_polygons_and_calculate_angles(brightest_star_coords_cropped_final)

    # Draw the polygons on the black image
    image_with_polygons = draw_polygons(black_image_with_stars, polygon1, polygon2)


    # Draw the points and polygons on the black image
    #black_image_with_stars= draw_polygons(black_image_with_stars, polygon1, polygon2, brightest_star_coords_cropped_final)



    centroid_x, centroid_y = int(new_centroid[0]), int(new_centroid[1])
    cv2.circle(black_image_with_stars, (centroid_x, centroid_y), 3, (0, 0, 255), -1)  # Blue circle for centroid



    return black_image_with_stars, new_centroid,angles,all_angles


import os
import pandas as pd
import torch

# assumes these already exist in your codebase
# from helpers import *
# from dataset import *
# from models import ViTForRegressionWithAngles
# from processing import process_star_image, process_features_image

# --------------------------------------------------
# Evaluation Runner (NO GUI)
# --------------------------------------------------

class StarFinderEvaluator:
    def __init__(self, csv_path, image_root, batch_size=5):
        self.csv_path = csv_path
        self.image_root = image_root
        self.batch_size = batch_size

        self.csv_data = pd.read_csv(csv_path)
        self.current_index = 0

    def load_images_batch(self):
        start = self.current_index
        end = min(start + self.batch_size, len(self.csv_data))

        images = []
        ras = []
        decs = []

        for i in range(start, end):
            ra = self.csv_data.iloc[i]["raan"]
            dec = self.csv_data.iloc[i]["dec"]
            file_name = os.path.basename(self.csv_data.iloc[i]["image_path"])
            image_path = os.path.join(self.image_root, file_name)

            if os.path.exists(image_path):
                images.append(image_path)
                ras.append(ra)
                decs.append(dec)
            else:
                print(f"[WARN] Image not found: {image_path}")

        self.current_index += self.batch_size
        return images, ras, decs

    def run_evaluation(self):
        images, ras, decs = self.load_images_batch()

        if not images:
            print("[INFO] No images to process.")
            return

        # Feature extraction
        processed_images = []
        polygon_angles = []

        for img in images:
            processed, _, _, angles = process_features_image(img)
            processed_images.append(processed)
            polygon_angles.append(angles)

        labels = list(zip(ras, decs))

        # Prepare inference dataset
        inference_data = [
            process_inference_data(image, label, angles)
            for image, label, angles in zip(processed_images, labels, polygon_angles)
        ]

        # Run model
        model = load_model()
        pred_ra, true_ra, pred_dec, true_dec = evaluate_model(
            model, inference_data, device="cpu"
        )

        # Print results
        for i, (pra, tra, pdec, tdec) in enumerate(
            zip(pred_ra, true_ra, pred_dec, true_dec)
        ):
            print(
                f"[{i}] "
                f"RA: {pra:.2f}° / {tra:.2f}° | "
                f"DEC: {pdec:.2f}° / {tdec:.2f}°"
            )
        metrics = compute_accuracy(pred_ra, true_ra,pred_dec, true_dec,threshold_deg=5
        )

        print("\n=== Evaluation Metrics ===")
        print(f"RA  MAE  : {metrics['ra_mae']:.2f}°")
        print(f"DEC MAE  : {metrics['dec_mae']:.2f}°")
        print(f"RA  Acc≤5° : {metrics['ra_acc']:.2f}%")
        print(f"DEC Acc≤5° : {metrics['dec_acc']:.2f}%")
        print(f"Both Acc≤5°: {metrics['both_acc']:.2f}%")

import torch

def angular_error_deg(pred, true):
    """
    pred, true: tensors in degrees [0, 360)
    returns absolute angular error in degrees
    """
    diff = torch.abs(pred - true)
    return torch.minimum(diff, 360 - diff)

def compute_accuracy(pred_ra, true_ra, pred_dec, true_dec, threshold_deg=5):
    ra_error  = angular_error_deg(pred_ra, true_ra)
    dec_error = angular_error_deg(pred_dec, true_dec)

    ra_mae  = ra_error.mean().item()
    dec_mae = dec_error.mean().item()

    ra_acc  = (ra_error <= threshold_deg).float().mean().item() * 100
    dec_acc = (dec_error <= threshold_deg).float().mean().item() * 100

    both_acc = ((ra_error <= threshold_deg) &
                (dec_error <= threshold_deg)).float().mean().item() * 100

    return {
        "ra_mae": ra_mae,
        "dec_mae": dec_mae,
        "ra_acc": ra_acc,
        "dec_acc": dec_acc,
        "both_acc": both_acc
    }





# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    evaluator = StarFinderEvaluator(
        csv_path="ra-dec.csv",
        image_root="star-images",
        batch_size=500
    )

    evaluator.run_evaluation()
