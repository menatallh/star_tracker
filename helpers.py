import numpy as np
import cv2

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
    print(cx,cy)

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




def rotate_image_about_centroid(image, angle):
    """
    Rotate the image about its centroid by the given angle.
    """
    # Calculate the centroid
    moments = cv2.moments(image)
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
    else:
        cX, cY = image.shape[1] // 2, image.shape[0] // 2

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return rotated_image




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

def calculate_center(image):
    """
    Calculate the centroid of the image.
    The centroid is calculated based on the spatial dimensions (height, width) of the image.
    """
    # Get the height and width of the image
    height, width = image.shape[:2]  # Only consider height and width, ignore channels
    # Return the center point of the image (centroid)
    return width // 2, height // 2

def rotate_image_about_center(image, angle):
    """
    Rotate an RGB image about its centroid by the given angle.
    """

    # Calculate the centroid of the image
    cX, cY = calculate_center(image)

    # Get the rotation matrix for the centroid
    rotation_matrix = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    # Check if the image has multiple channels (i.e., RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
        # Rotate each channel (R, G, B) separately
        rotated_channels = []
        for i in range(3):  # Iterate over the 3 channels
            rotated_channel = cv2.warpAffine(image[:, :, i], rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
            rotated_channels.append(rotated_channel)

        # Stack the rotated channels back into a 3D array (RGB image)
        rotated_image = np.stack(rotated_channels, axis=2)
    else:
        # For grayscale images, just apply the rotation directly
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return rotated_image
