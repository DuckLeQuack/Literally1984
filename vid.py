import cv2
import numpy as np
import os
from datetime import timedelta

# Original coordinates (screen resolution 1920x1080)
scanner_pts_original = np.array([
    [670, 475],
    [1285, 460],
    [1290, 1070],
    [680, 1070]
], dtype=np.int32)

# Define the screen resolution (1920x1080) and video resolution (854x480)
original_width = 1920
original_height = 1080
new_width = 854  # put in actual video resolution width
new_height = 480  # put in actual video resolution height

# Function to remap coordinates
def remap_coordinates(coords, original_width, original_height, new_width, new_height):
    remapped_coords = []
    for x, y in coords:
        new_x = int((x / original_width) * new_width)
        new_y = int((y / original_height) * new_height)
        remapped_coords.append([new_x, new_y])
    return np.array(remapped_coords, dtype=np.int32)

# Remap the coordinates
scanner_pts = remap_coordinates(scanner_pts_original, original_width, original_height, new_width, new_height)

def is_object_in_scanner(frame, scanner_mask, bg_subtractor, threshold=1700):
    fg_mask = bg_subtractor.apply(frame)
    scanner_fg = cv2.bitwise_and(fg_mask, fg_mask, mask=scanner_mask)
    motion_pixels = cv2.countNonZero(scanner_fg)
    return motion_pixels > threshold

def create_scanner_mask(frame_shape, scanner_pts):
    mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [scanner_pts], 255)
    return mask

# Function to format timestamp as HHMMSS
def format_timestamp(timestamp):
    return str(timedelta(seconds=timestamp)).split(".")[0].replace(":", "")

def process_and_save_video(video_path):
    cap = cv2.VideoCapture(video_path)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    scanner_mask = create_scanner_mask(first_frame.shape[:2], scanner_pts)

    # Create the "screenshots" folder if it doesn't exist
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")

    # Define output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter('scanner_output.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    frame_index = 0
    red_duration = 0  # Track the duration for which the rectangle is red (object detected)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        object_detected = is_object_in_scanner(frame, scanner_mask, bg_subtractor)

        # If the rectangle is red, start tracking the time
        if object_detected:
            red_duration += 1 / fps  # Increase time in seconds
        else:
            red_duration = 0  # Reset if object is no longer detected

        # If the rectangle stays red for more than 1.5 seconds, take a screenshot
        if red_duration > 1.8:
            timestamp = frame_index / fps
            formatted_timestamp = format_timestamp(timestamp)
            screenshot_filename = f"screenshots/screenshot_{timestamp:.2f}.png"
            cv2.imwrite(screenshot_filename, frame)  # Save the screenshot
            print(f"Screenshot taken at {timestamp:.2f}s: {screenshot_filename}")
            red_duration = 0  # Reset the timer after taking the screenshot

        # Display timestamp
        timestamp = frame_index / fps
        cv2.putText(frame, f"Time: {timestamp:.2f}s", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if object_detected:
            cv2.putText(frame, "Object Detected", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write the frame to output video
        output.write(frame)

        frame_index += 1

    cap.release()
    output.release()

# Example usage:
process_and_save_video("vid/Nesten alle varer sakte tempo 480P.mp4")
