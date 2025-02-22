import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, Canvas, Frame, Scale
import threading
from ultralytics import YOLO
from djitellopy import Tello
import time
import subprocess
import sys
import os
from scipy.spatial import KDTree
import random

print("file is at: " + sys.executable)
# Define the base directory where the frames will be saved
base_dir = r"C:\Users\97254\PycharmProjects\Final_Project\venv"
frames_dir = os.path.join(base_dir, "frames")

# Create the frames directory if it doesn't exist
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)
    print(f"Directory created at: {frames_dir}")
else:
    print(f"Directory already exists at: {frames_dir}")

# PID Controller Class
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        if dt < 0.1:  # Minimum threshold for dt (e.g., 50ms)
            dt = 0.1
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return 0.15*(self.kp * error + self.ki * self.integral + self.kd * derivative)


# Frame size
w, h = 640, 480
fbRange = [3400,4000]


# Initialize variables for HSV and area thresholds
h_min = None
h_max = None
s_min = None
s_max = None
v_min = None
v_max = None
area_min = None
area_max = None

# Initialize YOLO model
yolo_model = YOLO('yolov8n.pt')  # Replace with your YOLOv8 model path
PERSON_CLASS_ID = 0  # YOLOv8 class ID for person
CHAIR_CLASS_ID = 56  # YOLOv8 class ID for chair
SUITCASE_CLASS_ID = 28  # YOLOv8 class ID for suitcase
last_path_time = time.time()
skip_duration = 5

# Tkinter GUI setup
def create_gui():
    global h_min, h_max, s_min, s_max, v_min, v_max, area_min, area_max
    global kp_x, ki_x, kd_x, kp_y, ki_y, kd_y

    # Initialize Tkinter
    root = tk.Tk()
    root.title("Red Hat Detection and PID Parameters")
    root.geometry("400x600")  # Adjust the window size to fit your screen
    root.configure(bg="#f7f7f7")

    # Scrollable Frame Setup
    canvas = Canvas(root, bg="#f7f7f7")
    scrollable_frame = Frame(canvas, bg="#f7f7f7")
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    # Update scrollable frame size dynamically
    def configure_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    scrollable_frame.bind("<Configure>", configure_scroll_region)

    # Initialize shared variables for HSV
    h_min = tk.IntVar(value=0)
    h_max = tk.IntVar(value=179)
    s_min = tk.IntVar(value=140)
    s_max = tk.IntVar(value=240)
    v_min = tk.IntVar(value=143)
    v_max = tk.IntVar(value=255)
    area_min = tk.IntVar(value=2000)
    area_max = tk.IntVar(value=6000)

    # Initialize shared variables for PID
    kp_x = tk.DoubleVar(value=0.25)
    ki_x = tk.DoubleVar(value=0.02)
    kd_x = tk.DoubleVar(value=0.05)
    kp_y = tk.DoubleVar(value=0.25)
    ki_y = tk.DoubleVar(value=0.02)
    kd_y = tk.DoubleVar(value=0.05)

    # HSV Threshold Controls
    hsv_frame = ttk.LabelFrame(scrollable_frame, text="HSV Thresholds", padding=(10, 5))
    hsv_frame.pack(fill="x", padx=10, pady=5)

    ttk.Label(hsv_frame, text="H Min").pack()
    tk.Scale(hsv_frame, from_=0, to=179, variable=h_min, orient="horizontal").pack(fill="x", padx=10)

    ttk.Label(hsv_frame, text="H Max").pack()
    tk.Scale(hsv_frame, from_=0, to=179, variable=h_max, orient="horizontal").pack(fill="x", padx=10)

    ttk.Label(hsv_frame, text="S Min").pack()
    tk.Scale(hsv_frame, from_=0, to=255, variable=s_min, orient="horizontal").pack(fill="x", padx=10)

    ttk.Label(hsv_frame, text="S Max").pack()
    tk.Scale(hsv_frame, from_=0, to=255, variable=s_max, orient="horizontal").pack(fill="x", padx=10)

    ttk.Label(hsv_frame, text="V Min").pack()
    tk.Scale(hsv_frame, from_=0, to=255, variable=v_min, orient="horizontal").pack(fill="x", padx=10)

    ttk.Label(hsv_frame, text="V Max").pack()
    tk.Scale(hsv_frame, from_=0, to=255, variable=v_max, orient="horizontal").pack(fill="x", padx=10)

    # Area Threshold Controls
    area_frame = ttk.LabelFrame(scrollable_frame, text="Area Thresholds", padding=(10, 5))
    area_frame.pack(fill="x", padx=10, pady=5)

    ttk.Label(area_frame, text="Min Area").pack()
    tk.Scale(area_frame, from_=100, to=10000, variable=area_min, orient="horizontal").pack(fill="x", padx=10)

    ttk.Label(area_frame, text="Max Area").pack()
    tk.Scale(area_frame, from_=500, to=30000, variable=area_max, orient="horizontal").pack(fill="x", padx=10)

    # PID Parameter Controls
    pid_frame = ttk.LabelFrame(scrollable_frame, text="PID Parameters", padding=(10, 5))
    pid_frame.pack(fill="x", padx=10, pady=5)

    ttk.Label(pid_frame, text="KP (X-axis)").pack()
    tk.Scale(pid_frame, from_=0, to=2, resolution=0.01, variable=kp_x, orient="horizontal").pack(fill="x", padx=10)

    ttk.Label(pid_frame, text="KI (X-axis)").pack()
    tk.Scale(pid_frame, from_=0, to=1, resolution=0.01, variable=ki_x, orient="horizontal").pack(fill="x", padx=10)

    ttk.Label(pid_frame, text="KD (X-axis)").pack()
    tk.Scale(pid_frame, from_=0, to=1, resolution=0.01, variable=kd_x, orient="horizontal").pack(fill="x", padx=10)

    ttk.Label(pid_frame, text="KP (Y-axis)").pack()
    tk.Scale(pid_frame, from_=0, to=2, resolution=0.01, variable=kp_y, orient="horizontal").pack(fill="x", padx=10)

    ttk.Label(pid_frame, text="KI (Y-axis)").pack()
    tk.Scale(pid_frame, from_=0, to=1, resolution=0.01, variable=ki_y, orient="horizontal").pack(fill="x", padx=10)

    ttk.Label(pid_frame, text="KD (Y-axis)").pack()
    tk.Scale(pid_frame, from_=0, to=1, resolution=0.01, variable=kd_y, orient="horizontal").pack(fill="x", padx=10)

    # Start the Tkinter event loop
    root.mainloop()


# Start GUI in a separate thread
gui_thread = threading.Thread(target=create_gui)
gui_thread.daemon = True
gui_thread.start()

# Function to detect red hats
def detect_red_hat(frame, lower_red, upper_red, area_min, area_max):
    """Detects a red hat using HSV filtering and area thresholds."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area_min <= area <= area_max:  # Check area thresholds
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

            # Draw rectangle and center circle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Draw center point

            # Display the area as text near the rectangle
            text = f"Area: {int(area)}"
            source_area = area
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)

            return (x, y, x + w, y + h), mask, area, (cx, cy)

    return None, mask, 0, None


# Function to check overlap between person and hat
def is_overlap(box1, box2):
    """Checks if two bounding boxes overlap."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    overlap_x = max(0, min(x2, x4) - max(x1, x3))
    overlap_y = max(0, min(y2, y4) - max(y1, y3))
    return overlap_x > 0 and overlap_y > 0

# Define central square boundaries
def get_central_square_bounds(frame_width, frame_height):
    square_width = frame_width // 3
    square_height = frame_height // 3
    x_start = square_width
    y_start = square_height
    x_end = 2 * square_width
    y_end = 2 * square_height
    return x_start, y_start, x_end, y_end

# Check if the red dot is within the central square
def is_in_central_square(dot_x, dot_y, central_square_bounds):
    x_start, y_start, x_end, y_end = central_square_bounds
    return x_start <= dot_x <= x_end and y_start <= dot_y <= y_end


# =========================================================
#                RRT* + Segmentation Code
#         (Previously from path_calculation.py)
# =========================================================

class RRTStar:
    def __init__(self, start, goal, obstacle_map, step_size=20, max_iter=200):
        """
        :param start: (x, y) in downsampled coordinates
        :param goal:  (x, y) in downsampled coordinates
        :param obstacle_map: single-channel (0 or 255) downsampled image
        :param step_size: how far each expansion extends
        :param max_iter: max random samples
        """
        self.start = start
        self.goal = goal
        self.obstacle_map = obstacle_map
        self.step_size = step_size
        self.max_iter = max_iter
        self.nodes = [start]
        self.parent = {start: None}
        self.tree = KDTree([start])

    def is_collision(self, point):
        x, y = int(point[0]), int(point[1])
        h, w = self.obstacle_map.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return (self.obstacle_map[y, x] == 0)  # black => obstacle
        return True

    def nearest_node(self, point):
        dist, idx = self.tree.query(point)
        return self.nodes[idx]

    def new_point(self, nearest, random_point):
        direction = np.array(random_point, dtype=float) - np.array(nearest, dtype=float)
        length = np.linalg.norm(direction)
        if length == 0:
            return nearest
        direction /= length
        new_pt = np.array(nearest, dtype=float) + direction * min(self.step_size, length)
        return tuple(new_pt.astype(int))

    def reconstruct_path(self):
        path = []
        current = self.goal
        while current is not None:
            path.append(current)
            current = self.parent[current]
        return path[::-1]

    def plan(self):
        batch_update = 50
        for i in range(self.max_iter):
            rand_pt = (
                random.randint(0, self.obstacle_map.shape[1] - 1),
                random.randint(0, self.obstacle_map.shape[0] - 1)
            )
            nearest = self.nearest_node(rand_pt)
            new = self.new_point(nearest, rand_pt)

            if not self.is_collision(new):
                self.nodes.append(new)
                self.parent[new] = nearest

                if len(self.nodes) % batch_update == 0:
                    self.tree = KDTree(self.nodes)

                dist_to_goal = np.linalg.norm(np.array(new) - np.array(self.goal))
                if dist_to_goal < self.step_size:
                    print("Goal reached!")
                    self.nodes.append(self.goal)
                    self.parent[self.goal] = new
                    return self.reconstruct_path()

        print("Failed to find a valid path after max_iter.")
        return None


def create_segmentation_map(frame, model):
    """
    Returns a black/white occupancy map:
      - 255 => free
      - 0 => obstacle
    """
    results = model(frame, classes=[0,1,2,56])
    seg_map = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255

    for result in results:
        boxes = result.boxes
        masks = result.masks
        if masks is not None and boxes is not None:
            for box, mask_data in zip(boxes, masks.data):
                binary_mask = (mask_data.cpu().numpy() > 0.5).astype(np.uint8)*255
                seg_map[binary_mask == 255] = 0
    return seg_map


def run_rrt_path_calculation(frame, person_pos, chair_pos):
    """
    Perform the entire downsample + RRT* process on the given frame
    and return a final BGR image showing the path.

    :param frame: The drone-captured frame (numpy array).
    :param person_pos: (x, y) source in the frame's coordinates
    :param chair_pos:  (x, y) goal in the frame's coordinates
    :return: final_display_bgr (an image with the path drawn).
    """

    # 1) Create segmentation map from the same YOLO model used above
    model = YOLO('yolov8n-seg.pt')
    raw_seg_map = create_segmentation_map(frame, model)

    # 2) Downsample factor
    DS = 0.25
    orig_h, orig_w = raw_seg_map.shape[:2]
    small_w = int(orig_w * DS)
    small_h = int(orig_h * DS)
    seg_map_small = cv2.resize(raw_seg_map, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

    # Scale source/target
    px_small = int(person_pos[0] * DS)
    py_small = int(person_pos[1] * DS)
    cx_small = int(chair_pos[0]  * DS)
    cy_small = int(chair_pos[1]  * DS)

    # Optional: whiten around source/target
    region_size = 5
    sx1 = max(0, px_small - region_size)
    sy1 = max(0, py_small - region_size)
    sx2 = min(small_w, px_small + region_size)
    sy2 = min(small_h, py_small + region_size)
    seg_map_small[sy1:sy2, sx1:sx2] = 255

    gx1 = max(0, cx_small - region_size)
    gy1 = max(0, cy_small - region_size)
    gx2 = min(small_w, cx_small + region_size)
    gy2 = min(small_h, cy_small + region_size)
    seg_map_small[gy1:gy2, gx1:gx2] = 255

    # Threshold => ensure strictly 0 or 255
    _, seg_map_bin_small = cv2.threshold(seg_map_small, 127, 255, cv2.THRESH_BINARY)

    # 3) Run RRT*
    rrt_star = RRTStar((px_small, py_small),
                       (cx_small, cy_small),
                       seg_map_bin_small,
                       step_size=20, max_iter=200)
    path_small = rrt_star.plan()

    # 4) Draw path on final image
    final_display = raw_seg_map.copy()
    final_display_bgr = cv2.cvtColor(final_display, cv2.COLOR_GRAY2BGR)

    if path_small:
        print("Path found (RRT*)!")
        path_full = []
        for (sx, sy) in path_small:
            fx = int(sx / DS)
            fy = int(sy / DS)
            path_full.append((fx, fy))

        # Draw path
        for i in range(len(path_full) - 1):
            cv2.line(final_display_bgr, path_full[i], path_full[i+1], (0,255,0), 2)
    else:
        print("No path found (RRT*).")

    # Mark source & target
    cv2.circle(final_display_bgr, (person_pos[0], person_pos[1]), 5, (255,0,0), -1)  # Blue for source
    cv2.circle(final_display_bgr, (chair_pos[0],  chair_pos[1]),  5, (0,0,255), -1)  # Red for target

    return final_display_bgr

# Initialize Tello drone
tello = Tello()
tello.connect()
print(f"Battery level: {tello.get_battery()}%")

# Takeoff and rise to 2.5 meters
#tello.takeoff()
time.sleep(1)
#tello.send_rc_control(0, 0, 50, 0)  # Ascend
#time.sleep(2)  # Wait for the drone to reach ~2.5m altitude
#tello.send_rc_control(0, 0, 0, 0)  # Stop vertical movement

# Start video stream
tello.streamon()
time.sleep(2)  # Wait for the stream to initialize
# 1) Setup OpenCV windows once
cv2.namedWindow("HSV Mask", cv2.WINDOW_NORMAL)
cv2.moveWindow("HSV Mask", 0, 480)
cv2.resizeWindow("HSV Mask", 640, 480)

cv2.namedWindow("Original Frame", cv2.WINDOW_NORMAL)
cv2.moveWindow("Original Frame", 640, 0)
cv2.resizeWindow("Original Frame", 640, 480)

cv2.namedWindow("RRT Path", cv2.WINDOW_NORMAL)
cv2.moveWindow("RRT Path", 640, 480)
cv2.resizeWindow("RRT Path", 640, 480)

# Main function with updated logic
try:
    # Center of the frame
    center_x, center_y = w // 2, h // 2

    # Central square bounds
    central_square_bounds = get_central_square_bounds(w, h)

    # Initialize time for PID
    last_time = time.time()
    last_command_time = time.time()

    while True:
        # Periodically send a hover command to prevent auto-landing
        if time.time() - last_command_time > 5:  # Every 5 seconds
            tello.send_rc_control(0, 0, 0, 0)
            last_command_time = time.time()

        frame = tello.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (w, h))

        if h_min and h_max and s_min and s_max and v_min and v_max:
            lower_red = np.array([h_min.get(), s_min.get(), v_min.get()])
            upper_red = np.array([h_max.get(), s_max.get(), v_max.get()])
            min_area = area_min.get()
            max_area = area_max.get()

            hat_box, mask, hat_area, hat_center = detect_red_hat(frame, lower_red, upper_red, min_area, max_area)
            results = yolo_model(frame, classes=[0,1,2,56])
            detected_person = False
            person_pos = None
            target_pos = None

            # Persist Suitcase State
            previous_target_pos = None  # Store the last known bounding box for the target

            # Inside the main loop
            for result in results:
                for box in result.boxes:
                    # Check for Suitcase Detection
                    if int(box.cls[0]) == CHAIR_CLASS_ID and box.conf[0] > 0.5:
                        print("Target detected!")
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        target_pos = (cx, cy)

                        # Update last known suitcase position
                        previous_target_pos = (x1, y1, x2, y2)

                        # Draw a red bounding box around the suitcase
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box
                        cv2.putText(frame, "Target Detected", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # Blue dot for suitcase center

                    # If no suitcase is detected, persist the last known bounding box
                    elif previous_target_pos:
                        x1, y1, x2, y2 = previous_target_pos
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # Draw the persistent red bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box
                        cv2.putText(frame, "Target Detected", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # Blue dot for suitcase center

                    # Check for Red Hat Detection
                    if hat_box and hat_center:
                        print("Person with red hat Detected!")
                        (x1, y1, x2, y2) = hat_box
                        (cx, cy) = hat_center  # Explicitly unpack the center

                        person_pos = (cx, cy)
                        cv2.putText(frame, "Person with Red Hat Detected", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Ensure red dot is drawn

                        detected_person = True

                        # Check if red dot is in the central square
                        in_central_square = is_in_central_square(cx, cy, central_square_bounds)
                        if not in_central_square:
                            # Calculate errors for PID
                            current_time = time.time()
                            dt = current_time - last_time
                            last_time = current_time

                            error_x = cx - center_x
                            #error_y = center_y - cy

                            # Create PID Controllers dynamically
                            pid_x = PIDController(kp_x.get(), ki_x.get(), kd_x.get())
                            #pid_y = PIDController(kp_y.get(), ki_y.get(), kd_y.get())

                            # Compute PID corrections
                            roll = int(pid_x.compute(error_x, dt))  # Left/Right
                            #pitch =  int(pid_y.compute(error_y, dt))  # Forward/Backward
                        else:
                            roll = 0
                        print("source area is: " + str(hat_area))
                        in_area_range = hat_area >fbRange[0] and hat_area<fbRange[1]
                        if hat_area < fbRange[0]:
                            pitch = 20
                        elif hat_area > fbRange[1]:
                            pitch = -20
                        else:
                            pitch = 0
                        if in_area_range and in_central_square:
                            print("Red dot is in the central square. Hovering.")
                            tello.send_rc_control(0, 0, 0, 0)
                        else:
                            # Send control commands to Tello
                            #tello.send_rc_control(roll, pitch, 0, 0)
                            print(f"PID Control: Roll={roll}, Pitch={pitch}")

            current_time = time.time()
            if person_pos and target_pos and current_time - last_path_time > skip_duration:
                # We unify everything in one script
                print(f"Source: {person_pos}, Target: {target_pos}. Running path calculation in-memory...")

                # Grab current frame
                current_frame = tello.get_frame_read().frame
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                current_frame = cv2.resize(current_frame, (w, h))

                final_path_img = run_rrt_path_calculation(current_frame, person_pos, target_pos)


                # Optionally show it in a new window
                cv2.imshow("RRT Path", final_path_img)

                last_path_time = current_time
            # Draw central square on the frame
            x_start, y_start, x_end, y_end = central_square_bounds
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.imshow("Original Frame", frame)
            cv2.imshow("HSV Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting gracefully...")
    tello.land()

finally:
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()


