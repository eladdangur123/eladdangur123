import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, Canvas, Frame
import threading
from ultralytics import YOLO
from djitellopy import Tello
import time
import sys
import os
from scipy.spatial import KDTree
import random
import math
from math import sqrt

print("file is at:", sys.executable)

# ============ PATH SETUP ============
base_dir = os.path.join(os.path.expanduser('~'), "PycharmProjects", "Drone", "venv")
frames_dir = os.path.join(base_dir, "frames")
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)
    print(f"Directory created at: {frames_dir}")
else:
    print(f"Directory already exists at: {frames_dir}")

# ============ GLOBAL VARIABLES ============

display_frame = None       # "Original Frame"
display_mask = None        # HSV mask
display_rrt = None         # RRT Path image (if a target is found)

stop_flag = False          # Signal to stop the background thread

# We optionally recalc the path every 5 seconds (you set skip_duration=3 now)
last_path_time = time.time()
skip_duration = 3

# Frame width & height
w, h = 640, 480

fbRange = [800, 1700]     # Reference for forward/back control

PERSON_CLASS_ID = 0
CHAIR_CLASS_ID = 56
SUITCASE_CLASS_ID = 28
BICYCLE_CLASS_ID = 1

# Tkinter GUI variables
h_min = None
h_max = None
s_min = None
s_max = None
v_min = None
v_max = None
area_min = None
area_max = None

kp_x = None
ki_x = None
kd_x = None
kp_y = None
ki_y = None
kd_y = None

# ============ PID CONTROLLER ============
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        if dt < 0.1:
            dt = 0.1
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error

        # 0.15 is an overall scaling factor for smaller speeds
        return 0.15 * (self.kp * error + self.ki * self.integral + self.kd * derivative)


# ============ RRT* + Segmentation CODE ============

class RRTStar:
    def __init__(self, start, goal, obstacle_map, step_size=2, max_iter=3000):
        """
        step_size => how far each new node extends
        max_iter  => total # of random samples (nodes) to attempt
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
        h_, w_ = self.obstacle_map.shape[:2]
        if 0 <= x < w_ and 0 <= y < h_:
            return (self.obstacle_map[y, x] == 0)  # black => obstacle
        return True

    def nearest_node(self, point):
        dist, idx = self.tree.query(point)
        return self.nodes[idx]

    def new_point(self, nearest, random_point):
        """Extend from 'nearest' toward 'random_point' by step_size."""
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
        for _ in range(self.max_iter):
            rand_pt = (
                random.randint(0, self.obstacle_map.shape[1] - 1),
                random.randint(0, self.obstacle_map.shape[0] - 1)
            )
            nearest = self.nearest_node(rand_pt)
            new = self.new_point(nearest, rand_pt)

            # If no collision => add node
            if not self.is_collision(new):
                self.nodes.append(new)
                self.parent[new] = nearest

                # Rebuild KDTree every 'batch_update' nodes
                if len(self.nodes) % batch_update == 0:
                    self.tree = KDTree(self.nodes)

                # Check if new node is close enough to goal
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
    Creates a black/white occupancy map: 255 => free, 0 => obstacle
    Based on YOLO segmentation if available.
    """
    results = model(frame, classes=[0,1,2,3,11,56])
    seg_map = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255

    for result in results:
        boxes = result.boxes
        masks = result.masks
        if masks is not None and boxes is not None:
            for box, mask_data in zip(boxes, masks.data):
                binary_mask = (mask_data.cpu().numpy() > 0.5).astype(np.uint8)*255
                seg_map[binary_mask == 255] = 0
    return seg_map


def run_rrt_path_calculation_realtime(frame, user_pos, target_pos):
    """
    Replicates the 'real-time' RRT approach from the pygame script,
    but uses OpenCV & your segmentation map for obstacles.
    Also, we whiten (make free) a region around the user & target
    in the downsampled map so the RRT can start/end properly.
    """
    # 1) Create segmentation map from YOLO
    model = YOLO('/Users/eladdangur/PycharmProjects/Drone/venv/yolov8n-seg.pt')
    raw_seg_map = create_segmentation_map(frame, model)
    # raw_seg_map: 255=free, 0=obstacle

    # 2) Downsample for speed
    DS = 0.3
    seg_h, seg_w = raw_seg_map.shape[:2]
    small_w = int(seg_w * DS)
    small_h = int(seg_h * DS)
    seg_map_small = cv2.resize(raw_seg_map, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

    # Convert user & target to downsampled coords
    ux_small = int(user_pos[0] * DS)
    uy_small = int(user_pos[1] * DS)
    tx_small = int(target_pos[0] * DS)
    ty_small = int(target_pos[1] * DS)

    # -------------------------------
    # 2.5) Whiten a region around start/goal so they are not stuck in black
    region_size = 20  # how many pixels around user/target to force as free
    # Whiten around user
    sx1 = max(0, ux_small - region_size)
    sy1 = max(0, uy_small - region_size)
    sx2 = min(small_w - 1, ux_small + region_size)
    sy2 = min(small_h - 1, uy_small + region_size)
    seg_map_small[sy1:sy2+1, sx1:sx2+1] = 255

    # Whiten around target
    gx1 = max(0, tx_small - region_size)
    gy1 = max(0, ty_small - region_size)
    gx2 = min(small_w - 1, tx_small + region_size)
    gy2 = min(small_h - 1, ty_small + region_size)
    seg_map_small[gy1:gy2+1, gx1:gx2+1] = 255
    # -------------------------------

    # 3) Prepare data structures for RRT
    start = (ux_small, uy_small)
    goal  = (tx_small, ty_small)
    parent = {start: None}
    found  = False

    goal_threshold = 10    # how close we must get to goal (downsampled coords)
    Step = 2              # step size in downsampled coords
    max_iter = 4000

    def line_collision_check(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return True  # same point
        for i in range(1, steps+1):
            t = i / float(steps)
            x = int(x1 + dx * t)
            y = int(y1 + dy * t)
            # bounds check
            if x < 0 or x >= small_w or y < 0 or y >= small_h:
                return False
            # obstacle check
            if seg_map_small[y, x] == 0:
                return False
        return True

    def nearest_node(px, py):
        min_dist = 1e9
        best = None
        for nd in parent.keys():
            dist = math.hypot(nd[0]-px, nd[1]-py)
            if dist < min_dist:
                min_dist = dist
                best = nd
        return best

    # 4) RRT main loop
    nodes = [start]
    import random
    for _ in range(max_iter):
        rx = random.randint(0, small_w-1)
        ry = random.randint(0, small_h-1)
        if seg_map_small[ry, rx] == 0:
            continue  # skip obstacles
        nn = nearest_node(rx, ry)
        if nn is None:
            continue
        dx = rx - nn[0]
        dy = ry - nn[1]
        dist = math.hypot(dx, dy)
        if dist == 0:
            continue
        dx /= dist
        dy /= dist
        stepsize = min(Step, dist)
        newx = int(nn[0] + dx * stepsize)
        newy = int(nn[1] + dy * stepsize)
        if line_collision_check(nn, (newx, newy)):
            new_node = (newx, newy)
            parent[new_node] = nn
            nodes.append(new_node)
            # check goal
            gdist = math.hypot(newx - goal[0], newy - goal[1])
            if gdist < goal_threshold:
                # try connecting directly to goal
                if line_collision_check(new_node, goal):
                    parent[goal] = new_node
                    nodes.append(goal)
                    found = True
                    break

    # 5) Convert seg_map to color for final drawing
    final_display = cv2.cvtColor(raw_seg_map, cv2.COLOR_GRAY2BGR)

    # 6) Draw entire RRT in blue
    DS_inv = 1.0 / DS
    for node, par in parent.items():
        if par is not None:
            x1 = int(node[0] * DS_inv)
            y1 = int(node[1] * DS_inv)
            x2 = int(par[0] * DS_inv)
            y2 = int(par[1] * DS_inv)
            #cv2.line(final_display, (x1,y1), (x2,y2), (255,0,0), 1)

    # 7) If found, reconstruct path in green
    if found:
        print("Goal reached with real-time RRT!")
        path_nodes = []
        cur = goal
        while cur is not None:
            path_nodes.append(cur)
            cur = parent[cur]
        path_nodes.reverse()
        for i in range(1, len(path_nodes)):
            px, py = path_nodes[i-1]
            cx, cy = path_nodes[i]
            pX = int(px * DS_inv)
            pY = int(py * DS_inv)
            cX = int(cx * DS_inv)
            cY = int(cy * DS_inv)
            cv2.line(final_display, (pX, pY), (cX, cY), (0,255,0), 2)
            cv2.circle(final_display, (cX, cY), 3, (0,255,0), -1)
    else:
        print("No path found after max_iter in real-time RRT.")

    # 8) Mark user (blue) & target (red)
    cv2.circle(final_display, user_pos, 6, (255,0,0), -1)
    cv2.putText(final_display, "User", (user_pos[0]+8, user_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.circle(final_display, target_pos, 6, (0,0,255), -1)
    cv2.putText(final_display, "Destination", (target_pos[0]+8, target_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    return final_display



# ============ UTILITY ============

def distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return sqrt(dx*dx + dy*dy)


def detect_red_hat(frame, lower_red, upper_red, area_min, area_max):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_area = 0
    best_center = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area_min <= area <= area_max and area > best_area:
            x, y, w_, h_ = cv2.boundingRect(cnt)
            cx, cy = x + w_//2, y + h_//2
            best_box = (x, y, x+w_, y+h_)
            best_area = area
            best_center = (cx, cy)
    return best_box, mask, best_area, best_center


# ============ DRAW ONLY THE BOX UNDERNEATH THE CENTER BOX ============

def draw_lower_center_box(frame):
    """
    Instead of a 3x3 grid, we only draw the box UNDER the center box.
    That means the center box is row=1,col=1 in the 3x3,
    the box underneath it is row=2,col=1 (0-based).
    """
    box_w = w // 3
    box_h = h // 3

    # The "center" box is at row=1, col=1
    # The box directly underneath it is row=2, col=1
    # We'll draw a green rectangle for row=2,col=1
    box_x = box_w * 1
    box_y = box_h * 2
    cv2.rectangle(
        frame,
        (box_x, box_y),
        (box_x + box_w, box_y + box_h),
        (0, 255, 0), 2
    )

def get_lower_center_box_bounds():
    """
    Return the bounding box (x_start, y_start, x_end, y_end)
    for the box under the center box in a 3x3 grid.
    """
    box_w = w // 3
    box_h = h // 3
    box_x = box_w * 1
    box_y = box_h * 2
    return (box_x, box_y, box_x + box_w, box_y + box_h)


def is_in_lower_center_box(dot_x, dot_y):
    """
    Check if a point is inside that green box under the center box.
    """
    x_s, y_s, x_e, y_e = get_lower_center_box_bounds()
    return x_s <= dot_x <= x_e and y_s <= dot_y <= y_e


# ============ TKINTER GUI SETUP ============

def create_gui(root):
    global h_min, h_max, s_min, s_max, v_min, v_max
    global area_min, area_max
    global kp_x, ki_x, kd_x, kp_y, ki_y, kd_y

    root.title("Red Hat Detection and PID Parameters")
    root.geometry("400x600")
    root.configure(bg="#f7f7f7")

    canvas = Canvas(root, bg="#f7f7f7")
    scrollable_frame = Frame(canvas, bg="#f7f7f7")
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0,0), window=scrollable_frame, anchor="nw")

    def configure_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    scrollable_frame.bind("<Configure>", configure_scroll_region)

    # HSV
    h_min = tk.IntVar(value=150)
    h_max = tk.IntVar(value=179)
    s_min = tk.IntVar(value=80)
    s_max = tk.IntVar(value=255)
    v_min = tk.IntVar(value=71)
    v_max = tk.IntVar(value=255)
    area_min = tk.IntVar(value=150)
    area_max = tk.IntVar(value=930)

    # PID Gains
    kp_x = tk.DoubleVar(value=0.25)
    ki_x = tk.DoubleVar(value=0.02)
    kd_x = tk.DoubleVar(value=0.05)

    kp_y = tk.DoubleVar(value=0.25)
    ki_y = tk.DoubleVar(value=0.02)
    kd_y = tk.DoubleVar(value=0.05)

    hsv_frame = ttk.LabelFrame(scrollable_frame, text="HSV Thresholds", padding=(10,5))
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

    area_frame = ttk.LabelFrame(scrollable_frame, text="Area Thresholds", padding=(10,5))
    area_frame.pack(fill="x", padx=10, pady=5)
    ttk.Label(area_frame, text="Min Area").pack()
    tk.Scale(area_frame, from_=100, to=10000, variable=area_min, orient="horizontal").pack(fill="x", padx=10)
    ttk.Label(area_frame, text="Max Area").pack()
    tk.Scale(area_frame, from_=500, to=30000, variable=area_max, orient="horizontal").pack(fill="x", padx=10)

    pid_frame = ttk.LabelFrame(scrollable_frame, text="PID Parameters", padding=(10,5))
    pid_frame.pack(fill="x", padx=10, pady=5)

    ttk.Label(pid_frame, text="KP (Horizontal)").pack()
    tk.Scale(pid_frame, from_=0, to=2, resolution=0.01, variable=kp_x, orient="horizontal").pack(fill="x", padx=10)
    ttk.Label(pid_frame, text="KI (Horizontal)").pack()
    tk.Scale(pid_frame, from_=0, to=1, resolution=0.01, variable=ki_x, orient="horizontal").pack(fill="x", padx=10)
    ttk.Label(pid_frame, text="KD (Horizontal)").pack()
    tk.Scale(pid_frame, from_=0, to=1, resolution=0.01, variable=kd_x, orient="horizontal").pack(fill="x", padx=10)

    ttk.Label(pid_frame, text="KP (Distance)").pack()
    tk.Scale(pid_frame, from_=0, to=2, resolution=0.01, variable=kp_y, orient="horizontal").pack(fill="x", padx=10)
    ttk.Label(pid_frame, text="KI (Distance)").pack()
    tk.Scale(pid_frame, from_=0, to=1, resolution=0.01, variable=ki_y, orient="horizontal").pack(fill="x", padx=10)
    ttk.Label(pid_frame, text="KD (Distance)").pack()
    tk.Scale(pid_frame, from_=0, to=1, resolution=0.01, variable=kd_y, orient="horizontal").pack(fill="x", padx=10)

# ============ DRONE THREAD ============

def drone_video_loop():
    global display_frame, display_mask, display_rrt
    global stop_flag, last_path_time
    global h_min, h_max, s_min, s_max, v_min, v_max
    global area_min, area_max, kp_x, ki_x, kd_x, kp_y, ki_y, kd_y

    tello = Tello()
    tello.connect()
    print(f"Battery level: {tello.get_battery()}%")

    # Optional drone takeoff
    tello.takeoff()
    time.sleep(2)
    tello.send_rc_control(0, 0, 50, 0)
    time.sleep(8)
    tello.send_rc_control(0, 0, 0, 0)

    # Start video stream
    tello.streamon()
    time.sleep(2)

    # Center of the frame
    center_x = w // 2
    center_y = h // 2

    # Time trackers
    last_time = time.time()
    last_command_time = time.time()

    # The ideal area for the red hat
    target_area = (fbRange[0] + fbRange[1]) / 2.0

    yolo_model = YOLO('/Users/eladdangur/PycharmProjects/Drone/venv/yolov8n.pt')

    try:
        while not stop_flag:
            # Keep drone alive
            if time.time() - last_command_time > 2:
                tello.send_rc_control(0, 0, 0, 0)
                print('keep drone alive')
                last_command_time = time.time()

            frame = tello.get_frame_read().frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (w, h))

            mask_image = np.zeros((h, w), dtype=np.uint8)

            if all(v is not None for v in [h_min, h_max, s_min, s_max, v_min, v_max]):
                # 1) Detect Red Hat
                lower_red = np.array([h_min.get(), s_min.get(), v_min.get()])
                upper_red = np.array([h_max.get(), s_max.get(), v_max.get()])
                hat_box, mask, hat_area, hat_center = detect_red_hat(
                    frame.copy(),
                    lower_red, upper_red,
                    area_min.get(), area_max.get()
                )
                if mask is not None:
                    mask_image = mask

                # 2) YOLO detection for person + optional target
                person_center = None
                target_pos = None
                person_boxes = []
                results = yolo_model(frame, classes=[PERSON_CLASS_ID, BICYCLE_CLASS_ID])
                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        conf = box.conf[0]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2)//2
                        cy = (y1 + y2)//2

                        if cls_id == PERSON_CLASS_ID and conf > 0.5:
                            # Person bounding box in purple
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,255),2)
                            cv2.putText(frame, "Person", (x1,y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255),2)
                            person_center = (cx, cy)
                            person_boxes.append((x1, y1, x2, y2))

                        elif cls_id == BICYCLE_CLASS_ID and conf > 0.025:
                            # Red bounding box for target (only if found now)
                            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,0,255),2)
                            cv2.putText(frame, "Target Detected",(x1,y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
                            target_pos = (cx, cy)

                # 3) Draw green bounding box for hat if found
                if hat_box:
                    (hx1, hy1, hx2, hy2) = hat_box
                    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
                    if hat_center:
                        cv2.circle(frame, hat_center, 5, (0, 255, 0), -1)
                        cv2.putText(frame, "Red Hat", (hx1, hy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # Add this to show area:
                        cv2.putText(
                            frame,
                            f"Area: {int(hat_area)}",
                            (hx1, hy1 - 30),  # slightly above "Red Hat" text
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                # 4) Define user if person + hat exist and are close
                user_center = None
                if hat_box is not None:
                    hx1, hy1, hx2, hy2 = hat_box

                    for (px1, py1, px2, py2) in person_boxes:
                        # Check containment for each person
                        if (hx1 >= px1 and hx2 <= px2 and
                                hy1 >= py1 and hy2 <= py2):
                            # The hat box is contained in this person's box
                            # => This is likely our user
                            ux = (px1 + px2) // 2
                            uy = (py1 + py2) // 2
                            user_center = (ux, uy)

                            # Optionally draw or break if you only want the first match
                            cv2.circle(frame, user_center, 6, (0, 255, 0), -1)
                            cv2.putText(frame, "User Dot", (ux + 5, uy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            # If you only want one user, you can break here:
                            break

                # 5) Always do PID if user is found
                if user_center:
                    current_time = time.time()
                    dt = current_time - last_time
                    last_time = current_time

                    # Horizontal PID
                    error_x = user_center[0] - (w // 2)
                    pid_x_obj = PIDController(kp_x.get(), ki_x.get(), kd_x.get())
                    roll_float = pid_x_obj.compute(error_x, dt)
                    # clamp roll to ±40
                    roll = int(max(min(roll_float, 40), -40))

                    # Distance => area-based PID
                    error_area = (fbRange[0] + fbRange[1]) / 2.0 - hat_area
                    pid_area_obj = PIDController(kp_y.get(), ki_y.get(), kd_y.get())
                    pitch_float = pid_area_obj.compute(error_area, dt)
                    # clamp pitch to ±20 for slower approach
                    pitch = int(max(min(pitch_float, 20), -20))

                    # dead zone for pitch
                    if abs(error_area) < 200:
                        pitch = 0

                    # Movement text
                    moves = []
                    if pitch > 0:
                        moves.append("Forward")
                    elif pitch < 0:
                        moves.append("Backward")
                    if roll > 0:
                        moves.append("Right")
                    elif roll < 0:
                        moves.append("Left")
                    if not moves:
                        moves = ["Hover"]
                    mv_str = " ".join(moves)

                    cv2.putText(frame, f"Movement: {mv_str}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                    print(f"PID => roll={roll}, pitch={pitch}")
                    # Send RC to Tello
                    tello.send_rc_control(roll, pitch, 0, 0)

                    # 6) If target found => do RRT for display only
                    if target_pos and (time.time() - last_path_time > skip_duration):
                        current_frame = tello.get_frame_read().frame
                        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                        current_frame = cv2.resize(current_frame, (w,h))
                        rrt_img = run_rrt_path_calculation_realtime(current_frame, user_center, target_pos)
                        global display_rrt
                        display_rrt = rrt_img
                        last_path_time = time.time()

            # ============ DRAW ONLY THE BOX UNDER THE CENTER BOX ============
            # Instead of entire 3x3, we highlight the one below center
            draw_lower_center_box(frame)

            # Show frames
            global display_frame
            global display_mask
            display_frame = frame.copy()
            display_mask = mask_image.copy()

            time.sleep(0.02)

    except Exception as e:
        print("Exception in drone_video_loop:", e)
    finally:
        # Land immediately upon close
        tello.land()
        tello.streamoff()
        tello.end()

# ============ OPENCV WINDOWS & UPDATE LOOP ============

def create_opencv_windows():
    cv2.namedWindow("Original Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original Frame", 640, 480)
    cv2.moveWindow("Original Frame", 50, 50)

    cv2.namedWindow("HSV Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Mask", 640, 480)
    cv2.moveWindow("HSV Mask", 700, 50)

    cv2.namedWindow("RRT Path", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RRT Path", 640, 480)
    cv2.moveWindow("RRT Path", 1350, 50)

def update_opencv_display():
    global display_frame, display_mask, display_rrt, stop_flag

    if stop_flag:
        cv2.destroyAllWindows()
        return

    if display_frame is not None:
        cv2.imshow("Original Frame", display_frame)
    if display_mask is not None:
        cv2.imshow("HSV Mask", display_mask)
    if display_rrt is not None:
        cv2.imshow("RRT Path", display_rrt)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        on_closing()
        return

    root.after(20, update_opencv_display)

# ============ EXIT HANDLER ============

def on_closing():
    global stop_flag
    stop_flag = True
    cv2.destroyAllWindows()
    root.destroy()

# ============ MAIN ============

def main():
    global root
    root = tk.Tk()
    create_gui(root)
    create_opencv_windows()

    thread = threading.Thread(target=drone_video_loop, daemon=True)
    thread.start()

    root.after(100, update_opencv_display)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
