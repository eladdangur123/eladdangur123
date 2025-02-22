import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, Canvas, Frame, Scale
import threading
from ultralytics import YOLO
from djitellopy import Tello
import time

# PID Controller Class
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return 0.15*(self.kp * error + self.ki * self.integral + self.kd * derivative)


# Frame size
w, h = 640, 480

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
    v_min = tk.IntVar(value=75)
    v_max = tk.IntVar(value=255)
    area_min = tk.IntVar(value=500)
    area_max = tk.IntVar(value=5000)

    # Initialize shared variables for PID
    kp_x = tk.DoubleVar(value=0.4)
    ki_x = tk.DoubleVar(value=0.02)
    kd_x = tk.DoubleVar(value=0.05)
    kp_y = tk.DoubleVar(value=0.4)
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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return (x, y, x + w, y + h), mask, area
    return None, mask, 0

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

# Initialize Tello drone
tello = Tello()
tello.connect()
print(f"Battery level: {tello.get_battery()}%")

# Takeoff and rise to 2.5 meters
tello.takeoff()
time.sleep(1)
tello.send_rc_control(0, 0, 50, 0)  # Ascend
time.sleep(10)  # Wait for the drone to reach ~2.5m altitude
tello.send_rc_control(0, 0, 0, 0)  # Stop vertical movement

# Start video stream
tello.streamon()
time.sleep(2)  # Wait for the stream to initialize

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

            hat_box, mask, hat_area = detect_red_hat(frame, lower_red, upper_red, min_area, max_area)
            results = yolo_model(frame)
            detected_person = False

            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == PERSON_CLASS_ID and box.conf[0] > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        if hat_box and is_overlap(hat_box, (x1, y1, x2, y2)):
                            print("Person with red hat detected!")
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            cv2.putText(frame, "Person with Red Hat Detected", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                            detected_person = True

                            # Check if red dot is in the central square
                            in_central_square = is_in_central_square(cx, cy, central_square_bounds)

                            if not in_central_square:
                                # Calculate errors for PID
                                current_time = time.time()
                                dt = current_time - last_time
                                last_time = current_time

                                error_x = cx - center_x
                                error_y = center_y - cy

                                # Create PID Controllers dynamically
                                pid_x = PIDController(kp_x.get(), ki_x.get(), kd_x.get())
                                pid_y = PIDController(kp_y.get(), ki_y.get(), kd_y.get())

                                # Compute PID corrections
                                roll = int(pid_x.compute(error_x, dt))  # Left/Right
                                pitch = -1*int(pid_y.compute(error_y, dt))  # Forward/Backward

                                # Send control commands to Tello
                                tello.send_rc_control(roll, pitch, 0, 0)
                                print(f"PID Control: Roll={roll}, Pitch={pitch}")
                            else:
                                # Hover in place
                                tello.send_rc_control(0, 0, 0, 0)
                                print("Red dot is in the central square. Hovering.")

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