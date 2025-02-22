import cv2
import numpy as np
from ultralytics import YOLO
import random
import math

# Initialize YOLO model
yolo_model = YOLO('yolov8n.pt')  # Replace with your YOLOv8 model path
PERSON_CLASS_ID = 0  # YOLO class ID for person
CHAIR_CLASS_ID = 56  # YOLO class ID for chair (update if different)

# RRT Parameters
MAX_ITERATIONS = 500
STEP_SIZE = 20
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_goal_reached(node, goal, threshold=10):
    """Check if the node is close to the goal."""
    return distance(node, goal) < threshold

def generate_random_node():
    """Generate a random node within the frame."""
    return random.randint(0, FRAME_WIDTH - 1), random.randint(0, FRAME_HEIGHT - 1)

def steer(from_node, to_node, step_size):
    """Move from a node toward another node with a fixed step size."""
    theta = math.atan2(to_node[1] - from_node[1], to_node[0] - from_node[0])
    return (int(from_node[0] + step_size * math.cos(theta)),
            int(from_node[1] + step_size * math.sin(theta)))

def find_nearest_node(tree, random_node):
    """Find the nearest node in the tree to the random node."""
    return min(tree, key=lambda node: distance(node, random_node))

def rrt(start, goal):
    """Perform RRT algorithm to find a path."""
    tree = [start]
    parents = {start: None}

    for _ in range(MAX_ITERATIONS):
        random_node = generate_random_node()
        nearest_node = find_nearest_node(tree, random_node)
        new_node = steer(nearest_node, random_node, STEP_SIZE)

        tree.append(new_node)
        parents[new_node] = nearest_node

        if is_goal_reached(new_node, goal):
            print("Goal reached!")
            path = []
            current = new_node
            while current is not None:
                path.append(current)
                current = parents[current]
            path.reverse()
            return path

    print("Failed to find a path.")
    return None

# YOLO Detection
def detect_objects(frame):
    """Detect person and chair in the frame."""
    results = yolo_model(frame)
    person_coords, chair_coords = None, None

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = box.conf[0]

            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                if class_id == PERSON_CLASS_ID:
                    person_coords = (center_x, center_y)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif class_id == CHAIR_CLASS_ID:
                    chair_coords = (center_x, center_y)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Chair", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return person_coords, chair_coords, frame

# Main loop
def main():
    cap = cv2.VideoCapture(0)  # Webcam input

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        person_coords, chair_coords, annotated_frame = detect_objects(frame)

        if person_coords and chair_coords:
            print(f"Person: {person_coords}, Chair: {chair_coords}")
            path = rrt(person_coords, chair_coords)

            if path:
                for i in range(len(path) - 1):
                    cv2.line(annotated_frame, path[i], path[i + 1], (0, 255, 255), 2)

                cv2.circle(annotated_frame, person_coords, 5, (0, 0, 255), -1)
                cv2.circle(annotated_frame, chair_coords, 5, (255, 0, 0), -1)

        cv2.imshow("RRT Path Planning", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

