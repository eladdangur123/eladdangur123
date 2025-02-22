import sys
import cv2
import numpy as np
import heapq
import time
import os

print("file path_calculation file is at: " + sys.executable)
base_dir = r"C:\Users\97254\PycharmProjects\Final_Project\venv"
paths_dir = os.path.join(base_dir, "paths")

# Create the frames directory if it doesn't exist
if not os.path.exists(paths_dir):
    os.makedirs(paths_dir)
    print(f"Directory created at: {paths_dir}")
else:
    print(f"Directory already exists at: {paths_dir}")

# Parameters
w, h = 640, 480  # Frame dimensions

# A* Algorithm
def heuristic(a, b):
    """Heuristic function for A* (Euclidean distance)."""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def astar(grid, start, goal):
    """A* pathfinding on a 2D grid."""
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        neighbors = [
            (current[0] + dx, current[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]
        neighbors = [
            (x, y)
            for x, y in neighbors
            if 0 <= x < rows and 0 <= y < cols and grid[x, y] == 0
        ]

        for neighbor in neighbors:
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

def draw_path(img, path, color=(0, 0, 255), thickness=3):
    """Draw the calculated path on the image."""
    for i in range(len(path) - 1):
        start_point = tuple(map(int, path[i]))
        end_point = tuple(map(int, path[i + 1]))
        cv2.line(img, start_point, end_point, color, thickness)
    return img

# Main Function
if __name__ == "__main__":
    # Read arguments from the command line
    if len(sys.argv) != 6:
        print("Usage: path_calculation.py <frame_path> <person_x> <person_y> <chair_x> <chair_y>")
        sys.exit(1)
    else:
        print("Called Sub-Process Successfully!")

    frame_path = sys.argv[1]
    person_x, person_y = int(sys.argv[2]), int(sys.argv[3])
    chair_x, chair_y = int(sys.argv[4]), int(sys.argv[5])

    # Load the frame
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Error: Unable to load frame from {frame_path}")
        sys.exit(1)

    # Create a binary grid (0 = free, 1 = obstacle)
    grid = np.zeros((h, w), dtype=int)

    # Example obstacles (replace with actual obstacles if needed)
    grid[200:300, 300:400] = 1  # Example obstacle block

    # Start and goal positions
    person_pos = (person_y, person_x)  # (row, col)
    chair_pos = (chair_y, chair_x)    # (row, col)

    print(f"Calculating path from person ({person_x}, {person_y}) to chair ({chair_x}, {chair_y})")

    # Calculate path using A*
    path = astar(grid, person_pos, chair_pos)
    if path:
        print(f"Path found: {path}")
        path_points = [(p[1], p[0]) for p in path]  # Convert to (x, y) for OpenCV

        # Draw the path on the frame
        img_with_path = draw_path(frame, path_points, color=(0, 0, 255), thickness=5)

        # Save the image with the path
        path_filename = os.path.join(paths_dir, f"frame_{int(time.time())}.jpg")
        cv2.imwrite(path_filename, img_with_path)  # Save the frame
        print(f"Frame saved at: {path_filename}")
    else:
        print("No path found.")
