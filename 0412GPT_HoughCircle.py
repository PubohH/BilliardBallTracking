import cv2
import numpy as np

# Initialize Kalman filter
dt = 1 / 30.0
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1e-2, 0, 0, 0], [0, 1e-2, 0, 0], [0, 0, 1e-3, 0], [0, 0, 0, 1e-3]], np.float32)
kalman.measurementNoiseCov = np.array([[1e-1, 0], [0, 1e-1]], np.float32)

# Initialize parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Read the video
cap = cv2.VideoCapture('10.mp4')

# Allow user to select initial ROI in the first frame
ret, frame = cap.read()
r = cv2.selectROI(frame)

# Convert ROI to grayscale
roi = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Apply HoughCircles to the initial ROI to locate the pool ball
circles = cv2.HoughCircles(gray_roi, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

if circles is not None:
    # Extract the center and radius of the pool ball
    circles = np.round(circles[0, :]).astype("int")
    (x, y, r) = circles[0]

    # Initialize Kalman filter state
    kalman.statePre = np.array([x, y, 0, 0], dtype=np.float32)
    kalman.statePost = np.array([x, y, 0, 0], dtype=np.float32)

# Loop through subsequent frames and track the pool ball
while True:
    # Read frame from video
    ret, frame = cap.read()

    # If frame is not read, break out of loop
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Predict the next state of the Kalman filter
    prediction = kalman.predict()

    # Define ROI for pool ball
    roi = gray_frame[y - r:y + 2 * r, x - r:x + 2 * r]

    # Apply Lucas-Kanade optical flow to track the pool ball in the ROI
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_roi, roi, prediction[:2], None, **lk_params)

