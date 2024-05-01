import cv2
from tracker import Tracker
# Open the video file.
cap = cv2.VideoCapture('12.mp4')

# Create the KNN background subtractor.
bg_subtractor = cv2.createBackgroundSubtractorKNN()

# Set the history length for the background subtractor.
history_length = 20
bg_subtractor.setHistory(history_length)

# Create kernel for erode and dilate operations.
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))

# Create an empty list to store the tracked senators.
senators = []

# Counter to keep track of the number of history frames populated.
num_history_frames_populated = 0

# Start processing each frame of the video.
while True:
    # Read the current frame from the video.
    grabbed, frame = cap.read()

    # If there are no more frames to read, break out of the loop.
    if not grabbed:
        break

    # Apply the KNN background subtractor to get the foreground mask.
    fg_mask = bg_subtractor.apply(frame)

    if num_history_frames_populated < history_length:
        num_history_frames_populated += 1
        continue


    _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)


    cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)


    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    should_initialize_senators = len(senators) == 0
    id = 0
    for c in contours:
        if cv2.contourArea(c) > 500:
          
            (x, y, w, h) = cv2.boundingRect(c)
            
         
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            
            if should_initialize_senators:
                senators.append(Tracker(id, hsv_frame, (x, y, w, h)))
                
        id += 1

    for senator in senators:
        senator.update(frame, hsv_frame)

    cv2.imshow('Senators Tracked', frame)

    k = cv2.waitKey(110)


    if k == 27:
        break

