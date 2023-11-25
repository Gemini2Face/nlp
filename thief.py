import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Read the first frame and use it as the reference background
ret, background = cap.read()

if not ret:
    print("Error: Can't receive frame. Exiting ...")
    cap.release()
    out.release()
    exit()

# Convert the background frame to grayscale
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# Loop to read and process each frame
while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert current frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference
    difference = cv2.absdiff(background, gray_frame)

    # Apply threshold
    _, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a flag for significant movement
    significant_movement = False

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Threshold for contour area
            # Get the bounding box coordinates around the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw a rectangle around the significant movement
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            significant_movement = True

    # Display "UNSAFE" text if significant movement is detected
    if significant_movement:
        cv2.putText(frame, "UNSAFE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the frame to the video file
    out.write(frame)

    # Display the frame with rectangles
    cv2.imshow('Motion Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

