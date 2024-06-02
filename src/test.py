import cv2

# Replace the URL with the IP camera's stream URL
url = 'http://192.168.1.33/'
text = "live Cam Testing"
cv2.namedWindow(text, cv2.WINDOW_AUTOSIZE)

# Create a VideoCapture object
cap = cv2.VideoCapture(url)

# Check if the IP camera stream is opened successfully
if not cap.isOpened(): 
    print("Failed to open the IP camera stream")
    exit()

# Read and display video frames
while True:
    # Read a frame from the video stream
    has_frame, frame = cap.read()
    if has_frame:
        cv2.imshow(text, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()