# Import Libraries
import cv2
import numpy as  np


# Parameters for Shi-tomasi corner detection
st_params= dict(maxCorners=30,qualityLevel=0.2,minDistance=2, blockSize=7)

# Parameters for Lucas-Kande optical flow
lk_params=dict(winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1))

# Video Capture
cap = cv2.VideoCapture(r"C:\Users\Eswar\Desktop\Simpli_Learn\Kaggle\run.mp4")
#cap = cv2.VideoCapture(0)
# Color for optical flow
color = (0,255,0)

# Read the capture and get the first frame
ret, first_frame = cap.read()
if ret == True:
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# Convert frame to Grayscale
#prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Find the strongest corners in the first frame
prev = cv2.goodFeaturesToTrack(prev_gray,mask=None, **st_params)

# Create an image with the same dimensions as the frame for later drawing purposes
mask = np.zeros_like(first_frame)

# While loop
while(cap.isOpened()):

    # Read the capture and get the first frame
    ret, frame = cap.read()
    if ret == True:
            # Convert all frame to Grayscale (previously we did only the first frame)
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   
    # Calculate optical flow by Lucas-Kanade
    next , status, error = cv2.calcOpticalFlowPyrLK(prev_gray,gray,prev, None, **lk_params)
   
    # Select good feature for the previous position
    good_old= prev[status==1]
   
    # Select good feature for the next position
    good_new = next[status==1]
   
    # Draw optical flow track
    for i , (new, old) in enumerate(zip(good_new,good_old)):    
        # Return coordinates for the new point
        a,b = new.ravel()
       
        # Return coordinates for the old point
        c,d = old.ravel()
       
        # Draw line between new and old position
        mask = cv2.line(mask,(a,b),(c,d),color,2)
       
        # Draw filled circle
        frame = cv2.circle(frame,(a,b),3,color,-1)        
    # Overlay optical flow on original frame
    output = cv2.add(frame, mask)
   
    # Update previous frame
    prev_gray = gray.copy()
   
    # Update previous good features
    prev= good_new.reshape(-1,1,2)
   
    # Open new window and display the output
    cv2.imshow("Optical Flow", output)
   
    # Close the frame
   
    if cv2.waitKey(300) & 0xFF == ord("q"):
         break
# Release and Destroy
cap.release()
cv2.destroyAllWindows()