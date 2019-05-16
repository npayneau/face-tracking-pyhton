import cv2
import os
import requests

def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image
 
    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

def __downloadCascade():
    print("Downloading haarcascade for face detection")
    folder = "./cascade/"
    for url in ['https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml', 'https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_eye.xml']:
	    local_filename = folder + url.split('/')[-1]
	    # Check if already exists on users disk
	    if not os.path.exists(folder):
		    os.makedirs(folder)
	    # Stream download dataset to lcoal disk
	    r = requests.get(url, stream=True)
	    with open(local_filename, 'wb') as f:
		    for chunk in r.iter_content(chunk_size=1024):
		        if chunk:
		            f.write(chunk)

__downloadCascade()

# Initialize the cascade classifier for detecting faces
face_cascade = cv2.CascadeClassifier("./cascade/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./cascade/haarcascade_eye.xml")

# Initialize the camera (use bigger indices if you use multiple cameras)
cap = cv2.VideoCapture(0)
# Set the video resolution to half of the possible max resolution for better performance
cap.set(3, 1920 / 2)
cap.set(4, 1080 / 2)

imgGlasses = cv2.imread('glasses.png',-1)

# Standard text that is displayed above recognized face
exceptional_frames = 100
while True:
	print(exceptional_frames)
	# Read frame from camera stream and convert it to greyscale
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detect faces using cascade face detection
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	# Loop through detected faces and set new face rectangle positions
	for (x, y, w, h) in faces:
		color = (0, 255, 0)
		startpoint = (x, y)
		endpoint = (x + w, y + h)
		exceptional_frames = 0
		# Draw face rectangle on image frame
		cv2.rectangle(img, startpoint, endpoint, color, 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]		
		if h > 0 and w > 0:
			glass_symin = int(y + 1.5 * h / 5)
			glass_symax = int(y + 2.5 * h / 5)
			sh_glass = glass_symax - glass_symin
			face_glass_roi_color = img[glass_symin:glass_symax, x:x+w]			
			specs = cv2.resize(imgGlasses, (w, sh_glass),interpolation=cv2.INTER_CUBIC)
			transparentOverlay(face_glass_roi_color,specs)
		
		#eyes = eye_cascade.detectMultiScale(roi_gray)
		#for (ex,ey,ew,eh) in eyes:
		#	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	# Show image in cv2 window
	cv2.imshow("image", img)
	# Break if input key equals "ESC"
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
	exceptional_frames += 1