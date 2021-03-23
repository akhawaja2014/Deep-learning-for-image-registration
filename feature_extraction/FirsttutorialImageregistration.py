import numpy as np
import cv2
import matplotlib.pyplot as plt

img1_color = cv2.imread('/home/tgiencov/Registration Codes/Python image registration/im1.JPG')  # Image to be aligned. 
img2_color = cv2.imread('/home/tgiencov/Registration Codes/Python image registration/im2.JPG')    # Reference image. 
print(img1_color.shape)
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY) 
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY) 
height, width = img2.shape 

#print(height)
#print(width)

orb_detector = cv2.ORB_create(5000) 

#plt.imshow(img1)
#plt.show()
#print(img1)

kp1, d1 = orb_detector.detectAndCompute(img1, None) 
kp2, d2 = orb_detector.detectAndCompute(img2, None) 
print("Keyponts are")
print(kp1.shape)
# print('detectors are')
# print(len(d1))
imgre = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
plt.imshow(imgre), plt.show()

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 

# print(matcher)
matches = matcher.match(d1, d2) 
print(matches[0])
matches.sort(key = lambda x: x.distance) 

matches = matches[:int(len(matches)*90)] 
no_of_matches = len(matches) 
# print(len(matches))
p1 = np.zeros((no_of_matches, 2)) 
p2 = np.zeros((no_of_matches, 2)) 
print(p2.shape)


for i in range(len(matches)): 
  p1[i, :] = kp1[matches[i].queryIdx].pt 

  p2[i, :] = kp2[matches[i].trainIdx].pt 

print(p1[0,:])
print(kp1[0].pt)
# homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 


# transformed_img = cv2.warpPerspective(img1_color, homography, (width, height)) 

# cv2.imwrite('output.jpg', transformed_img)