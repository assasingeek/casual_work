import matplotlib.pyplot as plt
import cv2
import glob

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):

	img_copy = colored_img.copy()
	gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
	faces = f_cascade.detectMultiScale(gray, scaleFactor, 5)
	for (x, y, w, h) in faces:
		cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 4)              

	return img_copy

def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
	# test = cv2.imread('data/myphoto.jpg')
	test = [cv2.imread(file) for file in glob.glob("data/*.jpg")]

	haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
	# haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	
	i=0
	while(i<len(test)):
		faces_detected_img = detect_faces(haar_face_cascade, test[i])
		cv2.imwrite('output/image%03i.jpg' %i, faces_detected_img)
		plt.imshow(convertToRGB(faces_detected_img))
		plt.show()
		i=i+1

main()