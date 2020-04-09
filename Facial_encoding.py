from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os


ap =argparse.ArgumentParser()

ap.add_argument("-i", "--dataset",required =True,help="path to input directory of faces+ images")

ap.add_argument("-e", "--encodings", required= True, help="path to serialized db of facial encoding")

ap.add_argument("-d", "--detection-method", type=str, default="cnn", help="face detection model to use: either 'hog' or 'cnn' " )
args= vars(ap.parse_args())

print("[INFO] Quantifying faces...")
imagePaths = list(paths.list_images(args['dataset']))

knownEncodings=[]
knownNames=[]


for (i, imagePath) in enumerate(imagePaths):
	#extracting the person name from the image path
	print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	
	#load the input image and convert it from RGB (OpenCV ordering)
	#to dlib ordering (RGB)
	
	image = cv2.imread(imagePath)
	rgb =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	#detect the x,y coordinates of the bounding boxes
	#corresponding to each face in the input image

	boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
	encodings= face_recognition.face_encodings(rgb, boxes)
	
	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

print("[INFO] serializing encoding...")
data = {"encodings":knownEncodings, "names": knownNames}
f=open(args["encodings"],'wb')
f.write(pickle.dumps(data))
f.close()
