import cv2
import os
import json


def writeToJSONFile(path, fileName, data):
    filePathNameWExt = './' + path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)



cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')

imagePaths="test_images/"
data = {}
for file in os.listdir(imagePaths):	

	# Input image
	#print(file)
	img = cv2.imread(os.path.join(imagePaths,file))
	rows, cols, channels = img.shape

	# Use the given image as input, which needs to be blob(s).
	tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))

	# Runs a forward pass to compute the net output
	networkOutput = tensorflowNet.forward()
	a=0
	# Loop on the outputs
	for detection in networkOutput[0,0]:
		score = float(detection[2])
		if score > 0.1:
			a+=1

		# Example
		
		data[file] = a

		writeToJSONFile('./','image2products',data)
		# './' represents the current directory so the directory save-file.py is in
		# 'test' is my file name



