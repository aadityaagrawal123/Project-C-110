import cv2

import numpy as np

import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

camera = cv2.VideoCapture(0)

while True:

	status , frame = camera.read()

	if status:

		frame = cv2.flip(frame , 1)
	
		cv2.putText(frame, "Rock, Paper & Scissors", (115,40), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 3)
		#resize the frame
		img = cv2.resize(frame,(224,224))

		# expand the dimensions
		test_image = np.array(img, dtype=np.float32)
		test_image = np.expand_dims(test_image, axis=0)
	
		# normalize it before feeding to the model
		normalised_image = test_image/255.0
	
		# get predictions from the model
		prediction = model.predict(normalised_image)

		if (prediction[0,0] > prediction[0,1] and prediction[0,0] > prediction[0,2]):
			cv2.putText(frame, "Prediction: Rock", (160,85), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 3)
		elif (prediction[0,1] > prediction[0,0] and prediction[0,1] > prediction[0,2]):
			cv2.putText(frame, "Prediction: Paper", (160,85), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 3)
		elif (prediction[0,2] > prediction[0,0] and prediction[0,2] > prediction[0,1]):
			cv2.putText(frame, "Prediction: Scissor", (160,85), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 3)
		else:
			cv2.putText(frame, "Detecting...", (150,85), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 3)
			 
		cv2.imshow('Result Window' , frame)

		code = cv2.waitKey(1)
		
		if code == 32:
			break

camera.release()

cv2.destroyAllWindows()
