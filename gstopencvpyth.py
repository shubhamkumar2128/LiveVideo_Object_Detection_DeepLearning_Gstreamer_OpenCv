
#python rtod.py --txtf MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
#python rtod.py -p MobileNetSSD_deploy.prototxt.txt -m MobileNetSSD_deploy.caffemodel

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from yolovideodisp import Video

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--txtf", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("loading model...")
net = cv2.dnn.readNetFromCaffe(args["txtf"], args["model"])

print("starting video stream...")
vs = VideoStream(src="rtsp://admin:Ankitshukla28@10.110.42.215:7001/1?pos=2020-01-09T12:58:31").start()
time.sleep(2.0)
fps = FPS().start()
i=1
video = Video()
while True:
	if not video.frame_available():
            continue

        frame = video.frame()

	#frame = vs.read()
	frame = imutils.resize(frame, width=400)
	print("----------Frame %d---------- "%i)
	i+=1

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (400, 400)),		#convert into blob
		0.007843, (300, 300), 127.5)

	net.setInput(blob)
	detections = net.forward()

	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
			print(startX,startY,endX,endY)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	
	if key == ord("q"):
		break

	
	fps.update()

fps.stop()
print("elapsed time: {:.2f}".format(fps.elapsed()))
print("approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
