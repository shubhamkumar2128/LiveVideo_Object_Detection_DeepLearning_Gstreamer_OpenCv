# LiveVideo_Object_Detection_DeepLearning_Gstreamer_OpenCv
Object Detection using deeplearning with python and get video feed form live source using Gstreamer multimedia framework.

Running steps:

# Terminal one:

$~/Desktop/FCL$ gst-launch-1.0 uridecodebin uri= rtsp://admin:Ankitshukla28@10.110.42.215:7001/1?pos=2020-01-09T12:58:31 ! videobalance saturation=0 ! x264enc ! video/x-h264, stream-format=byte-stream ! rtph264pay ! udpsink host=127.0.0.1 port=5600


# Terminal two:

python gstopencvpyth.py --txtf MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

