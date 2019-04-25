from imageai.Detection import VideoObjectDetection
import os
import cv2 as cv

execution_path = os.getcwd()

camera = cv.VideoCapture("http://195.189.181.205/mjpg/video.mjpg") 

#camera = cv.VideoCapture("http://207.192.232.2:8000/mjpg/video.mjpg")
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
detector.loadModel(detection_speed="faster")

custom_objects = detector.CustomObjects(car=True, truck=True)

video_path = detector.detectCustomObjectsFromVideo(camera_input=camera,
                                output_file_path=os.path.join(execution_path, "rtmp://a.rtmp.youtube.com/live2")
                                , frames_per_second=30, log_progress=True)
print(video_path)
