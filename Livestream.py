from imageai.Detection import VideoObjectDetection
import os
import cv2 as cv

execution_path = os.getcwd()

camera = cv.VideoCapture("http://195.189.181.205/mjpg/video.mjpg") 

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

custom_objects = detector.CustomObjects(car=True, truck=True)

video_path = detector.detectCustomObjectsFromVideo(camera_input=camera,
                                output_file_path=os.path.join(execution_path, "camera_detected_1")
                                , frames_per_second=20, log_progress=True)
print(video_path)
