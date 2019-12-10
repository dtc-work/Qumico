import time
from os import path
from io import BytesIO
import sys

from PIL import Image

dir_path = path.dirname(__file__)
if dir_path not in sys.path:
    sys.path.append(dir_path)

import picamera
import picamera.array

def capture_image(export_path=None,image_name="sample.jpg"):
    if export_path is None:
        export_path = path.join(path.dirname(path.abspath(__file__)),"images")
    with picamera.PiCamera() as camera:
        camera.resolution = (1024, 768)
        camera.start_preview()
        # Camera warm-up time
        time.sleep(2)
        export_path = path.join(export_path, image_name) 
        camera.capture(export_path)


def capture_stream(resolution=(640, 480)):
    with picamera.PiCamera(resolution=resolution) as camera:
        with picamera.array.PiRGBArray(camera) as output:
            while True:
                output.truncate(0)
                camera.capture(output, "rgb",use_video_port=True)
                return output.array



        
if __name__ == "__main__":    
    capture_image() 
    