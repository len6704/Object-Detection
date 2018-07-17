import numpy as np
import cv2
import os

import object_detection as obj

#test_videos information
PATH_TO_TEST_VIDEO_DIR = "cnn_class2_videos"
TEST_VIDEO_FILE = "test"
TEST_VIDEO_PATH = os.path.join(PATH_TO_TEST_VIDEO_DIR,"{}.mp4".format(TEST_VIDEO_FILE))

#video detection
read = cv2.VideoCapture(TEST_VIDEO_PATH)
if read.isOpened():
    print ("read_success")
    fps = int(read.get(cv2.CAP_PROP_FPS))
    width = int(read.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(read.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width,height)
    write = cv2.VideoWriter("{}.avi".format(TEST_VIDEO_FILE),cv2.VideoWriter_fourcc('M','J','P','G'),5,size,True)

if write.isOpened():
    print ("write_success") 

while(1):
    check,frame_np = read.read()
    if (check==False):
        break
    else:
        frame_np_expanded = np.expand_dims(frame_np, axis=0)
        output_dict = obj.run_inference_for_single_image(frame_np, obj.detection_graph)
        obj.vis_util.visualize_boxes_and_labels_on_image_array(
          frame_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          obj.category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=2)
        write.write(frame_np) 
