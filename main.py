"""The master file for the whole project. To run the system, simply run this file."""

import os

import cv2
import numpy as np
import tensorflow as tf

from stereo_disparity_master.stereo_disparity import stereo_disparity

# parameters to be changed
master_path_to_dataset =
show_r_img = False
show_preprocessing = True # show the left image after preprocessing
show_disparity = True # show calculated disparity
wait_between_images = False # add a 2 second delay before moving onto the next image

# datasets
directory_to_cycle_left = "left-images/" # colour
directory_to_cycle_right = "right-images/" # greyscale
full_path_directory_left =  os.path.join(master_path_to_dataset,
                                         directory_to_cycle_left)
full_path_directory_right =  os.path.join(master_path_to_dataset,
                                          directory_to_cycle_right)
left_file_list = sorted(os.listdir(full_path_directory_left))

# CLACHE setup for image pre-processing
clache = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

# camera setup for distance ranging
camera_focal_length_px =
stereo_camera_baseline_m =
depth_constant = camera_focal_length_px * stereo_camera_baseline_m # to avoid unnecessary recalculations

# tensorflow setup for object detection
PATH_TO_CKPT = "inference_graph/frozen_inference_graph.pb"
label_mapper = {1:'person', 2:'car'} # to avoid loading all labels through standard means (was causing problems when I tried to run on CIS machines)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0') # max 200

# colours for drawing boxes and text (BGR)
object_colour = {"car":(255,0,0), "person":(0,0,255)}

# main loop
for filename_left in left_file_list:
    # get image path
    filename_right = filename_left.replace("_L", "_R")
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

    # check target images exist
    if not os.path.isfile(full_path_filename_right):
        print("Image {} not found, skipping".format(filename_right))
        continue

    # get target images and show if requested
    imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
    imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
    if show_l_img:
        cv2.imshow("left image",imgL)
        cv2.waitKey(5)
    if show_r_img:
        cv2.imshow("right image",imgR)
        cv2.waitKey(5)

    # image pre-processing (CLACHE on HSV)
    adjusted_img = cv2.cvtColor(imgL, cv2.COLOR_BGR2HSV)
    adjusted_img[:,:,2] = clache.apply(adjusted_img[:,:,2])
    adjusted_img = cv2.cvtColor(adjusted_img, cv2.COLOR_HSV2BGR)
    if show_preprocessing:
        cv2.imshow("Pre-processed Images", adjusted_img)

    # identify image objects
    objects = [] # will be in form [(x0, y0, x1, y1, label), ...]
    imgL_expanded = np.expand_dims(adjusted_img, axis=0)
    # get object regions, outputs in form:
    # boxes:   bounding boxes of objects (proportion of image)
    # scores:  probability that region contains specified object. Regions are
    #          returned sorted from most to least probable
    # classes: class of detected object, returned as 1 or 2 and mapped ot string
    #          accoring to label_mapper variable
    # num:     the number of objects detected (this is at most 200)
    boxes, scores, classes, num = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: imgL_expanded})
    image_height = len(imgL) # used to convert image bounding box coords to usable ints
    image_width = len(imgL[0])
    # if probability that region contains an object is >0.9, add coordinates to
    # list of identified objects
    # note scores is in form [[p1, p2, ..., pn]]
    for object in range(int(num[0])):
        if scores[0,object] < 0.9:
            break
        else:
            objects.append((int(image_width*boxes[0,object,1]), int(image_height*boxes[0,object,0]),
                            int(image_width*boxes[0,object,3]), int(image_height*boxes[0,object,2]),
                            label_mapper[classes[0,object]]))
    # setup for next stage
    output_img = imgL.copy()

    # check if there aren't any detections
    # if there aren't any detections, move on to next image in folder as no point
    # in depth calculaitons
    if len(objects) == 0:
        cv2.imshow("Detections", output_img)
        cv2.waitKey(10)
        print(full_path_filename_left)
        print("{} : nearest detect scene object 0m".format(full_path_filename_right))
        print()
        continue

    # calculate disparity
    disparity = stereo_disparity(full_path_filename_left, full_path_filename_right)
    cv2.imwrite("report/images/original_disparity_adj.png", disparity)
    for i in range(image_height): # remove zeros from disparity by replacing by RH value
        for j in range(130)[::-1]:
            if disparity[i][j] == 0:
                disparity[i][j] = disparity[i][j+1]
    np.place(disparity, disparity==0, 1) # catch-all just in case
    if show_disparity:
        cv2.imshow("Disparity", disparity)
        cv2.waitKey(10)

    # setup for getting min depth
    min_depth = np.inf
    # cycle through objects, calculate depth and draw boxes onto image
    for object in objects:
        # extract object data
        x, y, w, h, label = object
        # get depth as median over specified region
        object_depth = depth_constant / np.median(disparity[y:y+h, x:x+w])
        # compare to current min
        if object_depth < min_depth:
            min_depth = object_depth
        # put bounding rectangle and text on image
        cv2.rectangle(output_img, object[:2], object[2:4], object_colour[object[4]], 1)
        cv2.putText(output_img,
                    "{}: {}m".format(object[4], round(object_depth, 2)),
                    (object[0], object[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    object_colour[object[4]])

    # show output image with detections and write some required details to terminal
    cv2.imshow("Detections", output_img)
    cv2.waitKey(5)
    print(full_path_filename_left)
    print("{} : nearest detect scene object {}m".format(full_path_filename_right, round(min_depth, 2)))
    print()

    # pause if requested
    if wait_between_images:
        time.sleep(2)

input("Finished cycling through photots, press enter to exit")
cv2.destroyAllWindows()
