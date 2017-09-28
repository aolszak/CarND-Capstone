from styx_msgs.msg import TrafficLight

import numpy as np
import os
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
import time

class TLClassifier(object):
    def __init__(self):
        

        self.current_light = TrafficLight.UNKNOWN  # Default value if pass on network / or nothing detected.

        cwd = os.path.dirname(os.path.realpath(__file__))
     
        self.simulation = True  # Set to false for real  (sim one somehow works sorta for real too it's spooky)
        self.faster = True  # 10 for faster run time or anything else for larger network.

        CKPT = cwd+"/../../../../../Assets/ssd1000_bosch.pb"

        PATH_TO_LABELS = cwd+'/../../../../../Assets/label_map.pbtxt'
        NUM_CLASSES = 14

        self.category_index = {
            1: "Green", 
            2: "Red",
            3: "GreenLeft",
            4: "GreenRight",
            5: "RedLeft",
            6: "RedRight",
            7: "Yellow",
            8: "Off",
            9: "RedStraight",
            10: "GreenStraight",
            11: "GreenStraightLeft",
            12: "GreenStraightRight",
            13: "RedStraightLeft",
            14: "RedStraightRight"
        }

        self.image_np_deep = None
        self.detection_graph = tf.Graph()

        # https://github.com/tensorflow/tensorflow/issues/6698
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # end

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        print("Loaded graph")


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        run_network = True  # flag to disable running network if desired

        if run_network is True:
            image_np_expanded = np.expand_dims(image, axis=0)

            #time0 = time.time()

            # Actual detection.
            with self.detection_graph.as_default():
                (boxes, scores, classes, num) = self.sess.run(
                    [self.detection_boxes, self.detection_scores, 
                    self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: image_np_expanded})
            
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            min_score_thresh = .50
            for i in range(boxes.shape[0]):
                if scores is None or scores[i] > min_score_thresh:
                    
                    class_index = classes[i]
                    class_name = self.category_index[class_index]
                    # class_id = self.category_index[classes[i]]['id']  # if needed

                    print('TrafficLight class index {}'.format(class_index))

                    # Traffic light thing
                    self.current_light = TrafficLight.UNKNOWN

                    if class_name == 'Red':
                        self.current_light = TrafficLight.RED
                    elif class_name == 'Green':
                        self.current_light = TrafficLight.GREEN
                    elif class_name == 'Yellow':
                        self.current_light = TrafficLight.YELLOW

                    fx =  1345.200806
                    fy =  1353.838257
                    perceived_width_x = (boxes[i][3] - boxes[i][1]) * 800
                    perceived_width_y = (boxes[i][2] - boxes[i][0]) * 600

                    # ymin, xmin, ymax, xmax = box
                    # depth_prime = (width_real * focal) / perceived_width
                    # traffic light is 4 feet long and 1 foot wide?
                    perceived_depth_x = ((1 * fx) / perceived_width_x)
                    perceived_depth_y = ((3 * fy) / perceived_width_y )

                    estimated_distance = round((perceived_depth_x + perceived_depth_y) / 2)
        
        self.image_np_deep = image

        return self.current_light

    def get_light_name(self, index):
        name = "No lights"
        if index in self.category_index:
            name = self.category_index[index]
        return name
