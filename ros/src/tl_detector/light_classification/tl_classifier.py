import os
import cv2
import numpy as np
import tensorflow as tf
import rospy
import time

from styx_msgs.msg import TrafficLight

LABELS = [
    TrafficLight.UNKNOWN,
    TrafficLight.RED,
    TrafficLight.YELLOW,
    TrafficLight.GREEN,
    TrafficLight.UNKNOWN
]
LABELS_NAME = [
    "UNKNOWN", 
    "RED", 
    "YELLOW", 
    "GREEN", 
    "UNKNOWN"
]

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        # Path to frozen detection graph.
        PATH_TO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_inference_graph.pb")
        
        # Load a frozen model into memory        
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name = '')

            self.sess = tf.Session(graph=self.detection_graph)
        
        # Input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        # preprocess image
        image_ = cv2.resize(image, (300, 300))
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        image_ = image_.astype(np.float32)

        # traffic light detection
        start_time = time.time()
        num_detections, classes, scores, boxes = self.sess.run([self.num_detections, self.detection_classes, self.detection_scores, self.detection_boxes],
                                                  feed_dict={self.image_tensor: np.expand_dims(image_, axis=0)})
        self.time_taken_for_inference = time.time() - start_time
        rospy.logdebug("Time taken for inference: %s" % (self.time_taken_for_inference))
        classes = np.squeeze(classes)
        scores = np.squeeze(scores)
        boxes = np.squeeze(boxes)
        num_detections = np.squeeze(num_detections).astype(np.uint32)

        # identify type of signal (RED/GREEN/YELLOW)
        for i in range(num_detections):
            class_idx = classes[i]

            if scores[i] > 0.50:
                rospy.loginfo("Identified Traffic Light: %s" % (LABELS_NAME[int(class_idx)]))
                return LABELS[int(class_idx)]

        return TrafficLight.UNKNOWN
