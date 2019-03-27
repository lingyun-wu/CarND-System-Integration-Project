import os
import cv2
import numpy as np
import tensorflow as tf
import rospy
import time

from styx_msgs.msg import TrafficLight

LABELS = {
    1: TrafficLight.RED,
    2: TrafficLight.YELLOW,
    3: TrafficLight.GREEN,
    4: TrafficLight.UNKNOWN
}
LABELS_NAME = {
    1: "RED", 
    2: "YELLOW", 
    3: "GREEN", 
    4: "UNKNOWN"
}

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        # Path to frozen detection graph.
        PATH_TO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen_inference_graph.pb")
        
        # Load a frozen model into memory        
        self.detection_graph = tf.Graph()
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name = '')

            self.sess = tf.Session(graph=self.detection_graph, config=config)
        
        # Input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        # Preprocess image
        image_ = cv2.resize(image, (300, 300))
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)

        # traffic light detection
        boxes, scores, classes = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                  feed_dict={self.image_tensor: np.expand_dims(image_, axis=0)})

        boxes = np.squeeze(boxes)        
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        # identify type of signal (RED/GREEN/YELLOW)
        for i, box in enumerate(boxes):
            class_idx = classes[i]

            if scores[i] > 0.50:
                rospy.loginfo("Identified Traffic Light: %s" % (LABELS_NAME[class_idx]))
                return LABELS[class_idx]

        return TrafficLight.UNKNOWN
