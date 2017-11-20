from styx_msgs.msg import TrafficLight
from PIL import Image
from keras.models import load_model
import h5py
from PIL import Image as Img
import tensorflow as tf
import cv2

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
	self.dl = True
	if (self.dl == True):
	  self.model = load_model('model.h5')
	  self.model._make_predict_function()
	  self.graph = tf.get_default_graph()

        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
	if (self.dl == True):
	  self.model._make_predict_function()
	  cv_image = image
          cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
	  #img = Img.fromarray(cv_image, 'RGB')
	  with self.graph.as_default():
	    pred = self.model.predict(cv_image[None, :, :, :], batch_size=1)
	  print(pred)
	  if (pred[0][1] > 0.5):
  	    return 0
	  #return int(model.predict(image_array[None, :, :, :], batch_size=1))	
	
  	return TrafficLight.UNKNOWN
