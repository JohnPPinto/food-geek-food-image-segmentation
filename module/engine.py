import math
import cv2
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
from utils import rescale_boxes, bbox_yolo_to_pascal, clip_bbox, compute_nms, sigmoid, draw_detections

class YoloSegPredict:
    """
    This class helps in loading the model, predicting an image and providing coords and mask 
    array for the segmentation.
    Parameters:
        model_path: A string to the path directing towards the model location.
        conf_threshold: A float in the range (0, 1) for the confidence scores.
        iou_threshold: A float in the range (0, 1) for the Non maximum supression.
        num_masks: An int that contains the value of the predicted mask value used by Yolov8.
    """
    def __init__(self, model_path, conf_threshold = 0.7, iou_threshold = 0.5, num_masks=32):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_masks = num_masks
        
        # Initializing the model
        self.initialize_model(model_path)
    
    def initialize_model(self, model_path):
        EP_LIST = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = onnxruntime.InferenceSession(model_path, 
                                                        providers = EP_LIST)
        # Get meta data from the model
        self.get_meta_details()
        self.get_input_details()
        self.get_output_details()
    
    def get_meta_details(self):
        # Getting the model meta data.
        model_meta = self.ort_session.get_modelmeta()
        self.class_dict = eval(model_meta.custom_metadata_map['names'])
        self.class_list = list(self.class_dict.values())
        return self.class_list
    
    def get_input_details(self):
        # Getting the input data
        model_inputs = self.ort_session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
    def get_output_details(self):
        # Getting the output data
        model_outputs = self.ort_session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        
    def __call__(self, image):
        return self.segment_objects(image)
    
    def segment_objects(self, image):
        # Prepare the image array as a input tensor.
        input_tensor, self.input_img_resized = self.prepare_input(image)
        
        # Perform inference on the image
        outputs = self.inference(input_tensor)
        
        # Extract prediction data
        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])
        
        return self.input_img_resized, self.boxes, self.scores, self.class_ids, self.mask_maps
    
    def prepare_input(self, image):
        # Getting image info
        self.image_height, self.image_width = image.shape[:2]
        
        # Resize input image to input size
        input_img_resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Preprocessing the input image
        input_img = input_img_resized / 255.0 # Normalizing
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        
        return input_tensor, input_img_resized
    
    def inference(self, input_tensor):
        # Predicting using the Yolo onnx model
        outputs = self.ort_session.run(self.output_names, {self.input_names[0]: input_tensor})
        
        return outputs
    
    def process_box_output(self, box_output):
        # Extracting predictions from box outputs
        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4 # box data - mask data - box coords
        
        # Filter out confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        
        # Validating for no scores
        if len(scores) == 0:
            return [], [], [], np.array([])
        
        # Seprating the prediction from the first output
        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]
        
        # Getting class with the highest confidense score
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)
        
        # Getting the bounding box for all the objects
        boxes = self.extract_boxes(box_predictions)
        
        # Apply Non Maximum Suooression to suppress overlapping box
        indices = compute_nms(boxes=boxes, 
                              scores=scores, 
                              iou_threshold=self.iou_threshold)
        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]
    
    def extract_boxes(self, box_predictions):
        # Extract box from predictions
        boxes = box_predictions[:, :4]
        
        # Scale boxes to original image dimension
        boxes = rescale_boxes(boxes=boxes, 
                              input_shape=(self.input_height, self.input_width), 
                              output_shape=(self.image_height, self.image_width))
        
        # Convert the boxes to pascal voc format
        boxes = bbox_yolo_to_pascal(boxes=boxes)
        
        # Clipping the boxes range to a image limit
        boxes = clip_bbox(boxes=boxes, 
                          height=self.image_height, 
                          width=self.image_width)
        
        return boxes
    
    def process_mask_output(self, mask_predictions, mask_output):
        # if no mask prediction
        if mask_predictions.shape[0] == 0:
            return []
        
        mask_output = np.squeeze(mask_output)
        
        # Calculate the mask area for all the box
        num_mask, mask_height, mask_width = mask_output.shape
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))
        
        # Rescale the boxes to match the mask size
        scale_boxes = rescale_boxes(boxes=self.boxes,
                                    input_shape=(self.image_height, self.image_width),
                                    output_shape=(mask_height, mask_width))
        
        # Mask map for each box and mask pair
        mask_maps = np.zeros((len(scale_boxes), self.image_height, self.image_width))
        blur_size = (int(self.image_width/mask_width), int(self.image_height/mask_height))
        for i in range(len(scale_boxes)):
            # Rounding the scaled boxes
            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))
            
            # Rounding the base boxes
            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))
            
            # Cropping the scaled mask and resizing it to image dimension
            scale_crop_mask = masks[i][scale_y1: scale_y2, scale_x1: scale_x2]
            crop_mask = cv2.resize(scale_crop_mask, 
                                   (x2 - x1, y2 - y1), 
                                   interpolation=cv2.INTER_CUBIC)
            crop_mask = cv2.blur(crop_mask, blur_size)
            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask
        
        return mask_maps
    
    def draw_bbox(self, image, mask_alpha=0.5):
        # Drawing only the bounding box and filling it.
        return draw_detections(image=image,
                               boxes=self.boxes,
                               scores=self.scores,
                               class_ids=self.class_ids,
                               class_list=self.class_list,
                               mask_alpha=mask_alpha)
    
    def draw_masks(self, image, mask_alpha=0.5):
        # Drawing both the bounding box and the mask
        return draw_detections(image=image,
                               boxes=self.boxes,
                               scores=self.scores,
                               class_ids=self.class_ids,
                               class_list=self.class_list,
                               mask_alpha=mask_alpha,
                               mask_maps=self.mask_maps)
