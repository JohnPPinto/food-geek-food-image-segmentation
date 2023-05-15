import cv2
import matplotlib.pyplot as plt
from engine import YoloSegPredict

def predict(image_array, 
            model_path, 
            conf_threshold=0.5, 
            iou_threshold=0.5, 
            num_masks=32, 
            mask_alpha=0.5, 
            task='segment'):
    """
    This function predicts and plots the prediction on the image.
    
    Parameters:
        image_array: An array of the image.
        model_path: A string to the path directing towards the model location.
        conf_threshold=0.5: A float in the range (0, 1) for the confidence scores.
        iou_threshold=0.5: A float in the range (0, 1) for the Non maximum supression.
        num_masks=32: An int that contains the value of the predicted mask value used by Yolov8.
        mask_alpha=0.5: A float for the predicted mask opacity on the image.
        task='segment': A string containing either 'segment' or 'detect'.
    
    Returns: A dict containing:
        org_image: An array of the original image resized to the shape of the training data (512, 512).
        result_image: An array of resulted image after drawing prediction on the image. 
        boxes: An array containing the bounding box in pascal VoC format.
        masks: An array containing the predicted objects mask area.
        classes: An list containing the all the classes used for training.
        scores: An array containing the probability or confidence score. 
        class_ids: An array containing the indices of the detected object class/labels.
    """
    # Loading the model
    model = YoloSegPredict(model_path=model_path,
                           conf_threshold=conf_threshold,
                           iou_threshold=iou_threshold,
                           num_masks=num_masks)
    
    # Detecting objects in the image
    org_image, boxes, scores, class_ids, masks = model(image_array)
    
    # Getting the classes list
    classes = model.get_meta_details()
    
    # Drawing and Visualizing the resulted image
    if task == 'segment':
        result_image = model.draw_masks(image=image_array, mask_alpha=mask_alpha)
    elif task == 'detect':
        result_image = model.draw_bbox(image=image_array, mask_alpha=mask_alpha)
        
    return {'org_image': org_image, 
            'result_image': result_image, 
            'boxes': boxes, 
            'masks': masks, 
            'classes': classes, 
            'scores': scores, 
            'class_ids': class_ids} 

if __name__ == '__main__':
    # Getting the model path
    model_path = '../models/yolov8_5class_10percent/best.onnx'
    
    # Reading the image
    image_array = plt.imread('../datasets/images/val/pizza/420409.jpg')
    
    # Drawing and Visualizing the resulted image
    results = predict(image_array, 
                      model_path)
    
    for i in range(len(results['masks'])):
        print(f"[INFO] Detected: {results['classes'][results['class_ids'][i]]} and the confidence score: {results['scores'][i]}.")
    plt.imshow(results['result_image'])
    plt.axis(False)
    plt.show();
