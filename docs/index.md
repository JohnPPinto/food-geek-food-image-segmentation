---
date: 2023-06-01
---

June 1, 2023
# **How I Created a Food Segmentation Web App - Food Geek**

This project is a demonstration of my ability to work in the field of Deep Learning, showcase my skills to learn quickly and develop and deploy end-to-end Artificial Intelligence technologies. Currently, this project is still in work in progress phase, the project has completed the first stage of deployment and will be moving toward the continuous training (CT) stage, where I will be going back to the data pipeline and iterating improvements to all of the pipelines.

## **The Challenge**

This project creates a web application for different types of food, the web app is capable of understanding the foods by using a neural network architecture. Once the web app identifies the food with the help of the neural network it will display all the nutrients present in that food.

This is a simple implementation of a classification problem, where deep learning is really good at solving it, but there's a catch to this, assume a user eats only one food and uses the web app then the image classification problem will be a straightforward method. But this is not the case for everyone, some might click a photo containing 2 or more foods on a plate or a wide-view photo of the dining table or a buffet with multiple different foods.

When images have a lot of information, only performing image classification is not the best option, to clearly identify all the objects in this case food, object detection, or segmentation works the best. Now, selecting either one will do the work but object detection itself has a flaw here, many times foods can get messy, and drawing a box that has a strict shape can make the visualization bad. So the next best one is instance segmentation where polygon shapes can detect all the objects even though it's a messy plate.

## **Data Pipeline**

The first and foremost steps for building any deep learning project start with the data pipeline, from getting the data to processing it all falls into this space. This stage is the most critical of all in every project, most of the time improving your data can improve the result far better than any other method.

### **Gathering the Training Data**

In deep learning projects, the hardest part is getting a large quantity and a good quality dataset. I imagine this is why most of the papers and research are done on well-known datasets like ImageNet, COCO, etc.

So to start with I will also be using a public dataset, the most popular dataset in the food category is the "[Food - 101 Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)". This dataset is provided by "[ETH Zürich](https://ethz.ch/en.html)", a public research university in Zürich, Switzerland. They have also produced multiple papers and large datasets for deep learning.

This dataset contains 101,000 images for 101 different foods. All of these food images were been reviewed manually and each food class was segregated into a train set containing 750 images and a test set containing 250 images. They have on purpose not cleaned the training images which contain some noise, mostly in the form of color intensity and wrong labels. All the images are also rescaled to have a maximum side length of 512 pixels.

You can find the complete code over [here](https://github.com/JohnPPinto/food-geek-food-image-segmentation/blob/main/notebooks/01_downlaod_dataset.ipynb).

### **Preprocessing the Data**

In this stage, I will be keeping it simple. Neural networks need data for learning patterns and so do their needs for validating their performance. This can be achieved by separating the dataset into two different sets, one called the training set containing a good amount of the images and the next one is a validation set or a testing set, or both of them. 

The dataset is already split into a train and test in a ratio of 75:25, this means that 75 percent of the data is for training and the rest 25 percent is for testing. Making no changes to this, I will be using the same split.

The basic workflow for any machine learning project, in general, is to start small, this means that use a small part of the dataset and experiment with that, once you feel that data needs to be increased then you start adding more data to the neural network. This way you can scale the architecture and improve the performance.

Starting small for this project, I will be taking 5 classes on food: Chicken Curry, Chocolate Cake, Hamburger, Pizza, and ramen. Now, why did I select this food among the 101 different foods, this is because these foods are common and some are even in the fast food category. Along with the few classes, I have also decreased the dataset by 10 percent of the data for each class, which means 75 training images and 25 testing images.

You can find the complete code over [here](https://github.com/JohnPPinto/food-geek-food-image-segmentation/blob/main/notebooks/02_data_preprocessing_eda.ipynb).

### **Creating and Processing Data Annotations**

Now, that we have the images we will also be needing the annotations for all the images. Annotation is a file containing the metadata for the images, this metadata might contain filenames, class, box shape in a specific format, polygon shape in a specific format, and many more details.

Creating annotation is a critical stage where major mistakes are not tolerable, these sorts of data have a major impact on the neural network's result. To create an annotation for my food images, I will be using the tool called [CVAT](https://www.cvat.ai/), it's one of the best open-source tools for annotating any data in computer vision tasks. After, annotating my images I exported them in the format of Segmentation Mask annotation. This format outputs the selected segment of the object for the image in a new image containing 1 channel, a grayscale image that has a white color (1 Values) for the segmented object and the rest of the image is in black color(0 Values pixel).

<figure markdown>
  ![Mask Image](./images/mask_img.png){ width="450" }
  <figcaption>A Mask Image Containing Two Objects</figcaption>
</figure>

Now that I have created the mask for all the images, the trickiest part was to convert the mask image into a format that the neural network will look for and extract those annotations in its standard format. Using YOLO as my neural network, I will need to convert the mask into a text file that will contain the polygon location.

I like to create a function that can convert the mask image into a YOLO text file. This function takes the help of the OpenCV library which has a lot of functions to manipulate images and video files. 

* **Line: 2-4:** First, I read the image file as a grayscale image. Then apply a ```cv2.threshold``` of 1, where the maximum value is 255 for all the pixels in the image, and capture the height and width of the image.

* **Line: 6:** Now that the image is read and we have some metadata, we will get the polygon shape using the ```cv2.findContours``` function, this function uses ```CHAIN_APPROX_SIMPLE`` to get the minimum coordinates but important ones which helps in the compute size and time.

* **Line: 8-16:** Now that we have the coordinates for all the points of the polygon, we need to normalize the values between 0 and 1. Contours provide a tuple within which all the point's x and y coordinates are present in a list, I then divide them with height and width and append them in the list in the format of [x1, y1, x2, y2, ...]

* **Line: 18-31:** Getting the index of the class from the file path directory and creating a file where I add the values in the Yolo format, starting with the class index and then adding all the normalized coordinates in that one line, in Yolo format one object can have only one line.

``` py linenums="1"
def mask_to_yolo(src_filepath: str, dst_filepath: str, class_list: list, verbose: int = 0):
    # Reading the mask annotation file
    mask = cv2.imread(src_filepath, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    H, W = mask.shape

    # Getting the contours of the mask image
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Normalizing the contours
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / W)
                polygon.append(y / H)
            polygons.append(polygon)
    
    # Getting the class of the annotation
    class_idx = class_list.index(src_filepath.split('/')[-2])
    
    # Creating a file in yolo format and writing all the annotations
    with open(dst_filepath, 'w') as file:
        for polygon in polygons:
            for i, p in enumerate(polygon):
                if i == 0:
                    file.write(f'{class_idx} {p} ')
                elif i == len(polygon) - 1:
                    file.write(f'{p}\n')
                else:
                    file.write(f'{p} ')
    file.close()
    if verbose == 1:
        print(f'[INFO] Mask file is been converted into Yolo format: "{dst_filepath}".')
```
Now that we have the function to create an annotation file, it's time to pass it to all the mask files and create a text file for that mask polygon. I get the full path for the mask file and create a path with the same name changing the extension to .txt for a text file.

``` py linenums="1" hl_lines="5 13 14 15 16 17" 
def create_yolo_labels(src_root_dir: str, dst_root_dir: str, class_list: list, verbose: int = 0):
    # Getting all the mask files path
    for classes in tqdm(os.listdir(src_root_dir)):
        class_path = os.path.join(src_root_dir, classes)
        filepath = [os.path.join(class_path, i ) for i in os.listdir(class_path) if '.ipynb_checkpoints' not in i]

        # Creating the destination directory
        dst_dir = os.path.join(dst_root_dir, classes)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        
        # Creating Yolo format annotation file for every mask file
        for file in tqdm(filepath, desc=classes):
            mask_to_yolo(src_filepath=file,
                         dst_filepath=os.path.join(dst_dir, file.split('/')[-1].replace('.png', '.txt')),
                         class_list=class_list,
                         verbose=verbose)
    print('[INFO] All the mask files are converted into Yolo format.')
```

Now, it's time to test the converted annotation and the best way is to visualize and check it on different images and the function plot_yolo_segment does that for us.

* **Lines: 3-24:** Here I read the image file and the text file. Then I extract the yolo annotation from the file and add it to a list called norm_annot. After that using the odd and even method, I denormalize the annotation for the image and append it again to another list called denorm_annot. Finally, create a contour like format data for the annotation. 

* **Lines: 27-52:** Now that I have the contours, I can use ```cv2.boundingRect``` to get the box coordinates(top left x, top left y, width, height) and draw the contours on the image using ```cv2.drawContours``` with some opacity using ```cv2.addWeighted```. After that draw the rectangle box using `cv2.rectangle` and add the class text for identification using ```cv2.putText```. Finally, plot the original and annotated image for comparison. 

``` py linenums="1"
def plot_yolo_segment(img_filepath: str, annot_filepath: str, class_list: list):
    # Reading the image file
    img = cv2.imread(img_filepath)
    H, W, _ = img.shape
    
    # Reading the Yolo annotation
    with open(annot_filepath, 'r') as file:
        norm_annot = file.read().split('\n')
        norm_annot = [i.split(' ') for i in norm_annot]
        norm_annot = [[float(j) for j in i] for i in norm_annot[:-1]] # Last list is empty
    
    # Denormalizing the yolo annotation
    denorm_annot = []
    for annot in norm_annot:
        annot_list = []
        for i, v in enumerate(annot[1:]):
            if i%2 == 0:
                annot_list.append(int(v * W))
            else:
                annot_list.append(int(v * H))
        denorm_annot.append(annot_list)
        
    # Converting the annot list into contours format
    denorm_annot_cont = tuple(np.asarray([i]).reshape(-1, 1, 2) for i in denorm_annot)
    
    # Getting the bounding box coords
    annot_box = []
    for i in denorm_annot_cont:
        annot_box.append(cv2.boundingRect(i))
    
    # Visualizing the annotation on the image
    img_cp = img.copy()
    cv2.drawContours(img, denorm_annot_cont, -1, (255, 0, 0), -1)
    filled = cv2.addWeighted(img, 0.3, img_cp, 1-0.3, 0)
    
    # Drawing the bounding box and text
    class_idx = [int(i[0]) for i in norm_annot]
    for i, box in enumerate(annot_box):
        x, y, w, h = box
        cv2.rectangle(filled, (x, y), (x+w, y+h), (0, 255, 255), 3)
        cv2.putText(filled, class_list[class_idx[i]], (x, y-10 if y-10>10 else y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Plotting both the images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB))
    plt.title(f'Original Image')
    plt.axis(False)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB))
    plt.title(f'Annotated Image')
    plt.axis(False);
```

<figure markdown>
  ![Original and Annotated Image](./images/annot_img.png)
  <figcaption>Comparison between Original Image and Annotated Image</figcaption>
</figure>

You can find the complete code over [here](https://github.com/JohnPPinto/food-geek-food-image-segmentation/blob/main/notebooks/03_processing_data_annotation.ipynb).

## **Model Pipeline**

Now comes the fun part, building a model that is capable of segmenting food images. Since I'm dealing with a classification problem along with instance segmentation and a reason to learn something new in computer vision, I will be using one of the current state-of-the-art models called YOLOv8 which is provided by Ultralytics. Similar to the previous generation of YOLO models, the configuration is similar but additional smartness is added to all the tasks and modes. You can check the whole documentation on their [website](https://docs.ultralytics.com/) for quick reference.

Similar to the previous generation YOLO expects datasets to be in the following structure:

<figure markdown>
  ![YOLO Dataset Structure](./images/yolo_structure.png){width="350"}
  <figcaption>YOLOv8 Dataset Structure</figcaption>
</figure>

### **Model Training**

In my model training stage, my goal was to obtain a model that is powerful and lightweight and initially get a model as a baseline to work with, my goal was not to spend a lot of time in model experiments. So everything was dependent on the baseline model. Once the baseline model is successful, I can come back and improve the model. 

To make this possible, I selected three sizes of the YOLOv8 model for the baseline experiments: Nano, Small, and Medium models. As the name states so does the size of the model. For monitoring the model training performance yolo provides tensorboard and Weights and biases, along with this I used ClearML, which does not need much user input and tracks the model automatically.

``` py
# Setting up clearML task
task = Task.init(project_name='food_image_seg', task_name='exp1_yolov8n_5class_10percent')
```

Loading and training the model method is the same. All of this can be done in a few lines of code.

While loading the model the default pre-trained weight file changes, if you need a nano model you use "yolov8n", for small you use "yolov8s" and for medium "yolov8m".

``` py
# Loading the model - Experiment no.1
model = YOLO('yolov8n-seg.pt')

# Loading the model - Experiment no.2
model = YOLO('yolov8s-seg.pt')

# Loading the model - Experiment no.3
model = YOLO('yolov8m-seg.pt')
```

Model training does not change for any of the experiments, we define the configuration file in ".yaml", 512 for image size because all the images are of length 512, the model will run for 200 epochs a bigger run but to control the run I have given the patience 20 which will work with the early stopping callback.

In the .yaml file, I define the directories of the images and Yolo will automatically cache the labels based on the images directory. Along with this classes are defined and in the order that I have been using until now.

!!! note
    The path that needs to be mentioned in the .yaml file needs to be an absolute, not a relative path. There have been many issues related to this on the GitHub issue. This can be easy for local machines but for the cloud, it can be tough for getting the right path for Yolo to reach.

=== "Model Training Code"
    
    ``` py
    # Training the Yolo model
    results = model.train(data='train_config.yaml',
                          imgsz=512, 
                          epochs=200, 
                          patience=20,
                          project='train_logs_5class_10percent',
                          name='exp1_yolov8n_5class_10percent')

    # Closing the clearML task
    task.close()
    ```
=== "train_config.yaml"

    ``` yaml
    # Path to the dataset
    path: '../../datasets'
    train: 'images/train'
    val: 'images/val'

    # Classes details
    nc: 5
    names: ['chicken_curry', 'chocolate_cake', 'hamburger', 'pizza', 'ramen']
    ```

You can find the complete code over [here](https://github.com/JohnPPinto/food-geek-food-image-segmentation/blob/main/notebooks/04_model_training.ipynb).

### **Model Evaluation and Export**

Now that the training was over it's time to understand which model works best with our goals. Model evaluation is a simple task where we will go through each model and check its result on an unseen dataset, in our case it's the validation or test dataset. Together with this we also need to randomly visualize the result which gives an out of box knowledge.

Yolo provides us with a simple function to validate and predict any images or videos using your desired models, these are smart functions and they know where to look for data, either way, these are stored in the model depending on the model type.

``` py
model = YOLO('train_logs_5class_10percent/exp1_yolov8n_5class_10percent/weights/best.pt') # Load the model
metrics = model.val(name='exp1_best_model') # Run the validation mode
```

In Yolo visualizing the images is simple, once you get the result or the prediction for the images you can use the plot method provided by Yolo to visualize the predicted output.

``` py linenums="1" hl_lines="20 21 22 23 24"
def predict_and_visualize(model_path: str, test_dirpath: str, conf: float):
    # Getting the list of all the images in the directory
    imgs_list = glob.glob(os.path.join(test_dirpath, '*/*'))
    
    # Getting a random sample from the image list
    rand_imgs = random.sample(imgs_list, 10)
    
    # Loading the model
    model = YOLO(model_path)
    
    # Predicting and visualizing the random images
    plt.figure(figsize=(25, 12))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        
        # Reading the random images
        img = cv2.imread(rand_imgs[i])
        
        # Predicting the images and plotting it
        results = model.predict(img, conf=conf) # Using model predict mode
        pred_plot = results[0].plot()
        plot = cv2.cvtColor(pred_plot, cv2.COLOR_BGR2RGB)
        plt.imshow(plot)
        plt.axis(False);
```

Once all the evaluation is done, the model that is been selected for further process needs to be exported. Now, Yolo has mentioned in their documentation that they have multiple ways to export the model and all of this depends on the different criteria like on which system the model will run on eg. web application, mobile application, etc, and the device type in which it will run on whether CPU or GPU and many more.

In our case, I selected the ONNX format, which is capable of interoperability this means that the model is kind of universal in the AI world.

In Yolo exporting the model is as simple as other methods. You just need to call the mode and mention all the arguments and it's done.

``` py
# Exporting the experiment no. 2 best model
exp2_best_model_path = 'train_logs_5class_10percent/exp2_yolov8s_5class_10percent/weights/best.pt'

model = YOLO(exp2_best_model_path)
model.export(format='onnx', imgsz=512)
```

You can find the complete code for model evaluation over [here](https://github.com/JohnPPinto/food-geek-food-image-segmentation/blob/main/notebooks/05_model_evaluation.ipynb) and model export over [here](https://github.com/JohnPPinto/food-geek-food-image-segmentation/blob/main/notebooks/06_model_export.ipynb).

## **Deployment Pipeline**

I'm not gonna lie, this was the scariest part of the whole project. Getting the data and model was something that I have been doing for some time, this pipeline is sort of like an adventure in the unknown. Starting this pipeline I had the model in the ONNX format, a format that I have never used and it's a new technology that has less information on the web, so I started to read the docs and learn new things.

In the deployment pipeline, my goal is straightforward, getting the model to work, the user gives an input (images of food) and the model does the prediction on the images which is returned as an output. Once the model was ready, the next was to build a web app that could let a user interact with it.

### **Adapting to ONNX**

Before getting started with all the explanations, if you are interested in the Yolov8 architecture you can take a look at the below image (The architecture is quite big so you will need to click the image and zoom it).

<figure markdown>
  ![Model Architecture](./images/onnx_model.png){width="350"}
  <figcaption>Yolov8 Model Architecture (Image from <a href="https://netron.app/" target="_blank">netron.app</a>)</figcaption>
</figure>

First, we need to load the model with a few options like whether the inference will be done on CUDA or CPU and then we can read the metadata, inputs, and outputs data present in the model.

``` py linenums="1" hl_lines="12 13 21 28 36" title="engine.py"
class YoloSegPredict:
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
        # Get data from the model
        self.get_meta_details()
        self.get_input_details()
        self.get_output_details()
    
    def get_meta_details(self):
        # Getting the model metadata.
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
```

Now, that we have the model ready for action, we will need an image as an input for the model to predict on. First, we prepare the image, making it compatible with the model input requirements, then we perform the inference on the prepared image tensor and we process the output as per our need, in this case getting the best box and segment mask.

``` py linenums="39" title="engine.py"
    def __call__(self, image):
        return self.segment_objects(image)
    
    def segment_objects(self, image):
        # Prepare the image array as an input tensor.
        input_tensor, self.input_img_resized = self.prepare_input(image)
        
        # Perform inference on the image
        outputs = self.inference(input_tensor)
        
        # Extract prediction data
        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])
        
        return self.input_img_resized, self.boxes, self.scores, self.class_ids, self.mask_maps
```

The image preparing stage is similar to that of the image preparation done in PyTorch, the image tensor is initially reshaped to the model input shape then the tensor is normalized between the range of 0 and 1, then the tensor is permuted into the arrangement of [channels, height, width] and finally the tensor is added with a new axis at the start to indicate it as a batch tensor.

``` py linenums="55" title="engine.py"
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
```

Once I have the model ready and the image ready, then I use the model to predict the image. 

``` py linenums="69" title="engine.py"
    def inference(self, input_tensor):
        # Predicting using the Yolo onnx model
        outputs = self.ort_session.run(self.output_names, {self.input_names[0]: input_tensor})
        
        return outputs
```

Once we get the prediction result, we move forward, toward extracting and processing those predictions. After inferring any images our model gives out two outputs:

1. The output contains all the possible prediction boxes and masks, the format is such a way: The first four are for the boxes coordinate, the next is the probability scores or confidence scores for all the classes in our case it is 5 classes and the remaining are sort of mask coordinates which is 32 by default by Yolo.

2. The second output contains 32 different grayscale mask images that are predicted by the model, the shape of the output is [1, 32, 128, 128] here the mask images are in 128 X 128 shape.

!!! tip 

    The 32 different images come from the convolution layer having 32 filters. You can check this in the above image of Yolo Architecture.

Now, that we know everything about the outputs, it's time to extract the one that has the best prediction. Starting with the box prediction. I extract the classes, boxes, and mask values based on the highest confidence score. Once the boxes are extracted they are processed into the non-maximum suppression algorithm, which gives only the best predicted box coordinates rest of the boxes are suppressed.

=== "engine.py"

    ``` py linenums="75" hl_lines="7 8 9 23 26 27 28 35 36 37 38 39 40 41 42 43 44 45 46"
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

            # Separating the prediction from the first output
            box_predictions = predictions[..., :num_classes+4]
            mask_predictions = predictions[..., num_classes+4:]

            # Getting class with the highest confidence score
            class_ids = np.argmax(box_predictions[:, 4:], axis=1)

            # Getting the bounding box for all the objects
            boxes = self.extract_boxes(box_predictions)

            # Apply Non Maximum Suppression to suppress overlapping box
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
    ```

=== "rescale_boxes"

    ``` py linenums="1"
    # Rescale any bounding box
    def rescale_boxes(boxes, input_shape, output_shape):
        """
        This functions helps in re-scaling bounding box from one object to another.

        Parameters:
            boxes: An array containing the values of the bounding box.
            input_shape: A tuple or list containing values of the original object shape. E.g. (height, width)
            output_shape: A tuple or list containing values of the output object shape. E.g. (height, width)

        Returns:
            boxes: An array containing the values of the rescale boxes.
        """
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([output_shape[1], output_shape[0], output_shape[1], output_shape[0]])
        return boxes
    ```

=== "bbox_yolo_to_pascal"

    ``` py linenums="1"
    # Convert bounding box from YOLO format (x_c, y_c, w, h) into Pascal VOC format (x1, y1, x2, y2)
    def bbox_yolo_to_pascal(boxes):
        """
        This function helps in converting the bounding box format from YOLO to Pascal VOC.

        Parameters:
            boxes: An array containing the values of the bounding box in YOLO format.

        Returns:
            boxes_cp: An array containing the values of the bounding box in Pascal VOC format.
        """
        boxes_cp = boxes.copy()
        boxes_cp[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
        boxes_cp[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
        boxes_cp[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
        boxes_cp[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
        return boxes_cp
    ```

=== "clip_bbox"

    ``` py linenums="1"
    # Clipping the bounding box values
    def clip_bbox(boxes, height, width):
        """
        This function helps in clipping the values of the bounding box.

        Parameters:
            boxes: An array containing the values of the bounding box. 
            height: An int value of the height of a Image or Frame.
            width: An int value of the width of a Image or Frame.

        Return:
            clip_boxes: An array containing the clipped values of the bounding box.
        """
        clip_boxes = boxes.copy()
        clip_boxes[..., 0] = np.clip(boxes[..., 0], 0, width)
        clip_boxes[..., 1] = np.clip(boxes[..., 1], 0, height)
        clip_boxes[..., 2] = np.clip(boxes[..., 2], 0, width)
        clip_boxes[..., 3] = np.clip(boxes[..., 3], 0, height)
        return clip_boxes
    ```

=== "compute_nms"

    ``` py linenums="1" hl_lines="29"
    # Computing Non Maximum Suppression on all the bounding box
    def compute_nms(boxes, scores, iou_threshold):
        """
        This function helps in computing the Non Maximum Suppression on the 
        predicted bounding boxes.
        
        Parameters:
            boxes: An array containing the values of the bounding boxes.
            scores: An array containing the values of the confidence scores
                    for each bounding box.
            iou_threshold: A float value to suppress the bounding box.
                           Value should be within the range (0, 1).
        
        Returns: 
            Keep_boxes: A list containing the index for the boxes and scores 
                        array after computing Non Maximum Suppression.
        """
        # Getting the list of indices of sorted scores - descending order
        sorted_indices = np.argsort(scores)[::-1]
        
        # Looping through the indices and computing nms
        keep_boxes = []
        while sorted_indices.size > 0:
            # Picking the box with best score
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)
            
            # Compute the IoU of the picked box with the rest of the boxes
            ious = compute_iou(box=boxes[box_id, :], boxes=boxes[sorted_indices[1:], :])
            
            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]
            
            # Keeping only the indices that fit within the threshold
            sorted_indices = sorted_indices[keep_indices + 1]
            
        return keep_boxes
    ```

=== "compute_iou"

    ``` py linenums="1"
    # Computing the Intersection over the Union of the bounding box.
    def compute_iou(box, boxes):
        """
        This function helps in calculating the intersection over union of the bounding boxes.
        This function best works with prediction result, where one predicted box is computed with 
        multiple different predicted boxes.

        Parameters:
            box: An array containing values of a bounding box.
            boxes: An array containing values of multiple different bounding box.

        Returns:
            iou: An array containing iou values in between range (0, 1) for all the boxes array.
        """
        # Getting the intersection box
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area
        return iou
    ```

Now, with this, we have the bounding box, let's move to the next stage which is to extract the mask. The process to extract needs to be in a sequential method similar to the way I extracted the bounding box.

1. First I take the mask prediction that we got along with the boxes while extracting the boxes. This mask prediction is determined by the highest confidence score and performs a matrix multiplication with the second output of the inference result. Next, the result is processed using the sigmoid method and reshaped to the mask images i.e. [1, 128, 128]

2. A copy of the bounding box is rescaled to the shape of the mask image. 

3. Next, an image tensor of the same shape as the input image is been created containing only black pixels. 
    
    * First, the scaled mask copy is cropped to the bounding box size, this gives us a clear shape of the masking tensor. 
    * After that, the cropped mask is resized to the shape of the original bounding box, and all the values in the mask are given a threshold of 0.5, which makes it either a 0 or 1 value mask. 

4. This mask is then replaced with the mask area of the initially created black pixel image tensor.

=== "engine.py"

    ``` py linenums="124" hl_lines="9 10 11 14 15 16 19 35 36 37 38 39 40 41"
    def process_mask_output(self, mask_predictions, mask_output):
        # If no mask prediction
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
            
            # Cropping the scaled mask and resizing it to the image dimension
            scale_crop_mask = masks[i][scale_y1: scale_y2, scale_x1: scale_x2]
            crop_mask = cv2.resize(scale_crop_mask, 
                                   (x2 - x1, y2 - y1), 
                                   interpolation=cv2.INTER_CUBIC)
            crop_mask = cv2.blur(crop_mask, blur_size)
            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask
        
        return mask_maps
    ```

=== "sigmoid"

    ``` py
    # Compute sigmoid
    def sigmoid(x):
        """
        This function computes the mathematical sigmoid function.
        Parameters: x: An int or array.
        Returns: An int or array containing values after computing.
        """
        return 1 / (1 + np.exp(-x))
    ```

Now that, we have all the predictions, both the bounding box and the mask it's time to visualize the prediction. There are a total of two predictions so we can either visualize one or both. To simplify these two different functions are used one to draw the mask other to draw the box. If the mask is missing then the box will fill the box area with a color.

* Drawing a mask is simple unlike before when I used ```cv2.drawContours```, this time we can crop the mask tensor and the image tensor using the bounding box area and replacing the mask area with the color. Then using the ```cv2.addWeighted``` I can change the opacity of the color and place both images one above the other.

* After drawing the mask, I use the mask image and draw the bounding box using the ```cv2.rectangle``` and write a text of the detect class using the ```cv2.putText```.

=== "engine.py"

    ``` py linenums="168"
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
    ```

=== "draw_mask"

    ``` py linenums="1" hl_lines="30 33 34 35 36 39"
    # Drawing the mask prediction on the image or frame
    def draw_masks(image, boxes, class_ids, class_list, mask_alpha=0.5, mask_maps=None):
        """
        This function draws the predicted mask on the base image.

        Parameters:
            image: An array containing the values of the base image in RGB format.
            boxes: An array containing the values of the predicted bounding box in Pascal Voc format. 
            class_ids: An array containing the values of the predicted classes indices. 
            class_list: A list containing all the class names in proper order. 
            mask_alpha: Default = 0.5, A float in range (0, 1) for opacity of the mask area. 
            mask_maps: Default = None, An array containing the values of the mask area.

        Returns:
            (masked_image, colors): A tuple containing the masked image array and colors list used for the classes.
        """
        mask_image = image.copy()

        # Generating colors for every class
        rng = np.random.default_rng(3)
        colors = rng.uniform(0, 255, size=(len(class_list), 3))

        # Drawing predicted objects
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            color = colors[class_id]
            x1, y1, x2, y2 = box.astype(int)

            # Fill bounding box on condition
            if mask_maps is None:
                cv2.rectangle(mask_image, (x1, y1), (x2, y2), color, -1)
            else:
                # Fill mask on condition
                crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis] # Cropping the mask area
                crop_mask_image = mask_image[y1:y2, x1:x2] # Cropping the mask area from the image
                crop_mask_image = crop_mask_image * (1 - crop_mask) + crop_mask * color # Adding color to the mask area
                mask_image[y1:y2, x1:x2] = crop_mask_image # Replacing the mask area in the image

        # Returning mask image with color opacity
        return cv2.addWeighted(mask_image, mask_alpha, image, 1 - mask_alpha, 0), colors
    ```

=== "draw_detection"

    ``` py linenums="1" hl_lines="23 36 37 38 39 43 44 45 46 47 50 51 52 53 54 55 56 57"
    # Drawing the bounding box and adding label text on the predicted image mask
    def draw_detections(image, boxes, scores, class_ids, class_list, mask_alpha=0.5, mask_maps=None):
        """
        This function helps in drawing the predicted detection bounding box and mask.

        Parameters:
            image: An array containing the values of the base image in RGB format.
            boxes: An array containing the values of the predicted bounding box in Pascal Voc format.
            scores: An array containing the values of the confidence score for each predicted bounding box.
            class_ids: An array containing the values of the predicted classes indices. 
            class_list: A list containing all the class names in proper order. 
            mask_alpha: Default = 0.5, A float in range (0, 1) for opacity of the mask area. 
            mask_maps: Default = None, An array containing the values of the mask area.

        Returns:
            mask_image: An array containing the values for image with objects predicted.
        """
        image_height, image_width = image.shape[:2]
        size = min([image_height, image_width]) * 0.001 # Dynamic fontscale
        text_thickness = int(min([image_height, image_width]) * 0.001) # Dynamic thickness

        # Getting the Image with mask prediction using the function
        mask_image, colors = draw_masks(image, boxes, class_ids, class_list, mask_alpha, mask_maps)

        # Draw predicted bounding box and labels on the mask image
        for box, score, class_id in zip(boxes, scores, class_ids):
            color = colors[class_id]
            x1, y1, x2, y2 = box.astype(int)

            # Drawing rectangle
            cv2.rectangle(mask_image, (x1, y1), (x2, y2), color, 2)

            # Getting the box coords of the label text
            label = class_list[class_id]
            caption = f'{label} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(text=caption, 
                                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                          fontScale=size, 
                                          thickness=text_thickness)
            th = int(th * 1.2)

            # Drawing rectangle for the text
            cv2.rectangle(mask_image, 
                          (x1, y1), 
                          (x1 + tw, y1 - th if y1 - 10 > 0 else y1 + 10 + th), 
                          color, 
                          -1)

            # Adding the label text
            cv2.putText(mask_image, 
                        caption, 
                        (x1, y1 if y1 - 10 > 0 else y1 + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        size, 
                        (255, 255, 255), 
                        text_thickness, 
                        cv2.LINE_AA)
        return mask_image
    ```

This way we have both all the prediction data and the processed images with the result applied to them. Finally, we can conclude with the ONNX format and use it with the web app that we will be going through after this.

You can find the complete code for engine.py over [here](https://github.com/JohnPPinto/food-geek-food-image-segmentation/blob/main/module/engine.py) and for utils.py over [here](https://github.com/JohnPPinto/food-geek-food-image-segmentation/blob/main/module/utils.py)

### **Building API and Web App**

We have everything ready at this time, now we need an API and web app that does the same work. For the API I used Fast API and for the website I used Gradio. Both of them do the work very quickly and easily, this helps in deploying the model much faster with fewer errors.

* Starting with the Fast API, I first initiate the FastAPI app, then I create a Post method of name predict-to-json, here the API takes the image as input using the ```UploadFile``` function. The uploaded image is read in bytes format which is then transformed into a Numpy array, this array is passed to the model, and all the prediction array is transformed into a JSON format and returned as a dict type. 

* Similar to the above steps a new post method is been created called predict-to-image, here the input image is taken and processed after that the predicted image is been converted into bytes format, and the buffer is read as a PNG format file to display the image.

* Once the FastAPI app is been created, I then create the gradio app, for the gradio app there are a total of three functions:

1. gradio_predict: This function takes the input image array and gets the prediction result using the model, then the predicted mask is returned along with the classes predicted.

2. on_annot_select: This function triggers whenever the predicted class or food name is clicked, the class name is passed to the fds_food_info function in the food_info.py file, this function uses the API provided by the Food Data Center of the United States Department of Agriculture, which provided the nutritional information of the selected image and the output is shown in the textbox.

3. on_clear_btn: This function is to trigger whenever the clear button is clicked, all the content on the web app will be emptied and a new fresh app is ready to run a new image prediction.

The rest of the code uses the ```gradio.blocks``` function to place different GUI blocks of predefined functions like the input image block ```gradio.Image```, ```gradio.Button```, ```gradio.AnnotedImage```, ```gradio.Examples``` and ```gradio.Textbox```. All of the information is present in the Gradio documentation.

Now that the web app and FastAPI are ready we can mount the web app on the FastAPI, gradio offers a direct function to do this using ```gradio.mount_gradio_app```. This function mounts the gradio web app on the FastAPI app using the server provided by the FastAPI, in this case, Uvicorn is the server for the FastAPI and Gradio after mounting.

=== "app.py: FastAPI"

    ``` py linenums="1"
    model_path = 'models/yolov8s-seg-v1.onnx'

    app = FastAPI(title='Food Geek API',
                  description='''Upload any images of food and obtain predicted values out of the image, 
                                 return json and image result.''')

    # Prediction result - JSON format
    @app.post('/predict-to-json')
    async def api_predict_json(file: UploadFile = File(...)):
        """
        This API will take any food image file and return a JSON file of prediction results.
        The prediction result will contain numpy.ndarray which is dumped into JSON format.
        To convert it back into the numpy.ndarray, use ```numpy.asarray(json.loads(...)) # Replace ... with the variable.  
        """
        # Validating only image files
        extension = file.filename.split('.')[-1] in ('jpg', 'jpeg', 'png')
        if not extension:
            return f'File "{file.filename}" must be of Image format "JPG", "JPEG" or "PNG"'

        # Reading the image file
        content = await file.read()
        image = np.asarray(Image.open(BytesIO(content)))

        # Getting predictions results
        results = predict(image_array=image,
                          model_path=model_path,
                          conf_threshold=0.7)

        # Converting the results in json format
        results['org_image'] = json.dumps(results['org_image'].tolist())
        results['result_image'] = json.dumps(results['result_image'].tolist())
        results['boxes'] = json.dumps(results['boxes'].tolist())
        results['masks'] = json.dumps(results['masks'].tolist())
        results['scores'] = json.dumps(results['scores'].tolist())
        results['class_ids'] = json.dumps(results['class_ids'].tolist())

        return results

    # Prediction result - Image visualization
    @app.post('/predict-to-image')
    async def api_predict_image(file: UploadFile = File(...)):
        """
        This API takes any image file and applies prediction on the image.
        Once the process is done, Resulting image with prediction will be
        displayed as a png file.
        """
        # Validating only image files
        extension = file.filename.split('.')[-1] in ('jpg', 'jpeg', 'png')
        if not extension:
            return f'File "{file.filename}" must be of Image format "JPG", "JPEG" or "PNG"'

        # Reading the image file
        content = await file.read()
        image = np.asarray(Image.open(BytesIO(content)))

        # Getting predictions results
        results = predict(image_array=image,
                          model_path=model_path,
                          conf_threshold=0.7)

        # Converting the predicted image into PIL image
        img_base64 = Image.fromarray(results['result_image'])

        # buffering a PNG file and returning it.
        with BytesIO() as buf:
            img_base64.save(buf, format='PNG')
            img_bytes = buf.getvalue()
        return Response(img_bytes, media_type='image/png')
    ```
=== "app.py: Gradio"

    ``` py linenums="1"
    # Creating a predict function for the website
    def gradio_predict(img):  
    # Getting the prediction result for the image
    results = predict(image_array=img,
                      model_path=model_path,
                      conf_threshold=0.7)
    
    # formating the classes
    class_list = []
    for names in results['classes']:
        class_list.append(names.replace('_', ' '))
        
    # Validating the result
    if len(results['masks']) == 0:
        return (img, [([0, 0, 0, 0], 'No Food Detected')])
    else:
        # Isolating the result for every mask
        pred = []
        for i in range(len(results['masks'])):
            pred.append((results['masks'][i] / 2, class_list[results['class_ids'][i]]))
        return (img, pred)

    # Creating a function when segment is selected
    def on_annot_select(evt: gr.SelectData):
        info = fds_food_info(evt.value)
        return info

    # Creating a function to clear all data
    def on_clear_btn():
        return None, None, None

    # Creating the UI
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Header
        gr.Markdown('<center><h1>Food Geek</h1></center>')

        # Body
        with gr.Row():
            # Image uploading
            with gr.Column(min_width=768):
                with gr.Box():
                    with gr.Column():
                        input = gr.Image(type='numpy', 
                                         label='Image')
                        with gr.Row():
                            btn_clear = gr.Button(value='Clear')
                            btn_submit = gr.Button(value="Submit", 
                                                   variant='primary')
                        gr.Examples(examples=glob.glob('examples/*.jpg'),
                                    inputs=input)
            # Displaying resulted image
            output = gr.AnnotatedImage(label='Result').style(height=512, width=512, color_map={'': ''})

        # Additional info textbox 
        food_info_box = gr.Textbox(label='Food Info')

        # Footer
        gr.Markdown('Made by John - [Github Link](https://github.com/JohnPPinto/food-geek-food-image-segmentation)')

        # On selected event
        btn_submit.click(fn=gradio_predict, inputs=input, outputs=output)
        btn_clear.click(fn=on_clear_btn, inputs=None, outputs=[input, output, food_info_box])
        output.select(fn=on_annot_select, inputs=None, outputs=food_info_box)

    # Mounting the gradio app onto the fastAPI app
    app = gr.mount_gradio_app(app=app, blocks=demo, path='/')

    if __name__ == '__main__':
        uvicorn.run('app:app')
    ```

=== "food_info.py"

    ``` py 
    # loading env
    load_dotenv()
    API_KEY = os.getenv('API_KEY')

    # Matching classes with FDC ID
    food_list_id = {'chicken curry': '2341861', 
                    'chocolate cake': '2343346', 
                    'hamburger': '2342374', 
                    'pizza': '2344102', 
                    'ramen': '2341959'}

    # Function to provide info on the food
    def fds_food_info(food_name: str):
        """
        This function helps in collecting food info from the FDC database.
        Parameters:
            food_name: A string containing one of the classes.
        Returns: 
            food_info: A string containing all the nutritional info.
        """
        url = 'https://api.nal.usda.gov/fdc/v1/food'
        fdc_id = food_list_id[food_name]

        response = requests.get(url + '/' + fdc_id, 
                                headers={'Accept': 'application/json'}, 
                                auth=HTTPBasicAuth(API_KEY, ''))

        obj = json.loads(response.text) if response.status_code == 200 else None

        if obj != None:
            info_list = []
            for i in obj['foodNutrients']:
                info_list.append(f"{i['nutrient']['name']}: {i['amount']}")

            food_info = f'Food Name: {food_name}\nNutritional Info:\n\n' + ',\n'.join([str(elem) for i, elem in enumerate(info_list)])
            return food_info
        else:
            return response, obj, "There's no information for this food."
    ```
To properly deploy the application, I used Docker and HuggingFace spaces to host the web application, using Dockerfile, HuggingFace can easily host any kind of application.

Below are the images of the Fast API app and Gradio Web app

<figure markdown>
  ![FastAPI app](./images/fastapi_app.png)
  <figcaption>Food Geek - FastAPI (<a href="https://johnpinto-food-geek.hf.space/docs" target="_blank">Link</a>)</figcaption>
</figure>

<figure markdown>
  ![Gradio web app](./images/gradio_web_app.png)
  <figcaption>Food Geek - Web Application (<a href="https://johnpinto-food-geek.hf.space" target="_blank">Link</a>)</figcaption>
</figure>

## **Future Prospect**

There are many things that need to be done now that the first stage of deployment is been completed. Some of them are:

1. **Data Pipeline:** Currently, I'm using the public dataset Food-101, which was introduced back in the year 2014, a lot of time has passed and so has the food data, data needs to be fresh and this can be achieved by scraping the data from online websites. 

2. **Model Pipeline:** I have not yet used the full potential of the Yolo Model, using all the features and squeezing the model to get the right and balanced mAP for all the classes that need to be achieved.

3. **Deployment Pipeline:** Right now the model is running on HuggingFace which makes it difficult to interact with once it is been hosted. This can be improved by looking into cloud deployment.

4. **Monitoring Pipeline:** This is not something different it mostly goes hand-in-hand with the deployment pipeline, right now I don't know how the model is performing after the deployment, or even how the resources are been allocated to the web application, how long does it take for the model to give out predictions and how long does the whole process takes for the output to be seen by the user (lack of latency data)

5. **Automation Pipeline:** This is a level 2 MLOps feature, here most of the features will be automated, something that I'm interested in learning and implementing. This will help in also getting our model to train by the inputs it received, like a self-learning continuous training feature.

## **Final Words**

I'm quite satisfied with how it has turned out, it's not the perfect version of what I'm imagining but it is close and it's just the start of the project. The code is working, and the structure and workflow are cleanly implemented, this as a base now has quite potential for improvement. Now the project will need less coding for some time and more emphasis on the data and model improvement. One day, I hope that the model can work on multiple different classes or foods and thus fully close the feedback loop of continuous improvement.

*Do test the [Food Geek](https://johnpinto-food-geek.hf.space) application and let me know your feedback. You can reach out to me by any means possible GitHub, Twitter, Email, etc.* 

*If you find any issues in the code or have something to discuss you can raise an issue on the [GitHub website](https://github.com/JohnPPinto/food-geek-food-image-segmentation/issues).*
