import typing
import glob
import numpy as np
from PIL import Image
from io import BytesIO
import json
import gradio as gr
import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import Response
from predict import predict
from food_info import fds_food_info

model_path = 'models/yolov8s-seg-v1.onnx'

# ----- FastAPI ----- #

app = FastAPI(title='Food Geek API',
              description='''Upload any images of food and obtain predicted values out of the image, 
                             return json and image result.''')

# Prediction result - JSON format
@app.post('/predict-to-json')
async def api_predict_json(file: UploadFile = File(...)):
    """
    This API will take any food image file and return a json file of prediction result.
    The prediction result will contains numpy.ndarray which are dumped into json format.
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

# ----- Gradio ----- #

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
    gr.Markdown('Made by John - [Github Link]()')
    
    # On selected event
    btn_submit.click(fn=gradio_predict, inputs=input, outputs=output)
    btn_clear.click(fn=on_clear_btn, inputs=None, outputs=[input, output, food_info_box])
    output.select(fn=on_annot_select, inputs=None, outputs=food_info_box)

# Mounding the gradio app on to the fastAPI app
app = gr.mount_gradio_app(app=app, blocks=demo, path='/')

if __name__ == '__main__':
    uvicorn.run('app:app')