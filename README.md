# Food Geek: A Food Segmentation Application

A Food Segmentation project, built using Food 101 Dataset, YOLOv8, ONNX, FastAPI, Docker and Gradio.

You can find the live and running web application over [here](https://johnpinto-food-geek.hf.space) and the project documentation over [here](https://johnppinto.github.io/food-geek-food-image-segmentation).

## Local Installation

1. **Docker Method:**

* If you prefer to use a Docker Image, you can pull the [image](https://hub.docker.com/repository/docker/johnppinto/food-geek/general) from the docker hub by using the below command:

```
docker pull johnppinto/food-geek:0.2.0
```
* Once you have the docker image in your system you can use this command to run the web application on the local host port 7860 (http://localhost:7860/).

```
docker run -it --rm -p 7860:7860 johnppinto/food-geek:0.2.0
```

2. **Local Build Method:**

* Clone this repository and set the current working directory to demo directory.

```
git clone https://github.com/JohnPPinto/food-geek-food-image-segmentation.git
cd food-geek-food-image-segmentation/demo
```

* Execute the requirements file in your environment.

```
pip install -r requirements.txt
```

* Now you can run the app.py file.

```
python app.py
```

The web application will run on the following address: http://127.0.0.1:8000, by default uvicorn server uses this address to run any application.
