# TensorFlow Serving Tutorial
1. Download and install docker for MacOS: https://docs.docker.com/docker-for-mac/install/
2. (optional) Install nividia-docker to run your application on GPU
3. Pull TensorFlow Serving Image: ```docker pull tensorflow/serving```
4. Prepare TensorFlow Serving Serving using docker:
   ```
   docker run -it -p 8501:8501 -v "$(pwd)/serving/hair_seg/:/models/hair_seg" -e MODEL_NAME=hair_seg tensorflow/serving
   ```
