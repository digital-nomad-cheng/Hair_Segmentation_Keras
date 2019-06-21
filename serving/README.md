# TensorFlow Serving Tutorial
1. Download and install docker: https://docs.docker.com/docker-for-mac/install/
2. (optional) Install nividia-docker if you want to run your application on GPU
3. Pull TensorFlow Serving Image: ```docker pull tensorflow/serving```
4. Prepare TensorFlow Serving Serving using docker:
   ```
   docker run -it -p 8501:8501 -v "$(pwd)/serving/hair_seg/:/models/hair_seg" -e MODEL_NAME=hair_seg tensorflow/serving
   ```
5. Get prediction results using client program. I have provided a client script you can use [client.py](https://github.com/ItchyHiker/Hair_Segmentation_Keras/blob/master/serving/client.py).



