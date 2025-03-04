!! update ported some of the network to pytorch for it's easier to convert to ncnn. Check it here:
[mobile phone portrait matting](https://github.com/ItchyHiker/mobile_phone_portrait_matting)
# Hair_Segmentation_Keras
Implement some light weight hair segmentation network with keras which can be used on mobile devices easily.

# Dataset
1. [CelebA Face/Hair segmentation database](http://www.cs.ubbcluj.ro/~dadi/face-hair-segm-database.html)
2. [Figaro](http://www.eecs.qmul.ac.uk/~urm30/Figaro.html)
3. [LFW Part Labels](http://vis-www.cs.umass.edu/lfw/part_labels/)

# Model
1. [DeeplabV3plus]: MobileNetV2 as the encoder
2. [PrismaNet](https://blog.prismalabs.ai/real-time-portrait-segmentation-on-smartphones-39c84f1b9e66): network architecture as described in the Prisma-AI blog
![a.jpg](https://cdn-images-1.medium.com/max/2400/1*y0S1deISIdDnbDhpqD4h4g.png)
3. [FastDeepMatting](https://arxiv.org/abs/1707.08289) 
![1.png](https://github.com/ItchyHiker/Hair_Segmentation_Keras/blob/master/assets/1.png)
![2.png](https://github.com/ItchyHiker/Hair_Segmentation_Keras/blob/master/assets/2.png)
4. PrismaNet + FastDeepMatting: base PrismaNet architecture plus the feathering block in fast deep matting

# Training

# Results
## DeeplabV3plus
![4.jpg](https://github.com/ItchyHiker/Hair_Segmentation_Keras/blob/master/imgs/results/DeeplabV3plus/1803151818-00000296.jpg)
## PrismaNet
![4.jpg](https://github.com/ItchyHiker/Hair_Segmentation_Keras/blob/master/imgs/results/PrismaNet/1803151818-00000296.jpg)
## FastDeepMatting

## PrismaNet + FastDeepMatting
![4.jpg](https://github.com/ItchyHiker/Hair_Segmentation_Keras/blob/master/imgs/results/PrismaMattingNet/4.jpg)

**Matting methods used channel split operation which is unportable to CoreML as I wrote.**

# Serving
I have also use this model to predict hair color with tensorflow serving. Follow instructions bellow.
1. Use this scripts to ```python serving/keras_to_serving.py``` generate model used for tensorflow serving deployment.
2. Prepare tensorflow serving environments. Please refer to [README.md](https://github.com/ItchyHiker/Hair_Segmentation_Keras/blob/master/serving/README.md)

# Todo

- [x] port to iOS using CoreML [Hair_Segmentation_iOS](https://github.com/ItchyHiker/Hair_Segmentation_iOS)
- [x] tutorial about how to serve the model using tensorflow serving and use the results for hair color prediction. [Hair Color Predict Using TensorFlow Serving](https://github.com/ItchyHiker/Hair_Segmentation_Keras/blob/master/serving/README.md)
- [ ] update to TensorFlow2.0 Keras API
