"""
train hair segmentation model on CelebA dataset
"""
import argparse
import os
import glob 

from keras import callbacks
from keras.optimizers import SGD, Adam

from utils.data import load_data
from utils.learning_rate import create_lr_schedule
from utils.loss import dice_coef_loss, dice_coef, recall, precision
from nets.DeeplabV3plus import DeeplabV3plus
from nets.FastDeepMatting import FastDeepMatting
from nets.Prisma import PrismaNet
from nets.PrismaMattingNet import PrismaMattingNet

def train(data_path, model_name, input_shape, pretrained_weights_path, epochs, batch_size):

    # create training and validation data generators
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    train_gen, val_gen = load_data(train_path, val_path, target_size=input_shape[0:2], batch_size=batch_size)
    
    # calculate number of training and validation samples for fit_generator methods
    nb_train_samples = len(glob.glob(train_path + '/images/images/*.jpg'))
    nb_val_samples = len(glob.glob(val_path + '/images/images/*.jpg'))
    print("number of training samples:", nb_train_samples)
    print("number of validation samples:", nb_val_samples)
    
    # load model and weights
    if model_name == 'DeeplabV3plus':
        model = DeeplabV3plus(input_shape=input_shape, classes=1, alpha=1)
    if model_name == "PrismaNet":
        model = PrismaNet(input_shape=input_shape, n_classes=1) 
    if model_name == "FastDeepMatting":
        model = FastDeepMatting(input_shape=(256, 256, 3), n_classes=1) 
    if model_name == "PrismaMattingNet":
        model = PrismaMattingNet(input_shape=(256, 256, 3), n_classes=1)
    if pretrained_weights_path is not None:    
        model.load_weights(os.path.expanduser(pretrained_weights_path), by_name=True)
    model.summary()
        
    # keras callbacks
    lr_base = 0.0005 * (float(batch_size) / 16)
    learning_rate = callbacks.LearningRateScheduler(
        create_lr_schedule(epochs, lr_base=lr_base, mode='power_decay'))

    tensorboard = callbacks.TensorBoard(log_dir='./logs')

    checkpoint_path = 'checkpoints/checkpoint_weights.{epoch:02d}-{val_loss:.2f}.h5'
    checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_weights_only=False,
                                           save_best_only=True)
    
    # unfreeze all the layers
    for layer in model.layers:
        layer.trainable = True
    
    model.compile(
        optimizer=Adam(lr=0.001),
        # loss='sparse_categorical_crossentropy',
        loss=dice_coef_loss,
        metrics=[
            dice_coef,
            recall,
            precision,
            'binary_crossentropy',   
        ],       
    )

    model.fit_generator(
        generator = train_gen,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs,
        validation_data = val_gen,
        validation_steps = nb_val_samples // batch_size,
        callbacks = [tensorboard, learning_rate, checkpoint],
    )
    
    trained_model_path = 'models/CelebA_{0}_{1}_hair_seg_model.h5'.format(model_name, input_shape[0])
    model.save(trained_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='/home/ubuntu/dataset/skin',
        help='dataset path, which includes ./train and ./val to suit flow_from_directory function'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='DeeplabV3plus',
        help='which segmentation network to use'
    )
    parser.add_argument(
        '--input_shape',
        type=tuple,
        default=(224, 224, 3),
        help='input image shape',
    )
    parser.add_argument(
        '--pretrained_weights_path',
        type=str,
        # default='checkpoints/checkpoint_weights.138--0.91.h5',
        # default='models/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
        help='weights from pretrained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=150,
        help='number of epochs to train'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
    )
    args, _ = parser.parse_known_args()
    train(**vars(args))
