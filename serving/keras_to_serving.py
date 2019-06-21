import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from utils.custom_objects import custom_objects

model = load_model('models/CelebA_DeeplabV3plus_256_hair_seg_model.h5', custom_objects=custom_objects())
print('model input:', model.input)
print('model output:', model.output)
model.summary()

K.set_learning_phase(0)

export_path='serving/hair_seg/1'
with K.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={"input_image":model.input},
        outputs={t.name: t for t in model.outputs}
    )

