import re
import argparse

import coremltools
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

from utils.custom_objects import custom_objects

def main(input_model_path):
    """convert keras hdf5 model to CoreML model"""
    output_model_path = re.sub(r'h5$', 'mlmodel', input_model_path)
    custom_funcs = custom_objects()
    print(custom_funcs)

    with CustomObjectScope({'relu6': custom_funcs['relu6']}):
        model = load_model(input_model_path, custom_objects=custom_objects())
        model.summary()
        print(model.input)
        print(model.output)
        
        coreml_model = coremltools.converters.keras.convert(model, 
                                                        input_names='input',
                                                        image_input_names='input',
                                                        output_names='output',
                                                        red_bias=-1.,
                                                        green_bias=-1,
                                                        blue_bias=-1,
                                                        image_scale=2./255,
                                                        add_custom_layers=True,
                                                        custom_conversion_functions=custom_funcs)
    coreml_model.save(output_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_model_path',
        type=str,
        default='models/CelebA_DeeplabV3plus_256.h5',
    )
    args, _ = parser.parse_known_args()
    main(**vars(args))
