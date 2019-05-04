import numpy as np
from keras.models import load_model


class ImageEncoder:
    # Parameters for encoding line image in features
    nb_features_encoding = 8  # Nb of features in output of encoder
    encoder_input_width = 384
    encoder_input_height = 256

    def __init__(self, image_analyzer, model_path='../deep-learning-model/encoder_8_feats.h5'):
        self.image_analyzer = image_analyzer
        self.model_path = model_path
        # Load model for encoding line images into features
        self.encoder_model = load_model(self.model_path)
        self.encoded_image_line = None

    def get_nb_features_encoding(self):
        return self.nb_features_encoding

    def encode_image_ligne(self, image):
        input_height = image.shape[0]
        input_width = image.shape[1]
        padding_height = (self.encoder_input_height - input_height) // 2
        padding_width = (self.encoder_input_width - input_width) // 2

        image_padded = np.zeros((1, self.encoder_input_height, self.encoder_input_width), dtype=np.float32)
        image_padded[0, padding_height:-padding_height, padding_width:-padding_width] = image
        image_padded = image_padded / 255.0

        encoded = self.encoder_model.predict(image_padded)[0]

        return encoded

    def execute(self):
        image_line = self.image_analyzer.get_image_ligne()
        self.encoded_image_line = self.encode_image_ligne(image_line)

    def get_encoded_image(self):
        return self.encoded_image_line
