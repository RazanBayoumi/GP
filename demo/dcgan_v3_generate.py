import os
import sys
import numpy as np
from random import shuffle
import h5py

def main():
    seed = 42
    np.random.seed(seed)

    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    img_dir_path = current_dir + '/data/jpg'
    txt_dir_path = current_dir + '/data/sample_caption_vectors.hdf5'
    model_dir_path = current_dir + '/models'

    img_width = 64
    img_height = 64

    from keras_text_to_image.library.dcgan_v3 import DCGanV3
    from keras_text_to_image.library.utility.image_utils import img_from_normalized_img
    from keras_text_to_image.library.utility.img_cap_loader import load_normalized_img_and_its_text

    image_label_pairs = load_normalized_img_and_its_text(img_dir_path, txt_dir_path, img_width=img_width, img_height=img_height)

    shuffle(image_label_pairs)

    gan = DCGanV3()
    gan.load_model(model_dir_path)

    h = h5py.File(txt_dir_path)
    caption_vectors = np.array(h['vectors'])
    caption_image_dic = {}

    for cn, caption_vector in enumerate(caption_vectors):
        generated_image = gan.generate_image_from_text(caption_vector)
        generated_image.save(current_dir + '/data/outputs/' + DCGanV3.model_name + '-generated-' + str(cn)  + '.jpg')


if __name__ == '__main__':
    main()
