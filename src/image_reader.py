from PIL import Image
import os


def read_image(image_path: str) -> (str, Image):
    file_name = image_path.split('/')[1]
    image = Image.open(os.path.realpath(image_path)).convert('L')
    return file_name, image
