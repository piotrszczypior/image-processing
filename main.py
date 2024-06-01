from src.image_reader import read_image
import src.plotting as plot
import numpy as np
from src.plotting import Orientation
import os
import src.transformation as transform

OUTPUT_DIRECTORY = 'output'


def open_image(image_path: str):
    file_name, image = read_image(image_path)
    plot.plot_image(file_name, image, title="Image of {}".format(file_name))


def create_gray_plot(image_path: str, coord: int, orientation: Orientation):
    file_name, image = read_image(image_path)
    data = np.array(image)
    plot.plot_gray_level(data, coord, orientation, file_name)


def crop_and_save_image(image_path: str, left: int, upper: int, right: int, lower: int, output_filename: str):
    file_name, image = read_image(image_path)
    cropped_image = image.crop((left, upper, right, lower))
    plot.plot_image(output_filename, cropped_image, title=None)
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    cropped_image.save(f'{OUTPUT_DIRECTORY}/{output_filename}')
    print(f'File saved under {OUTPUT_DIRECTORY}/{output_filename}')


def transformation_by_constant(image_path: str, constant: float):
    file_name, image = read_image(image_path)
    image = transform.multiply_by_constant(image, constant)
    plot.plot_image(file_name, image, title=f'transformed {file_name} by constant {constant}')


def transformation_by_logarithmic(image_path: str, constant: float):
    file_name, image = read_image(image_path)
    image = transform.logarithmic_transformation(image, constant)
    plot.plot_image(file_name, image,
                    title=f'transformed {file_name} by logarithmic transformation by {constant}')


def plot_image_after_change_in_grayscale(image_path: str, m: float, e: float):
    file_name, image = read_image(image_path)
    image = np.array(image)
    image = transform.change_in_grayscale_transformation(image, m, e)
    plot.plot_image(file_name, image, title=f'transformed {file_name} by change in grayscale transformation')


def plot_image_after_gamma_correction(image_path, gamma: float, constant=1):
    file_name, image = read_image(image_path)
    image = np.array(image)
    image = transform.gamma_correction(image, constant, gamma)
    plot.plot_image(file_name, image, title=f'{file_name} after gamma correction \n by c={constant} and gamma={gamma}')


def plot_histogram_of_image(image_path: str):
    file_name, image = read_image(image_path)
    image = np.array(image)
    plot.plot_histogram(image, title='Histogram of {}'.format(file_name))


def plot_image_after_histogram_equalization(image_path: str):
    file_name, image = read_image(image_path)
    image = np.array(image)
    image = transform.histogram_equalization(image)
    plot.plot_image(file_name, image, title=f'{file_name} after histogram equalization')
    plot.plot_histogram(image, title=f'Histogram of {file_name} after histogram equalization')


def plot_image_after_local_histogram_equalization(image_path: str, kernel_size: int):
    file_name, image = read_image(image_path)
    image = np.array(image)
    image = transform.local_histogram_equalization(image, kernel_size=kernel_size)
    plot.plot_image(file_name, image, title=f'{file_name} after local histogram equalization')
    plot.plot_histogram(image,
                        title=f'Histogram of {file_name} after local histogram equalization \n with mask = ({kernel_size},{kernel_size})')


def plot_image_after_enhancement_of_local_statistics(image_path: str):
    file_name, image = read_image(image_path)
    image = np.array(image)
    image = transform.enhance_quality_local_statistics(image)
    plot.plot_image(file_name, image, title=f'{file_name} after enhancement of local statistics')
    plot.plot_histogram(image, title=f'Histogram of {file_name} after enhancement of local statistics')


def plot_image_after_mean_filtering(image_path: str, kernel_size: int):
    file_name, image = read_image(image_path)
    image = np.array(image)
    image = transform.mean_filter(image, kernel_size)
    plot.plot_image(file_name, image, title=f'{file_name} after mean filtering with ({kernel_size}, {kernel_size})')


def plot_image_after_median_filter(image_path: str, kernel_size: int):
    file_name, image = read_image(image_path)
    image = np.array(image)
    image = transform.median_filter(image, kernel_size)
    plot.plot_image(file_name, image,
                    title=f'{file_name} after median \n filtering with ({kernel_size}, {kernel_size})')


def plot_image_after_minimum_filter(image_path: str, kernel_size: int):
    file_name, image = read_image(image_path)
    image = np.array(image)
    image = transform.minimum_filter(image, kernel_size)
    plot.plot_image(file_name, image,
                    title=f'{file_name} after minimum \n filtering with ({kernel_size}, {kernel_size})')


def plot_image_after_maximum_filter(image_path: str, kernel_size: int):
    file_name, image = read_image(image_path)
    image = np.array(image)
    image = transform.maximum_filter(image, kernel_size)
    plot.plot_image(file_name, image,
                    title=f'{file_name} after maximum \n filtering with ({kernel_size}, {kernel_size})')


def plot_image_after_gaussian_filter(image_path: str, kernel_size: int):
    file_name, image = read_image(image_path)
    image = np.array(image)
    image = transform.apply_gaussian_filter(image, kernel_size)
    plot.plot_image(file_name, image,
                    title=f'{file_name} after gaussian filtering \n with mask ({kernel_size}, {kernel_size})')


def plot_image_after_edge_detection(image_path: str, kernel_size: int):
    file_name, image = read_image(image_path)
    image = np.array(image)
    image = transform.sobel_filter(image, kernel_size)
    plot.plot_image(file_name, image, title=f'{file_name} after edge detection \n with mask = {kernel_size}')


def plot_image_after_laplace_operator(image_path: str, kernel: list[list[int]], constant: int):
    file_name, image = read_image(image_path)
    image = np.array(image)
    image = transform.laplace_operator(image, np.array(kernel), constant)
    plot.plot_image(file_name, image,
                    title=f'{file_name} after laplace operator \n with kernel {kernel} \n and constant = {constant}')


def plot_image_after_unsharp_masking_filer(image_path: str, strength: float):
    file_name, image = read_image(image_path)
    image = np.array(image)
    image = transform.unsharp_masking(image, strength=strength)
    plot.plot_image(file_name, image, title=f'{file_name} after unsharp masking filter \n with strength = {strength}')


def plot_image_after_high_boost_filtering(image_path: str, boost: float):
    file_name, image = read_image(image_path)
    image = np.array(image)
    image = transform.high_boost_filtering(image, k=boost)
    plot.plot_image(file_name, image, title=f'{file_name} after high boost filtering \n with boost = {boost}')
