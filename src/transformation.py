import numpy as np
import cv2 as cv
from scipy.ndimage import uniform_filter


def multiply_by_constant(image, constant: float):
    image = np.array(image)
    return np.uint8(np.clip(image * constant, 0, 255))


def logarithmic_transformation(image, constant: float):
    image = np.array(image, dtype=float)
    c = constant / np.log(1 + np.max(image))
    log_transformed = c * np.log(1 + image)
    return np.uint8(np.clip(log_transformed, 0, 255))


def change_in_grayscale_transformation(image, m=0.45, e=8):
    image = np.array(image)
    image = image + 1
    image = image.astype(np.float32) / 255.0
    transformed = 1 / np.power(1 + (m / image), e)
    return np.uint8(transformed * 255)


def gamma_correction(image, constant=1, gamma=0.5):
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    corrected = np.power(image, gamma)
    return multiply_by_constant(corrected, constant * 255)


def histogram_equalization(image):
    return cv.equalizeHist(image)


def local_histogram_equalization(image, kernel_size=3):
    clahe = cv.createCLAHE(tileGridSize=(kernel_size, kernel_size))
    return clahe.apply(image)


def local_statistics(image, size=3):
    local_mean = uniform_filter(image, size=size)
    local_mean_sq = uniform_filter(image ** 2, size=size)
    local_variance = local_mean_sq - local_mean ** 2
    return local_mean, local_variance


def enhance_quality_local_statistics(image, constant=22.8, k0=0, k1=0.1, k2=0, k3=0.1):
    image = np.array(image, dtype=float)
    mG = np.mean(image)
    sigmaG = np.std(image)
    mSxy, varSxy = local_statistics(image)
    sigmaSxy = np.sqrt(varSxy)
    enhanced_image = np.copy(image)
    condition1 = (k0 * mG <= mSxy) & (mSxy <= k1 * mG)
    condition2 = (k2 * sigmaG <= sigmaSxy) & (sigmaSxy <= k3 * sigmaG)
    enhancement_condition = condition1 & condition2
    enhanced_image[enhancement_condition] = constant * image[enhancement_condition]
    return np.clip(enhanced_image, 0, 255).astype(np.uint8)


def mean_filter(image, kernel_size=3):
    return cv.blur(image, (kernel_size, kernel_size))


def median_filter(image, kernel_size=3):
    return cv.medianBlur(image, kernel_size)


def minimum_filter(image, kernel_size=3):
    return cv.erode(image, np.ones((kernel_size, kernel_size), np.uint8))


def maximum_filter(image, kernel_size=3):
    return cv.dilate(image, np.ones((kernel_size, kernel_size), np.uint8))


def apply_gaussian_filter(image, kernel_size):
    kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    sigma = (kernel_size - 1) / 6
    return cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def sobel_filter(img, kernel_size=3):
    img = cv.GaussianBlur(img, (kernel_size, kernel_size), 0)
    Gx = cv.Sobel(src=img, ddepth=cv.CV_64F, dx=1, dy=0, ksize=kernel_size)
    Gy = cv.Sobel(src=img, ddepth=cv.CV_64F, dx=0, dy=1, ksize=kernel_size)
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    return (G / np.max(G) * 255).astype(np.uint8)


def laplace_operator(image, kernel, constant):
    image = apply_gaussian_filter(image, kernel_size=3)
    laplace = cv.filter2D(src=image, ddepth=cv.CV_16S, kernel=kernel)
    image = image + constant * laplace
    image = cv.convertScaleAbs(image)
    return apply_gaussian_filter(image, kernel_size=3)


def unsharp_masking(image, strength=1.0):
    blurred =  cv.GaussianBlur(image, (5, 5), 0)
    mask = cv.subtract(image, blurred)
    return cv.addWeighted(image, 1.0, mask, strength, 0)


def high_boost_filtering(image, k=1.5):
    blurred = apply_gaussian_filter(image, kernel_size=0)
    mask = cv.subtract(image, blurred)
    return cv.addWeighted(image, 1.0 + k, mask, k, 0)
