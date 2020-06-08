from pybalu import feature_extraction
from skimage import filters, morphology
import numpy as np
import cv2


def features_from_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    binary_img = segmented_img(img.copy())

    # geometric features
    flusser = feature_extraction.flusser_features(binary_img)
    hu = feature_extraction.hugeo_features(binary_img)
    fourier = feature_extraction.fourier_des_features(binary_img)

    # intensity features
    mean, std, kurtosis, skew, laplacian, gradient = feature_extraction.basic_int_features(img)
    gabor = feature_extraction.gabor_features(img)

    # descriptors con default values
    lbp = local_binary_patterns(img, shape=(8, 8), mapping='uniform')
    hog = histogram_oriented_gradients(img, n_orientations=10, shape=(8, 8))
    har = haralick_textures(img, dist=3)

    # geometric data
    img_features = {}
    img_features.update({f'geo: flusser{i}': flusser[i] for i in range(len(flusser))})
    img_features.update({f'geo: hu{i}': hu[i] for i in range(len(hu))})
    img_features.update({f'geo: fourier{i}': fourier[i] for i in range(len(fourier))})
    img_features.update({'int: mean': mean, 'int: std': std, 'int: kurtosis': kurtosis,
                         'int: skew': skew, 'int: laplacian': laplacian,
                         'int: gradient': gradient})
    img_features.update({f'int: gabor{i}': gabor[i] for i in range(len(gabor))})
    img_features.update({f'des: lbp{i}': lbp[i] for i in range(len(lbp))})
    img_features.update({f'des: hog{i}': hog[i] for i in range(len(hog))})
    img_features.update({f'des: har{i}': har[i] for i in range(len(har))})
    return img_features


def segmented_img(img):
    # thresholding
    threshold = filters.threshold_otsu(img)
    binary = (img > threshold)

    # removemos pequeñas interrupciones y cerramos los contornos
    clean = morphology.remove_small_objects(binary, binary.size // 100.0, connectivity=2)
    closed = morphology.binary_closing(clean)
    region = morphology.remove_small_holes(closed, binary.size // 100.0, connectivity=2)

    return region.astype(np.int64)


def local_binary_patterns(img, shape, mapping):
    pixel_w, pixels_h = shape
    return feature_extraction.lbp_features(img, hdiv=pixel_w, vdiv=pixels_h, mapping=mapping)


def histogram_oriented_gradients(img, n_orientations, shape):
    # options, con esto se dan 512 hog features
    pixel_w, pixels_h = shape
    bins = n_orientations
    return feature_extraction.hog_features(img, v_windows=pixels_h, h_windows=pixel_w, n_bins=bins)


def haralick_textures(img, dist):
    return feature_extraction.haralick_features(img, distance=dist)
