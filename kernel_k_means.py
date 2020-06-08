import numpy as np
from scipy.spatial.distance import pdist, squareform
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import os

# read the input images
def read_image(path):
    return cv2.imread(path).reshape((10000, 3))

# the implementation of new kernel which is basically multiplying two RBF kernels
def make_kernel(image, gamma_s, gamma_c):
    color_similarity = squareform(pdist(image, 'sqeuclidean'))

    coordinate = []
    for row in range(100):
        for col in range(100):
            coordinate.append((row, col))
    coordinate = np.array(coordinate)
    spatial_similarity = squareform(pdist(coordinate, 'sqeuclidean'))

    kernel = np.exp(-gamma_s * spatial_similarity) * np.exp(-gamma_c * color_similarity)

    return kernel

def initialization(image, k, method):
    if method == 'random':
        classification = np.random.randint(k, size=10000)
        return classification
    elif method == 'equal-divide':
        classification = []
        num_in_clusters = 10000 / k
        for data in range(10000):
            classification.append(data // num_in_clusters)
        return np.array(classification, dtype = np.int)

def plot_result(result, images):
    result = result.reshape((100, 100))
    image = plt.imshow(result)
    images.append([image])
    return images

def classify(kernel, classification, k):
    cluster_sum = np.zeros(k, dtype=np.int)
    all_term = np.zeros((10000, k), dtype=np.float32)

    for i in range(10000):
        cluster_sum[classification[i]] += 1

    for cluster in range(k):
        all_term[:, cluster] += (1 / cluster_sum[cluster]) ** 2 * np.sum(
            kernel[classification == cluster][:, classification == cluster])
        all_term[:, cluster] -= (2 / cluster_sum[cluster]) * np.sum(kernel[:, classification == cluster], axis=1)

    new_classification = np.argmin(all_term, axis=1)

    return new_classification

def kernel_k_means(image, k, name, gamma_s, gamma_c):
    methods = ['random', 'equal-divide']
    fig = plt.figure()
    for method in methods:
        images = []
        classification = initialization(image, k, method)
        kernel = make_kernel(image, gamma_s, gamma_c)

        while True:
            prev_classification = classification
            images = plot_result(classification, images)
            classification = classify(kernel, prev_classification, k)
            if np.array_equal(prev_classification, classification):
                break

        images = plot_result(classification, images)
        animate = animation.ArtistAnimation(fig, images, interval=500, repeat_delay=1000)
        animate.save(os.path.join(f'./{method}_{name}_{k}_kernel_k_means.gif'), writer=PillowWriter(fps=20))

if __name__ == "__main__":
    image1 = read_image('./image1.png')
    image2 = read_image('./image2.png')

    gamma_s = 1/(100*100)
    gamma_c = 1/(255*255)

    clusters = [2, 3, 4]
    for cluster in clusters:
        kernel_k_means(image1, cluster, 'image1', gamma_s, gamma_c)
        kernel_k_means(image2, cluster, 'image2', gamma_s, gamma_c)