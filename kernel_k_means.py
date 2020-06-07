import numpy as np
from scipy.spatial.distance import pdist, squareform
import cv2
import matplotlib.pyplot as plt
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
    #C = np.array([np.random.randint(100, size=2) for _ in range(k)], dtype = np.float32)
    if method == 'random':
        classification = np.random.randint(k, size=10000)
        return classification

def plot_result(result, file_name, clusters, iteration):
    result = result.reshape((100, 100))
    plt.imshow(result)
    plt.savefig(os.path.join(f'./{clusters}/{file_name}_{iteration}.png'))


def classify(kernel, classification, k):
    # this segment of code is to compute the third part of formula
    cluster_sum = np.zeros(k, dtype=np.int)
    third_term = np.zeros(k, dtype=np.float32)

    for i in range(10000):
        cluster_sum[classification[i]] += 1

    for cluster in range(k):
        for p in range(10000):
            for q in range(10000):
                if classification[p] == cluster and classification[q] == cluster:
                    third_term[cluster] += kernel[p][q]
        if cluster_sum[cluster] == 0:
            continue
        third_term[cluster] /= (cluster_sum[cluster] ** 2)

    # this segment of code is to compute the all part of formula
    second_term = np.zeros((10000, k), dtype=np.float32)
    all_term = np.zeros((10000, k), dtype=np.float32)
    for j in range(10000):
        for cluster in range(k):
            for n in range(10000):
                if classification[n] == cluster:
                    second_term[j, cluster] += kernel[j][n]
            if cluster_sum[cluster] != 0:
                second_term[j][cluster] = (second_term[j][cluster] * 2) / cluster_sum[cluster]
            all_term[j][cluster] = kernel[j][j] - second_term[j][cluster] + third_term[cluster]

    new_classification = np.argmin(all_term, axis=1)
    return new_classification


def kernel_k_means(image, k, name):
    methods = ['random']
    for method in methods:
        classification = initialization(image, k, method)
        kernel = make_kernel(image, gamma_s, gamma_c)
        iteration = 0

        while True:
            print(f"iteration: {iteration}")
            prev_classification = classification
            plot_result(classification, name, k, iteration)
            iteration += 1
            classification = classify(kernel, prev_classification, k)
            if np.array_equal(prev_classification, classification) or iteration >= 20:
                break

        plot_result(classification, name, k, iteration)

if __name__ == "__main__":
    image1 = read_image('./image1.png')
    image2 = read_image('./image2.png')

    gamma_s = 0.01
    gamma_c = 0.01

    clusters = [2, 3, 4]
    for cluster in clusters:
        kernel_k_means(image1, cluster, 'image1')