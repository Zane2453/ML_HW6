import numpy as np
from scipy.spatial.distance import pdist, squareform
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import os

gamma_s = 1/(100*100)
gamma_c = 1/(255*255)

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

def nearest(data_point, cluster_centers):
    min_dist = 1e100
    cluster_num = np.shape(cluster_centers)[0]
    for i in range(cluster_num):
        dist = np.linalg.norm(data_point - cluster_centers[i, :])
        # choose the shortest distance
        if min_dist > dist:
            min_dist = dist
    return min_dist

def initialization(Laplacian, k, method):
    C = np.array([np.random.randint(100, size=k) for _ in range(k)], dtype = np.float32)
    if method == 'random':
        classification = np.random.randint(k, size=10000)
        return C, classification
    elif method == 'k-means++':
        classification = np.random.randint(k, size=10000)
        data_num, dimension = np.shape(Laplacian)
        first_cluster = np.random.randint(data_num, size=1, dtype=np.int)
        C[0, :] = Laplacian[first_cluster, :]
        for i in range(1, k):
            distance = np.zeros(data_num, dtype=np.float32)
            for j in range(data_num):
                distance[j] = nearest(Laplacian[j, :], C[0:i, :])
            distance = distance / distance.sum()
            cluster = np.random.choice(data_num, size=1, p=distance)
            C[i, :] = Laplacian[cluster, :]
        return C, classification

def plot_result(result, images):
    result = result.reshape((100, 100))
    image = plt.imshow(result)
    images.append([image])

    return images

def update_C(Laplacian, classification, k):
    new_C = np.zeros((k, k), dtype=np.float32)
    cluster_sum = np.zeros((k, k), dtype=np.int)
    for data in range(10000):
        cluster_sum[classification[data]] += np.ones(k, dtype=np.int)
        new_C[classification[data]] += Laplacian[data]
    for cluster in range(k):
        if cluster_sum[cluster][0] == 0:
            cluster_sum[cluster] += np.ones(k, dtype=np.int)
    new_C = np.true_divide(new_C, cluster_sum)

    return new_C

def classify(Laplacian, C, k):
    new_classification = np.zeros(10000, dtype=np.int)
    distance = np.zeros(k, dtype=np.float)

    for data in range(10000):
        for cluster in range(k):
            delta = abs(np.subtract(Laplacian[data, :], C[cluster, :]))
            distance[cluster] = np.square(delta).sum(axis=0)
        new_classification[data] = np.argmin(distance)

    return new_classification

def plot_eigenspace(Laplacian, classification, k, method, way, name):
    colors = ['red', 'green', 'blue']
    plt.clf()
    plt.title(f'{method}_{name}_{k}_{way}_eigenspace')
    if k == 2:
        for cluster in range(k):
            for data in range(10000):
                if classification[data] == cluster:
                    plt.scatter(Laplacian[data][0], Laplacian[data][1], s=8, c=colors[cluster])
    elif k == 3:
        ax = plt.figure().add_subplot(111, projection='3d')
        for cluster in range(k):
            for data in range(10000):
                if classification[data] == cluster:
                    ax.scatter(Laplacian[data][0], Laplacian[data][1], Laplacian[data][2], c=colors[cluster], marker='o')
    else:
        return
    plt.savefig(os.path.join(f'./{method}_{name}_{k}_{way}_eigenspace.png'))

def k_means(Laplacian, image, k, name, way):
    methods = ['random', 'k-means++']
    images = []
    fig = plt.figure()
    for method in methods:
        C, classification = initialization(Laplacian, k, method)
        iteration = 0

        while True:
            prev_classification = classification
            iteration += 1
            images = plot_result(classification, images)
            classification = classify(Laplacian, C, k)
            if np.array_equal(prev_classification, classification) or iteration >= 100:
                break

            C = update_C(Laplacian, classification, k)

        images = plot_result(classification, images)
        animate = animation.ArtistAnimation(fig, images, interval=500, repeat_delay=1000)
        animate.save(os.path.join(f'./{method}_{name}_{k}_{way}.gif'), writer=PillowWriter(fps=20))

        plot_eigenspace(Laplacian, classification, k, method, way, name)

def normalize_cut(image, k, name):
    weight = make_kernel(image, gamma_s, gamma_c)
    degree = np.diag(np.power(np.sum(weight, axis=1), -0.5))
    L_sym = np.eye(10000) - np.dot(degree, np.dot(weight, degree))
    eigen_values, eigen_vectors = np.linalg.eig(L_sym)
    U = eigen_vectors[:, np.argsort(eigen_values)[1: k + 1]].real.astype(np.float32)

    # normalization
    T = U.copy()
    for row in range(10000):
        sum = np.sum(np.power(U[row], 2)) ** 0.5
        if sum != 0:
            T[row, :] /= sum

    k_means(T, image, k, name, 'normalize_cut')

def radio_cut(image, k, name):
    weight = make_kernel(image, gamma_s, gamma_c)
    degree = np.diag(np.sum(weight, axis=1))
    L = degree - weight
    eigen_values, eigen_vectors = np.linalg.eig(L)
    U = eigen_vectors[:, np.argsort(eigen_values)[1: k + 1]].real.astype(np.float32)

    k_means(U, image, k, name, 'radio_cut')

if __name__ == "__main__":
    image1 = read_image('./image1.png')
    image2 = read_image('./image2.png')

    clusters = [2, 3, 4]
    for cluster in clusters:
        normalize_cut(image1, cluster, 'image1')
        normalize_cut(image2, cluster, 'image2')
        radio_cut(image1, cluster, 'image1')
        radio_cut(image2, cluster, 'image2')
