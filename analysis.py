import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import pandas as pd
import ast
import glob
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import make_axes_locatable
from umap import UMAP
from skimage import io
from itertools import product

def load_data():
    """
    Load the results .csv file as pandas array
    :return: pandas dataframe
    """
    data = pd.read_csv('Result.csv', sep=';')
    return data


def patch_list_to_points(path):
    """
    Extract all annotation lists from patchlist as a [n, 4] numpy array
    :param path: path to patchlist
    :return: numpy array
    """
    with open(path, 'r') as file:
        liste = [line.strip().split(";") for line in file]

    all_labels = []
    images = []
    coords = []
    for row in liste:
        if int(row[4]) not in (1,2):
            continue
        all_labels.append(ast.literal_eval(row[5]))
        img_name = row[0].split("\\")[-1]
        images.append(img_name)
        coords.append([int(row[1]), int(row[2])])

    all_labels = np.array(all_labels)
    coords = np.array(coords)

    return all_labels, coords, images

def filter_points(points):
    """
    Filter points to remove those with too many 0.5s
    :param points:
    :return:
    """
    temp = points[:] == 0.5

    temp = np.sum(temp, axis=1)

    indices = temp < 2
    out_points = points[indices, :]
    print(points.shape, out_points.shape)
    return out_points, indices

def combine_labels(label_list, coords_list, images_list):
    """
    Concatenate labels from a list of extracter patchlists
    :param label_list:
    :return: numpy array
    """
    labels = np.concatenate(label_list, axis=0)
    coords = np.concatenate(coords_list, axis=0)
    images = np.concatenate(images_list)

    #labels, indices = filter_points(labels)
    #coords = coords[indices]
    #images = images[indices]
    return labels, coords, images


def scatter_labels(labels, legend, label_names=('Ruffles Qty', 'Ruffles Size', 'Fragmentation', 'Diffusion')):
    """
    Scatter plot of feature pairs
    :param labels:
    :param legend:
    :param label_names:
    :return:
    """
    for i in range(labels[0].shape[1]):
        for j in range(labels[0].shape[1]):
            if j <= i:
                continue
            for l, lbl in zip(labels, legend):
                #l = filter_points(l)
                x = l[:, i]
                y = l[:, j]

                plt.scatter(x, y, alpha=0.45, label=lbl)
            plt.xlabel(label_names[i])
            plt.ylabel(label_names[j])
            plt.xlim(-0.1,1.1)
            plt.ylim(-0.1,1.1)
            plt.legend()
            plt.show()

def label_clustering(labels, seed=105, n_clusters=7):
    """
    Cluster labels and visualize
    :param labels_list:
    :param seed:
    :return:
    """
    knn = KMeans(n_clusters=n_clusters, random_state=seed)

    knn_labels = knn.fit_predict(labels).astype('int').ravel()

    return labels, knn_labels, knn

def load_image(image_name):
    """
    Find and load image into np array + convert into RGB for pretty pictures
    :param image_name: name of the image to load
    :return: image as RGB numpy array
    """
    image_path = glob.glob(f"images/*/*/*/{image_name}", recursive=True)[0]
    image = io.imread(image_path).astype('float32')


    image[0] *= 255.0 / np.percentile(image[0], 99)
    image[1] *= 255.0 / np.percentile(image[1], 99)
    image = np.clip(image, 0, 255.0)
    image = (image).astype('uint8')


    image = np.array([image[1], image[0], np.zeros(image.shape[1:])])
    image = np.moveaxis(image, 0, -1).astype('uint8')
    return image


def localize_clusters(image, coords, labels, size=64):
    """
    Localize the crops of each cluster in the original image
    :param image: original image
    :param coords: np array of coordinates
    :param labels: np array of classes
    :param size: Size of the crops
    :return:
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    ax = plt.subplot()
    ax.imshow(image)

    for (y,x), c in zip(coords, labels):
        rect = patches.Rectangle((y,x), size, size, alpha=0.62, facecolor=colors[c], edgecolor='white', linewidth=0.33)
        ax.add_patch(rect)

    return ax

def umap_clusters(X, y, seed=105):
    """
    Generate the UMAP visualization of the clustering
    :param X: numpy array of features
    :param y: numpy array of classes found by clustering
    :param seed: random state
    :return:
    """
    #pca = TSNE(n_components=2, perplexity=30)
    umap = UMAP(n_components=2, random_state=seed)
    umap_data = umap.fit_transform(X)

    for l in np.unique(y):
        idx = y == l

        plt.scatter(umap_data[idx,0], umap_data[idx,1], alpha=0.45, label=l)
        #plt.xlabel(label_names[i])
        #plt.ylabel(label_names[j])
    plt.legend(title='Cluster index')
    plt.title('UMAP Visualization of K-Means Clustering')
    plt.savefig('umap.png', bbox_inches='tight', dpi=450)
    plt.close()

def quantify_clusters(X, y):
    """
    Show average features of different clusters
    :param X: features
    :param y: cluster indicex
    :return:
    """
    cluster_array = []
    for l in np.unique(y):
        idx = y == l
        l_X = X[idx]

        cluster_array.append(list(np.median(l_X, axis=0)))

    cluster_array = np.array(cluster_array)

    ax = plt.subplot()
    mat = ax.matshow(cluster_array)
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(['Ruffles Qty', 'Ruffles Size', 'Fragmentation', 'Diffusion'], rotation=45)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mat, cax=cax)
    plt.savefig('codebar.png', bbox_inches='tight', dpi=450)
    plt.show()

def quantify_stuff():
    """
    Just count how many of each class there is and do a lil bar graph
    """

    # Linear - Lots of Ruffles
    # Small ruffles - Big ruffles
    # Continuous - Fragmented
    # Sharp - Diffuse

    features = ['Ruffles Qty', 'Ruffles Size', 'Fragmentation', 'Diffusion']
    classes = ['< 0.5', '= 0.5', '> 0.5']

    data = load_data()

    for i, row in data.iterrows():
        classifications = np.array(ast.literal_eval(row[4]))

        for c in range(3):
            offset = 0.20*(c-1)
            x = np.array([1,2,3,4])+offset
            plt.bar(x,classifications[:,c], width=0.20, label=classes[c])

        plt.xticks((1,2,3,4), features)
        plt.ylabel('Number of crops')
        plt.xlabel('Feature')
        plt.title(f'Image {i+1}')
        plt.legend()
        plt.savefig(f'bars_image_{i+1}.png', bbox_inches='tight', dpi=450)
        plt.close()

def classify_by_values(X):
    """
    Transform values to classes (<
    :param X:
    :return:
    """

    out_X = np.where(X < 0.5, 0, X)
    out_X = np.where(out_X > 0.5, 2, out_X)
    out_X = np.where(out_X == 0.5, 1, out_X)

    return out_X.astype('uint8')

if __name__== '__main__':
    paths = ('patchlist1_32.txt', 'patchlist2.txt', 'patchlist3.txt')
    #paths = ('patchlist1_32.txt',)
    feature_names = ['Ruffles_Qty', 'Ruffles_Size', 'Fragmentation', 'Diffusion']

    features_list, coords_list, images_list = [], [], []
    for p in paths:
        features, coords, images = patch_list_to_points(p)
        features_list.append(features)
        coords_list.append(coords)
        images_list.append(images)

    features, coords, images = combine_labels(features_list, coords_list, images_list)
    # Basic classification
    X = features
    class_X = classify_by_values(X)

    quantify_stuff()
    for c in range(4):
        y = class_X[:, c]
        for img_name in np.unique(images):
            im_y = y[images == img_name]
            im_coords = coords[images == img_name]

            image = load_image(img_name)
            localize_clusters(image, im_coords, im_y)
            plt.title(feature_names[c])
            plt.savefig(f'{feature_names[c]}_{img_name[:-4]}.png', bbox_inches='tight', dpi=450)
            plt.close()


    # Clustering and UMAP
    n_clusters = 7
    X, y, knn_model = label_clustering(features, n_clusters=n_clusters)
    for img_name in np.unique(images):
        im_y = y[images == img_name]
        im_coords = coords[images == img_name]

        image = load_image(img_name)
        localize_clusters(image, im_coords, im_y)
        plt.title('Clustering Classes')
        plt.savefig(f'clustering_{img_name[:-4]}.png', bbox_inches='tight', dpi=450)
        plt.close()
    umap_clusters(X, y)

    quantify_clusters(X, y)
    #labels = combine_labels(labels)
    #scatter_labels(labels, paths)