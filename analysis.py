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

DEFAULT_COLORS = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')

def load_data():
    """
    Load the results .csv file as pandas array
    :return: pandas dataframe
    """
    data = pd.read_csv('Result.csv', sep=';')
    return data


def patch_list_to_points(path, filter=False, structures=(1,)):
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
    all_structures = []
    for row in liste:
        if int(row[4]) not in structures:
            continue
        all_labels.append(ast.literal_eval(row[5]))
        img_name = row[0].split("\\")[-1]
        images.append(img_name)
        coords.append([int(row[1]), int(row[2])])
        all_structures.append(int(row[4]))

    all_labels = np.array(all_labels)
    coords = np.array(coords)
    images = np.array(images)
    all_structures = np.array(all_structures)

    if filter:
        all_labels, coords, images, all_structures = filter_points(all_labels, coords, images, all_structures)

    return all_labels, coords, images, all_structures

def filter_points(points, coords, images, structures):
    """
    Filters out any point with labels equal to 0.5
    :param points: array of shape (N, 4) of points with 4 features
    :return: filtered array
    """
    temp = points[:] == 0.5

    temp = np.sum(temp[:, 0:2], axis=1)

    indices = temp < 1
    out_points = points[indices, :]
    out_coords = coords[indices]
    out_images = images[indices]
    out_structures = structures[indices]

    return out_points, out_coords, out_images, out_structures

def filter_structures_only(points, coords, images, structures):
    """
    Only keep points with strcutures == 1
    :param points:
    :param coords:
    :param images:
    :param structures:
    :return:
    """
    indices = structures == 1

    out_points = points[indices, :]
    out_coords = coords[indices]
    out_images = images[indices]
    out_structures = structures[indices]

    return out_points, out_coords, out_images, out_structures


def combine_labels(label_list, coords_list, images_list, structures_list):
    """
    Concatenate labels from a list of extracter patchlists
    :param label_list:
    :return: numpy array
    """
    labels = np.concatenate(label_list, axis=0)
    coords = np.concatenate(coords_list, axis=0)
    images = np.concatenate(images_list)
    structures = np.concatenate(structures_list)

    return labels, coords, images, structures


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
    Cluster labels
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


def localize_clusters(image, coords, labels, size=64, n_examples=0, im_name='', colors=DEFAULT_COLORS):
    """
    Localize the crops of each cluster in the original image
    :param image: original image
    :param coords: np array of coordinates
    :param labels: np array of classes
    :param size: Size of the crops
    :param n_examples: Number of examples of each class to extract
    :return:
    """

    ax = plt.subplot()
    ax.axis('off')
    ax.imshow(image)

    counts = [0 for _ in range(7)]
    for (y,x), c in zip(coords, labels):
        rect = patches.Rectangle((y,x), size, size, alpha=0.62, facecolor=colors[c], edgecolor='white', linewidth=0.33)
        ax.add_patch(rect)

        if np.random.random() < 0.25 and n_examples > 0:
            if counts[c] < n_examples:
                counts[c] += 1
                crop = image[x:x+size, y:y+size, 0]
                io.imsave(f'crops/class_{c}-crop_{counts[c]}_{im_name}.tif', crop)

    return ax

def umap_clusters(X, y, method='UMAP', seed=105):
    """
    Generate the UMAP visualization of the clustering
    :param X: numpy array of features
    :param y: numpy array of classes found by clustering
    :param seed: random state
    :return:
    """
    if method == 'UMAP':
        umap = UMAP(n_components=2, random_state=seed)
    elif method == 'TSNE':
        umap = TSNE(n_components=2, perplexity=30, random_state=seed)
    elif method == 'PCA':
        umap = PCA(n_components=2, random_state=seed)

    umap_data = umap.fit_transform(X)

    for l in np.unique(y):
        idx = y == l

        plt.scatter(umap_data[idx,0], umap_data[idx,1], alpha=0.45, label=l)
        #plt.xlabel(label_names[i])
        #plt.ylabel(label_names[j])
    plt.legend(title='Cluster index')
    plt.title(f'{method} Visualization of K-Means Clustering')
    plt.savefig(f'{method}.pdf', bbox_inches='tight', dpi=450)
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
    ax.set_xticklabels(['Class 1', 'Class 2', 'Class 3', 'Class 4'], rotation=45)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mat, cax=cax)
    plt.savefig('codebar.pdf', bbox_inches='tight', dpi=450)
    plt.show()

def classify_by_values(X):
    """
    Transform values to classes
    :param X:
    :return:
    """

    out_X = np.where(X < 0.5, 0, X)
    out_X = np.where(out_X > 0.5, 2, out_X)
    out_X = np.where(out_X == 0.5, 1, out_X)

    return out_X.astype('uint8')


def count_classes(labels):
    """
    Count the amount of each class for plottin'
    :param X:
    :return:
    """
    counts = [[], [], []]
    for c in (0,1,2):
        counts[c] = (list(np.sum(labels == c, axis=0)))

    counts = np.array(counts).T

    return np.array(counts)

def plot_bars(counts, title):
    """
    Just count how many of each class there is and do a lil bar graph
    """
    # Linear - Lots of Ruffles
    # Small ruffles - Big ruffles
    # Continuous - Fragmented
    # Sharp - Diffuse

    features = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
    classes = ['< 0.5', '= 0.5', '> 0.5']

    for c in range(3):
        offset = 0.20*(c-1)
        x = np.array([1,2,3,4])+offset
        plt.bar(x,counts[:,c], width=0.20, label=classes[c])

    plt.xticks((1,2,3,4), features)
    plt.ylabel('Number of crops')
    plt.xlabel('Feature')
    plt.title(title)
    plt.legend()
    plt.savefig(f'bars_{title}.pdf', bbox_inches='tight', dpi=450)
    plt.close()

def plot_feature_distribution(features):
    """
    Plot a histogram of the distribution of features' values
    :param features:
    :return:
    """

    fig, axes = plt.subplots(4,1,sharey='all' ,sharex='all')
    for f in range(features.shape[1]):
        axes[f].hist(features[:,f], bins=np.arange(0,1,0.05), edgecolor='black', linewidth=1.0)
        #axes[f].set_ylim(0,160)
    plt.xticks(np.arange(0,1,0.1))
    #plt.title(f'Class {f+1}')
    plt.xlabel('Feature value')
    plt.ylabel('Number of crops')
    #plt.show()
    plt.savefig(f'hist_features.pdf', bbox_inches='tight', dpi=450)
    plt.close()

def quantify_cluster_proportions(classes, images):
    """
    Quantify the amount of crops in each cluster for every image
    :param y: classes
    :param images: image names
    :return:
    """

    all_images = np.unique(images)
    all_classes = np.unique(classes)

    for i, img in enumerate(all_images):
        plt.figure(figsize=(2, 2))
        counts = np.array([0 for _ in all_classes])
        y = classes[images == img]

        for c in all_classes:
            counts[c] += np.sum(y == c)

        plt.bar(all_classes, counts/len(y), color=DEFAULT_COLORS, edgecolor='black', linewidth=1.0)
        plt.title(f'Image {i+1}')
        plt.ylim(0,0.5)
        plt.xlabel('Cluster')
        plt.ylabel('Proportion of crops')
        plt.xticks(all_classes, all_classes)
        #plt.show()
        plt.savefig(f'n_classes_{img}.pdf', bbox_inches='tight', dpi=450)
        plt.close()




if __name__== '__main__':
    np.random.seed(105)

    paths = ('patchlist1_32.txt', 'patchlist2.txt', 'patchlist3_ancienLog_32_nouveauLog.txt', 'patchlist4.txt')
    #paths = ('patchlist1_32.txt', 'patchlist2.txt', 'patchlist4.txt')
    feature_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4']

    features_list, coords_list, images_list, structures_list = [], [], [], []
    for p in paths:
        features, coords, images, structures = patch_list_to_points(p, filter=False, structures=(1,2))

        class_counts = count_classes(classify_by_values(features[structures == 1]))
        plot_bars(class_counts, p)

        features_list.append(features)
        coords_list.append(coords)
        images_list.append(images)
        structures_list.append(structures)

    features, coords, images, structures = combine_labels(features_list, coords_list, images_list, structures_list)

    # Show where ambiguous vs structure
    y = structures
    for img_name in np.unique(images):
        im_y = y[images == img_name]
        im_coords = coords[images == img_name]

        image = load_image(img_name)
        localize_clusters(image, im_coords, im_y, colors=['black', 'cyan', 'red'])
        plt.title('Structures')
        plt.savefig(f'structures_{img_name[:-4]}.pdf', bbox_inches='tight', dpi=450)
        plt.close()

    # Basic classification
    features, coords, images, structures = filter_points(features, coords, images, structures)
    plot_feature_distribution(features)
    X = features
    class_X = classify_by_values(X)

    for c in range(4):
        y = class_X[:, c]
        for img_name in np.unique(images):
            im_y = y[images == img_name]
            im_coords = coords[images == img_name]

            image = load_image(img_name)
            localize_clusters(image, im_coords, im_y)
            plt.title(feature_names[c])
            plt.savefig(f'{feature_names[c]}_{img_name[:-4]}.pdf', bbox_inches='tight', dpi=450)
            plt.close()


    # Clustering and UMAP
    n_clusters = 7
    X, y, knn_model = label_clustering(features, n_clusters=n_clusters)
    quantify_cluster_proportions(y, images)

    for img_name in np.unique(images):
        im_y = y[images == img_name]
        im_coords = coords[images == img_name]

        image = load_image(img_name)
        localize_clusters(image, im_coords, im_y, n_examples=5, im_name=img_name)
        plt.title('Clustering Classes')
        plt.savefig(f'clustering_{img_name[:-4]}.pdf', bbox_inches='tight', dpi=450)
        plt.close()
    umap_clusters(X, y, method='UMAP')
    quantify_clusters(X, y)


    #labels = combine_labels(labels)
    #scatter_labels(labels, paths)