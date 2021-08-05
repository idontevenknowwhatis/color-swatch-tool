import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import skimage
from sklearn.cluster import KMeans
from skimage import io
from skimage import color
from skimage import data
import cv2
import os, glob
from skimage.segmentation import felzenszwalb, quickshift, slic, chan_vese
import math
resize_size = 1024
def colorAnalyzerRGB(clusters, image):
    # print(image.shape)
    #image_resized = cv2.resize(image, (resize_size, resize_size))
    image_resized = image
    image_flat = image_resized.reshape((image_resized.shape[0] * image_resized.shape[1], 3))
    km = KMeans(n_clusters = clusters)
    km.fit(image_flat)
    colors = km.cluster_centers_.astype(int)
    labels = km.labels_

    return colors, labels


def colorAnalyzerHSV(clusters, image):
    import warnings
    warnings.filterwarnings("ignore")
    image_resized = skimage.color.rgb2hsv(image)
    #image_resized = cv2.resize(image_resized, (resize_size, resize_size))
    image_flat = image_resized.reshape((image_resized.shape[0] * image_resized.shape[1], 3))
    km = KMeans(n_clusters = clusters)
    km.fit(image_flat)
    colors = km.cluster_centers_
    labels = km.labels_
    colors = np.expand_dims(colors, axis = 0)
    colors = skimage.color.hsv2rgb(colors) * 255
    colors = np.squeeze(colors, axis = 0)
    return colors, labels  

def colorAnalyzerXYZ(clusters, image):
    image_resized = skimage.color.rgb2xyz(image)
    #image_resized = cv2.resize(image_resized, (resize_size, resize_size))
    image_flat = image_resized.reshape((image_resized.shape[0] * image_resized.shape[1], 3))
    km = KMeans(n_clusters = clusters)
    km.fit(image_flat)
    colors = km.cluster_centers_
    labels = km.labels_
    colors = np.expand_dims(colors, axis = 0)
    colors = skimage.color.xyz2rgb(colors) * 255
    colors = np.squeeze(colors, axis = 0)
    return colors, labels  

def colorAnalyzerRGBCIE(clusters, image):
    image_resized = skimage.color.rgb2rgbcie(image)
    #image_resized = cv2.resize(image_resized, (resize_size, resize_size))
    image_flat = image_resized.reshape((image_resized.shape[0] * image_resized.shape[1], 3))
    km = KMeans(n_clusters = clusters)
    km.fit(image_flat)
    colors = km.cluster_centers_
    labels = km.labels_
    colors = np.expand_dims(colors, axis = 0)
    colors = skimage.color.rgbcie2rgb(colors) * 255
    colors = np.squeeze(colors, axis = 0)
    return colors, labels  


def colorAnalyzerLAB(clusters, image):
    # print(image.shape)
    image_resized = skimage.color.rgb2lab(image)
    #image_resized = cv2.resize(image_resized, (resize_size, resize_size))
    image_flat = image_resized.reshape((image_resized.shape[0] * image_resized.shape[1], 3))
    km = KMeans(n_clusters = clusters)
    km.fit(image_flat)
    colors = km.cluster_centers_
    labels = km.labels_
    colors = np.expand_dims(colors, axis = 0)
    colors = skimage.color.lab2rgb(colors) * 255
    colors = np.squeeze(colors, axis = 0)
    return colors, labels

def recolor(image, selected, colors, labels):

    recolored_image = np.zeros(image.shape)
    loc = 0

    for i in range(recolored_image.shape[0]):
        for j in range(recolored_image.shape[1]):

            if(labels[loc] not in selected):
                R, G, B = image[i][j]
                recolored_image[i][j] = (0.2125 * R + 0.7154 * G + 0.0721 * B)/255
            else:
                recolored_image[i][j] = colors[labels[loc]]/255

            loc +=1 
    
    return recolored_image

def histogram(colors, labels, width, height):
    #https://buzzrobot.com/dominant-colors-in-an-image-using-k-means-clustering-3c7af4622036
    numLabels = np.arange(0, colors.shape[0] + 1)
    (hist, _) = np.histogram(labels, bins = numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    colors_sorted = colors[(-hist).argsort()]
    hist = hist[(-hist).argsort()]
    chart = np.zeros((height, width, 3), np.uint8)
    start = 0
    for i in range(colors.shape[0]):
        end = start + width / colors.shape[0]
                
        #getting rgb values
        r = colors_sorted[i][0].item()
        g = colors_sorted[i][1].item()
        b = colors_sorted[i][2].item()

        #using cv2.rectangle to plot colors
        cv2.rectangle(chart, (int(start), 0), (int(end), height), (r,g,b), -1)
        start = end	

    #display chart
    plt.figure()
    plt.axis("off")
    plt.imshow(chart)
    plt.show()

def histogram_brightness(colors, width, height):
    #https://buzzrobot.com/dominant-colors-in-an-image-using-k-means-clustering-3c7af4622036
    chart = np.zeros((height, width, 3), np.uint8)
    start = 0
    colors_brightnesses = []

    for i in colors:
        colors_brightnesses.append(brightness(i))
    colors_brightnesses = np.asarray(colors_brightnesses)
    colors_sorted = colors[(- colors_brightnesses).argsort()]

    for i in range(colors.shape[0]):
        end = start + width / colors.shape[0]
                
        #getting rgb values
        r = colors_sorted[i][0].item()
        g = colors_sorted[i][1].item()
        b = colors_sorted[i][2].item()

        #using cv2.rectangle to plot colors
        cv2.rectangle(chart, (int(start), 0), (int(end), height), (r,g,b), -1)
        start = end	

    #display chart
    plt.figure()
    plt.axis("off")
    plt.imshow(chart)
    plt.show()

def brightness(color):
    # http://alienryderflex.com/hsp.html
    import math
    r2 = color[0] * color[0] * 0.299
    g2 = color[1] * color[1] * 0.587
    b2 = color[2] * color[2] * 0.114

    return math.sqrt(r2 + g2 + b2)

def histogram_HSV(colors, width, height):
    #https://buzzrobot.com/dominant-colors-in-an-image-using-k-means-clustering-3c7af4622036
    chart = np.zeros((height, width, 3), np.uint8)
    start = 0
    colors_brightnesses = []
    import colorsys    
    # https://www.alanzucconi.com/2015/09/30/colour-sorting/
    colors_sorted = sorted(colors, key=lambda rgb: colorsys.rgb_to_hsv(*rgb)	)

    for i in range(colors.shape[0]):
        end = start + width / colors.shape[0]
                
        #getting rgb values
        r = colors_sorted[i][0].item()
        g = colors_sorted[i][1].item()
        b = colors_sorted[i][2].item()

        #using cv2.rectangle to plot colors
        cv2.rectangle(chart, (int(start), 0), (int(end), height), (r,g,b), -1)
        start = end	

    #display chart
    plt.figure()
    plt.axis("off")
    plt.imshow(chart)
    plt.show()

def comparison(image, clusters, width, height, mode): 

#HSV, XYZ, RGBCIE

    colors, labels = colorAnalyzerRGB(clusters, image)
    print("######################## RGB ######################## ")
    plt.axis("off")
    plt.imshow(image)

    if mode == 0:
        histogram(colors, labels, width, height)
    elif mode == 1:
        histogram_brightness(colors, width, height)
    elif mode == 2:
        histogram_HSV(colors, width, height)

    colors, labels = colorAnalyzerLAB(clusters, image)
    print("######################## LAB ######################## ")
    plt.axis("off")
    plt.imshow(image)

    if mode == 0:
        histogram(colors, labels, width, height)
    elif mode == 1:
        histogram_brightness(colors, width, height)
    elif mode == 2:
        histogram_HSV(colors, width, height)
    
    colors, labels = colorAnalyzerHSV(clusters, image)
    print("######################## HSV ######################## ")
    plt.axis("off")
    plt.imshow(image)

    if mode == 0:
        histogram(colors, labels, width, height)
    elif mode == 1:
        histogram_brightness(colors, width, height)
    elif mode == 2:
        histogram_HSV(colors, width, height)

    colors, labels = colorAnalyzerXYZ(clusters, image)
    print("######################## XYZ ######################## ")
    plt.axis("off")
    plt.imshow(image)

    if mode == 0:
        histogram(colors, labels, width, height)
    elif mode == 1:
        histogram_brightness(colors, width, height)
    elif mode == 2:
        histogram_HSV(colors, width, height)

    colors, labels = colorAnalyzerRGBCIE(clusters, image)
    print("######################## RGBCIE ######################## ")
    plt.axis("off")
    plt.imshow(image)

    if mode == 0:
        histogram(colors, labels, width, height)
    elif mode == 1:
        histogram_brightness(colors, width, height)
    elif mode == 2:
        histogram_HSV(colors, width, height)
    
def posterizationSLIC(image, cells, compactness, sigma, clusters, gaussianKernel):
    mask = slic(image, n_segments = cells, compactness = compactness, sigma = sigma, convert2lab = True)
    recolored = np.zeros(image.shape)

    for n in range(cells):
        locations = np.argwhere(mask == n)
        if(locations.shape[0] == 0):
            continue
        lab =  skimage.color.rgb2lab(image)
        colorValues = np.zeros((locations.shape[0], 3))
        i = 0
        for location in locations:
            x = location[0]
            y = location[1]
            colorValues[i] = lab[x][y]
            i += 1
        if(clusters < 0):
            count = locations.shape[0]
            newCluster = math.floor(math.pow(count, 0.25) / 5)

        else:
            newCluster = clusters
        km = KMeans(n_clusters = newCluster)
        km.fit(colorValues)
        colors = km.cluster_centers_
        labels = km.labels_
        colors = np.expand_dims(colors, axis = 0)
        colors = skimage.color.lab2rgb(colors) * 255
        colors = np.squeeze(colors, axis = 0)
        # print(locations.shape[0], newCluster)
        for i in range(locations.shape[0]):
            x = locations[i][0]
            y = locations[i][1]
            recolored[x][y] = colors[labels[i]]/255
    return recolored

def posterizationFelzen(image, scale, sigma, min_size, clusters, gaussianKernel):
    blurred = cv2.GaussianBlur(image, gaussianKernel, 0)

    mask = felzenszwalb(blurred, scale = scale, sigma =  sigma, min_size = min_size)
    recolored = np.zeros(image.shape)
    cells = np.amax(mask)
    # print(cells)
    for n in range(cells + 1):
        locations = np.argwhere(mask == n)
        # print(locations.shape[0])
        if(locations.shape[0] == 0):
            continue
        lab =  skimage.color.rgb2lab(image)
        colorValues = np.zeros((locations.shape[0], 3))
        i = 0
        for location in locations:
            x = location[0]
            y = location[1]
            colorValues[i] = lab[x][y]
            i += 1
        newCluster = clusters
        km = KMeans(n_clusters = newCluster)
        km.fit(colorValues)
        colors = km.cluster_centers_
        labels = km.labels_
        colors = np.expand_dims(colors, axis = 0)
        colors = skimage.color.lab2rgb(colors) * 255
        colors = np.squeeze(colors, axis = 0)
        
        for i in range(locations.shape[0]):
            x = locations[i][0]
            y = locations[i][1]
            recolored[x][y] = colors[labels[i]]/255
    return recolored

def posterizationQuickshift(image, ratio, kernel_size, sigma, clusters ):
    mask = quickshift(image, ratio = ratio, kernel_size = kernel_size, sigma = sigma,convert2lab = True)
    recolored = np.zeros(image.shape)
    cells = np.amax(mask)
    # print("Total clusters:", cells)
    for n in range(cells + 1):
        locations = np.argwhere(mask == n)
        if(locations.shape[0] == 0):
            continue
        lab =  skimage.color.rgb2lab(image)
        colorValues = np.zeros((locations.shape[0], 3))
        i = 0
        for location in locations:
            x = location[0]
            y = location[1]
            colorValues[i] = lab[x][y]
            i += 1
        newCluster = clusters
        km = KMeans(n_clusters = newCluster)
        km.fit(colorValues)
        colors = km.cluster_centers_
        labels = km.labels_
        colors = np.expand_dims(colors, axis = 0)
        colors = skimage.color.lab2rgb(colors) * 255
        colors = np.squeeze(colors, axis = 0)
        # print(locations.shape[0], newCluster)
        for i in range(locations.shape[0]):
            x = locations[i][0]
            y = locations[i][1]
            recolored[x][y] = colors[labels[i]]/255
    return recolored

def posterizationChan(image, roundness, lambda1, lambda2, clusters):
    gray = skimage.color.rgb2gray(image)
    mask = chan_vese(gray, mu = roundness, lambda1 = lambda1, lambda2 = lambda2)
    recolored = np.zeros(image.shape)
    mask = mask.astype(int)
    for n in range(2):
        locations = np.argwhere(mask == n)
        if(locations.shape[0] == 0):
            continue
        lab =  skimage.color.rgb2lab(image)
        colorValues = np.zeros((locations.shape[0], 3))
        i = 0
        for location in locations:
            x = location[0]
            y = location[1]
            colorValues[i] = lab[x][y]
            i += 1
        newCluster = clusters[n]
        km = KMeans(n_clusters = newCluster)
        km.fit(colorValues)
        colors = km.cluster_centers_
        labels = km.labels_
        colors = np.expand_dims(colors, axis = 0)
        colors = skimage.color.lab2rgb(colors) * 255
        colors = np.squeeze(colors, axis = 0)
        # print(locations.shape[0], newCluster)
        for i in range(locations.shape[0]):
            x = locations[i][0]
            y = locations[i][1]
            recolored[x][y] = colors[labels[i]]/255
    return recolored