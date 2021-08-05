# color-swatch-tool

Course project for Computational Aesthetics in 2018. Color clustering is usually done by finding a preset number of clusters in RGB space using some method such as K-means. The first avenue of investigation is whether transforming the input RGB space to different color spaces, such as the perceptual CIERGB or LAB spaces, have any positive effects on the quality of colors produced. The human eye does not perceive red, green and blue equally as we have a heightened sensitivity to green for example. Therefore, the aforementioned perceptual color spaces that were designed to accomodate human vision is hypothesized to work better. Experiments were done by interviewing fellow students and showing them the original image, then an image in which each pixel takes on the color value of the nearest cluster in some color space. The second image is reconstruction of the original with a vastly reduced number of colors. The reconstructed image that the student believes is most accurate is considered to be an example of good performance from the color space used for clustering. It was found that XYZ, CIERGB and LAB spaces perform the best. The second avenue of investigation is using image segmentation to discretize an input image into segments which will then be colored using an appropriate color, as defined by the color clusters for that image. Of interest is whether any image segmentation techniques such as Felzenswab or Chan-vese produce particularly aesthetically pleasing results. Results were not consistent - there was no segmentation technique that was clearly superior.
