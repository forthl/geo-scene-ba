import numpy as np
import matplotlib.pyplot as plt

def plotGraph(image, title = None):
    arr = np.asarray(image)
    plt.imshow(arr)
    if title is not None:
        plt.title(title)
    plt.show()
    
def pointcloudOnImage(point_cloud, image, title='PC on Image', dot_size=5):
    # Init axes.
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig.canvas.set_window_title(title)
    ax.imshow(image)
    ax.scatter(point_cloud[0, :], point_cloud[1, :], c=point_cloud[2, :], s=dot_size)
    ax.axis('off')
    
    plt.show()
