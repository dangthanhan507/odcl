import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import namedtuple
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from itertools import cycle
from collections import defaultdict
"""
This program uses sklearn to implment meansshift to segment images
"""


"""
function to recreate image with new rgb values
"""

def sort_masks(mask_list):

  for idx, mask in enumerate(mask_list):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_LAB2RGB)
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    cv2.imshow("gray", gray_mask)

    _,binary = cv2.threshold(gray_mask,100,255,cv2.THRESH_BINARY)
    #cnts,hierarchy = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    print(contours)
    print(type(contours))

    print("============================")
    cnt = contours[1]
    if (len(cnt) == 0):
      print(len(cnt))

    #area_list = [cv2.contourArea(cnt) for cnt in contours]
    if idx != 0:
      cv2.drawContours(rgb_mask, cnt, -1, color=(0, 255, 0), thickness=2)


    cv2.imshow("CONTOURS", rgb_mask)
    cv2.waitKey(0)
  
  
  

  


def get_segmentations(flat_img, labels, shape, centers):

  mask_dict = defaultdict(list)
  print("CREATING SEGMENTED")
  print("CENTERS ARRAY")
  print(centers)

  for i in range(len(centers)):
    
    for idx,pixel in enumerate(flat_img):
      cluster = labels[idx]
      if cluster == i:
        print("IN CLUSTER")
        print(cluster)
        avg = centers[cluster]
      else:
        avg = np.array([0, 0, 0])

      mask_dict[i].extend([avg])

  for key, value in mask_dict.items():
    mask_dict[key] = np.uint8(value)

  return mask_dict
    


def plot_LAB(image_LAB, image_BGR):

  y,x,z = image_LAB.shape
  LAB_flat = np.reshape(image_LAB, [y*x,z])

  colors = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
  colors = np.reshape(colors, [y*x,z])/255.

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(x=LAB_flat[:,2], y=LAB_flat[:,1], s=10,  c=colors, lw=0)
  ax.set_xlabel('A')
  ax.set_ylabel('B')

  plt.show()


if __name__ == "__main__":
    """
    test image is the image in path directory

    edit devmode to true to include relevant print messages
    """

    devmode = True   
    path = r'Cropped_Detections\0.jpeg'
    img = cv2.imread(path)
    lab_org = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    if devmode:
        print("PRINTING LAB IMAGE'S ARRAY")
        print(lab_org)
        print(lab_org.shape)

    L,A,B = cv2.split(lab_org)

    if devmode:
        print("AFTER DSPLIT")
        print("A")
        print("=======================")
        print(A)
        print("=======================")
        print("B")
        print("=======================")
        print(B)

        cv2.imshow("A", A)
        cv2.imshow("B", B)

    y,x,z = lab_org.shape
    LAB_flat = np.reshape(lab_org, [y*x,z])

    if devmode:
        print("AB COLOR SPACE")
        print("SHAPE")
        print(LAB_flat.shape)
        print("====================")
        print(LAB_flat)
        cv2.waitKey(0)

        #plot A and B values with mat plot lib
        plot_LAB(lab_org, img)

    bandwidth = estimate_bandwidth(LAB_flat, quantile=0.2, n_jobs=-1)

    print("BANDWIDTH")
    print(bandwidth)
    
    ms = MeanShift(bandwidth=bandwidth,
                   bin_seeding=True,
                   cluster_all=True,
                   n_jobs=-1,
                   min_bin_freq = 400)
    
    ms.fit_predict(LAB_flat)

    labels = ms.labels_
    cluster_centers = np.array(ms.cluster_centers_).astype(int)
    
    labels_unique = np.unique(labels)
    n_clusters = len(np.unique(labels))

    plt.figure(1)
    plt.clf()

    mask_dict = get_segmentations(LAB_flat, labels, lab_org.shape, cluster_centers)

    print(mask_dict.keys())

    for key, val in mask_dict.items():
      print("=======================================")
      print()
      print(mask_dict)
    
    masks = [mask for mask in [np.reshape(val, lab_org.shape) for _, val in mask_dict.items()]]

    
    if devmode:
      print("SHAPE OF SEGMENTED IMAGE")
      print("=========================")
      print(img.shape)
      print("=========================")
      print("ARRAY FOR SEG_IMG")
      print(img)
      print("=========================")
      print("CLUSTER CENTERS")
      print(cluster_centers)

    """
    converting segmented LAB image to BGR for display
    """
    
    #cluster1 = cv2.cvtColor(masks[0], cv2.COLOR_LAB2BGR)

    if devmode:
      print("=======================")
      for mask in masks:
        cv2.imshow("MASK", mask)
        cv2.waitKey(0)

    sort_masks(masks)

    
    
    
    
