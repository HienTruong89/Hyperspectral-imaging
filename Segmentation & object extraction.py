import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data
from skimage.feature import peak_local_max
from skimage import filters
from skimage import io
import os
import spectral.io.envi as envi
import matplotlib.patches as mpatches
from skimage.measure import label, regionprops
from scipy.ndimage import find_objects
from skimage import morphology
from scipy import ndimage as ndi
import pandas as pd
from skimage.color import label2rgb
import glob
import fnmatch

from skimage.morphology import dilation
from skimage.morphology import erosion
from skimage.morphology import binary_erosion
from skimage.morphology import binary_opening
from skimage.morphology import binary_closing
from skimage.morphology import disk, dilation, binary_erosion
import matplotlib.pyplot as plt
# PART 1: Correct dark and white balance & loading images
os.chdir("C:\\Users\\")
# Read dark image
d=envi.open('C:/Users/')
Dark = d.load().astype("float64")
# calcualate the avergae of intensity of 350 number of lines in hypercubes
Ones=np.ones([350,1]) # create the array for dark 
D=np.mean(Dark,axis=0)
Dhtarray=np.reshape(np.tensordot(Ones,D,axes=0),[350,320,235]) # reshape the dark reference

# Read white folder
w=envi.open('C:/Users/')
white=w.load().astype("float64")
# calcualate the avergae of intensity of 350 number of lines in hypercubes
Ones=np.ones([350,1])  # create the array for white
W=np.mean(white,axis=0)
Whtarray=np.reshape(np.tensordot(Ones,W,axes=0),[350,320,235]) # reshape the white reference

# Read images containing in samples file
image1 =envi.open('C:/Users/')
img1 = image1.load().astype("float64")
plt.imshow(img1[:,:,150])

# correction white and dark balance
Aaux=(img1-Dhtarray)
plt.imshow(Aaux[:,:,150]) # checking the Aaux
Baux=(Whtarray-Dhtarray)
plt.imshow(Baux[:,:,150]) # checking the Baux

# create the loop for the whole wavelength 
tray1=np.zeros([350,320,235])
for i in range(0,234):
        tray1[:,:,i]=Aaux[:,:,i]/Baux[:,:,i]
# checking the plot at 1 wavelength 
plt.imshow(Aaux[:,:,150]/Baux[:,:,150]) 
# or imshow like this
trayx=tray1[:,:,150]
plt.imshow(trayx)
# save the image        
plt.imsave('C:/Users/)
# PART 2: SEGMENTATION/SEGREGATION
# Thresholding for removing the background and keep honey
bina_img= tray1[:,:,32]<0.4
plt.imshow(bina_img)
# thresholding removing the straight line next to the image
bina_img2= tray1[:,:,154]<0.15
plt.imshow(bina_img2)
bina_img3= bina_img * bina_img2
plt.imshow(bina_img3)
# Thresholding for removing the shadow
bina_img4= tray1[:,:,52]>0.26
plt.imshow(bina_img4)
plt.show()
bina_img5 = bina_img3 * bina_img4
plt.imshow(bina_img5)
#checking after removing background
bina_img6=tray1[:,:,1] * bina_img5
plt.imshow(bina_img6)
# Remove small objects
img_clean = morphology.remove_small_objects(bina_img5,900)
plt.imshow(img_clean)
# Opening the image to remove small objects
pro=binary_opening(img_clean,selem=None, out=None)
plt.imshow(pro)
# Closing the image to remove small holes
cro=binary_closing(pro,selem=None, out=None)
plt.imshow(cro)
# Remove small objects second time
img_clean2 = morphology.remove_small_objects(cro,900)
plt.imshow(img_clean2)
plt.show ()
# final masking 
bina_img7=bina_img5 *img_clean2
plt.imshow(bina_img7)
bina_img7x=bina_img7*tray1[:,:,57]
plt.imshow(bina_img7x)
# stacking images of the whole tray 
list=[]
for i in range(235):
    bina_img8 =tray1[:,:,i]
    list.append(bina_img8)
bina_img8=np.stack(list,axis=2)    
plt.imshow(bina_img8[:,:,57])
# Compute connected regions in the image; we're going to use this
from skimage import io, img_as_bool, measure, morphology
from skimage.segmentation import clear_border
# remove artifacts connected to image border
cleared = clear_border(bina_img7)
labels = measure.label(bina_img7,background = 0)
# measure objects and turn out the object size
props = measure.regionprops(labels)
fig,ax2=plt.subplots()
ax2.imshow(labels)
props[0]['Centroid'] # centroid of first labelled object
a=props[0].centroid
print(a)
for prop in props:
    print('Label: {} >> Object centroid:{}'.format (prop.label,prop.centroid))
# on the labeled image labels, regionprops is performed
fig,ax3=plt.subplots()
ax3.imshow(labels)#plot the labeled image on the previous plot
# Draw rectangle around honey regions. 
for prop in props: 
   minr, minc, maxr, maxc = prop.bbox 
   rect = mpatches.Rectangle((minc, minr),
                              maxc - minc, 
                              maxr - minr, 
                              fill=False, 
                              edgecolor='red', 
                              linewidth=0.5)
   ax3.add_patch(rect)
for i in props:
    print (i['Centroid'][1],i['Centroid'][0]) # printing the x and y values of the # centroid where centroid[1] is the x value # and centroid[0] is the y value print i['Centroid'][1],i['Centroid'][0] # plot a red circle at the centroid, ro stands # for red
    plt.plot(i['Centroid'][1],i['Centroid'][0],'+')
plt.tight_layout()
#save individual objects in png and csv 
objects = ndi.find_objects(labels) # from the object slices, i can see the position of objects
slice_x,slice_y=objects[0]
roi=bina_img7x[slice_x,slice_y]
plt.imshow(roi)
plt.imsave('C:/Users/) 
roi_check=roi*tray1[43:68,21:74,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
#Stack images for each object to 235 bands
_roi=[]
for i in range(235):
    _roi2=roi*tray1[43:68,21:74,i]
    _roi.append(_roi2)
_roi3=np.stack(_roi,axis=2)
_roi3.shape # checking the dimension
#save _roi3 to csv
from numpy import savetxt
import numpy as np
import pandas as pd 
x=np.resize(_roi3,[_roi3.shape[0]*_roi3.shape[1],_roi3.shape[2]])
savetxt('C:/Users/',x,delimiter=',')
# continue for each object in a tray

