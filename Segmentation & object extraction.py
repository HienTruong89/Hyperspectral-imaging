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
os.chdir("C:\\Users\\ttruong\\OneDrive - Massey University\\Data 2020& python code\\2000 samples -11-11-2020\\Hien 11-11-2020_2000 honey samples\\Line-scanning-11-11-2020\\")
# Read dark image
d=envi.open('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Line-scanning-11-11-2020/dark_2020_11_11_09_36_42/raw_0.hdr')
Dark = d.load().astype("float64")
# calcualate the avergae of intensity of 350 number of lines in hypercubes
Ones=np.ones([350,1]) # create the array for dark 
D=np.mean(Dark,axis=0)
Dhtarray=np.reshape(np.tensordot(Ones,D,axes=0),[350,320,235]) # reshape the dark reference

# Read white folder
w=envi.open('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Line-scanning-11-11-2020/white 1_2020_11_11_15_33_25/raw_0.hdr')
white=w.load().astype("float64")
# calcualate the avergae of intensity of 350 number of lines in hypercubes
Ones=np.ones([350,1])  # create the array for white
W=np.mean(white,axis=0)
Whtarray=np.reshape(np.tensordot(Ones,W,axes=0),[350,320,235]) # reshape the white reference

# Read images containing in samples file
os.chdir("C:\\Users\\ttruong\\OneDrive - Massey University\\Data 2020& python code\\2000 samples -11-11-2020\\Hien 11-11-2020_2000 honey samples\\Line-scanning-11-11-2020\\samples\\")   
image1 =envi.open('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Line-scanning-11-11-2020/samples/Tray 6_2020_11_11_11_40_39/raw_0.hdr')
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
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/tray6.png',trayx)
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
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_19.png',roi) 
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
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_19.csv',x,delimiter=',')
# continue for each object in a tray
# tray 1 @ object 2
slice_x,slice_y=objects[1]
roi_c12=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c12)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_13.png',roi_c12) 
roi_check=roi_c12*tray1[45:71,81:138,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c12=[]
for i in range(235):
    _c12x=roi_c12*tray1[45:71,81:138,i]
    _c12.append(_c12x)
_c12xx=np.stack(_c12,axis=2)
_c12xx.shape 
y=np.resize(_c12xx,[_c12xx.shape[0]*_c12xx.shape[1],_c12xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_13.csv',y,delimiter=',')
# tray 1  @ object 3
slice_x,slice_y=objects[2]
roi_c13=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c13)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_7.png',roi_c13) 
roi_check=roi_c13*tray1[47:73,146:203,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c13=[]
for i in range(235):
    _c13x=roi_c13*tray1[47:73,146:203,i]
    _c13.append(_c13x)
_c13xx=np.stack(_c13,axis=2)
_c13xx.shape 
y=np.resize(_c13xx,[_c13xx.shape[0]*_c13xx.shape[1],_c13xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_7.csv',y,delimiter=',')
 # tray 1 @ object 4
slice_x,slice_y=objects[3]
roi_c21=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c21)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_1.png',roi_c21) 
roi_check=roi_c21*tray1[48:73,211:264,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c21=[]
for i in range(235):
    _c21x=roi_c21*tray1[48:73,211:264,i]
    _c21.append(_c21x)
_c21xx=np.stack(_c21,axis=2)
_c21xx.shape 
y=np.resize(_c21xx,[_c21xx.shape[0]*_c21xx.shape[1],_c21xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_1.csv',y,delimiter=',')

 # tray 1 @ object 5
slice_x,slice_y=objects[4]
roi_c22=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c22)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_20.png',roi_c22) 
roi_check=roi_c22*tray1[72:98,21:77,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c22=[]
for i in range(235):
    _c22x=roi_c22*tray1[72:98,21:77,i]
    _c22.append(_c22x)
_c22xx=np.stack(_c22,axis=2)
_c22xx.shape 
y=np.resize(_c22xx,[_c22xx.shape[0]*_c22xx.shape[1],_c22xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_20.csv',y,delimiter=',')
 # tray 1 @ object 6
slice_x,slice_y=objects[5]
roi_c23=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c23)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_14.png',roi_c23) 
roi_check=roi_c23*tray1[74:99,79:135,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c23=[]
for i in range(235):
    _c23x=roi_c23*tray1[74:99,79:135,i]
    _c23.append(_c23x)
_c23xx=np.stack(_c23,axis=2)
_c23xx.shape 
y=np.resize(_c23xx,[_c23xx.shape[0]*_c23xx.shape[1],_c23xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_14.csv',y,delimiter=',')
# tray 1 @ object 7
slice_x,slice_y=objects[6]
roi_c31=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c31)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_8.png',roi_c31) 
roi_check=roi_c31*tray1[76:103,147:204,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c31=[]
for i in range(235):
    _c31x=roi_c31*tray1[76:103,147:204,i]
    _c31.append(_c31x)
_c31xx=np.stack(_c31,axis=2)
_c31xx.shape 
y=np.resize(_c31xx,[_c31xx.shape[0]*_c31xx.shape[1],_c31xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_8.csv',y,delimiter=',')
# tray 1 @ object 8
slice_x,slice_y=objects[7]
roi_c32=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c32)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_2.png',roi_c32) 
roi_check=roi_c32*tray1[78:103,210:267,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c32=[]
for i in range(235):
    _c32x=roi_c32*tray1[78:103,210:267,i]
    _c32.append(_c32x)
_c32xx=np.stack(_c32,axis=2)
_c32xx.shape 
y=np.resize(_c32xx,[_c32xx.shape[0]*_c32xx.shape[1],_c32xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_2.csv',y,delimiter=',')
# tray 1 @ object 9
slice_x,slice_y=objects[8]
roi_c33=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c33)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_21.png',roi_c33) 
roi_check=roi_c33*tray1[102:128,17:73,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c33=[]
for i in range(235):
    _c33x=roi_c33*tray1[102:128,17:73,i]
    _c33.append(_c33x)
_c33xx=np.stack(_c33,axis=2)
_c33xx.shape 
y=np.resize(_c33xx,[_c33xx.shape[0]*_c33xx.shape[1],_c33xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_21.csv',y,delimiter=',')
# tray 1 @ object 10
slice_x,slice_y=objects[9]
roi_c41=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c41)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_15.png',roi_c41) 
roi_check=roi_c41*tray1[103:130,81:138,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c41=[]
for i in range(235):
    _c41x=roi_c41*tray1[103:130,81:138,i]
    _c41.append(_c41x)
_c41xx=np.stack(_c41,axis=2)
_c41xx.shape 
y=np.resize(_c41xx,[_c41xx.shape[0]*_c41xx.shape[1],_c41xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_15.csv',y,delimiter=',')
# tray 1 @ object 11
slice_x,slice_y=objects[10]
roi_c42=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c42)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_9.png',roi_c42) 
roi_check=roi_c42*tray1[106:132,146:203,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c42=[]
for i in range(235):
    _c42x=roi_c42*tray1[106:132,146:203,i]
    _c42.append(_c42x)
_c42xx=np.stack(_c42,axis=2)
_c42xx.shape 
y=np.resize(_c42xx,[_c42xx.shape[0]*_c42xx.shape[1],_c42xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_9.csv',y,delimiter=',')
#tray 1 @ object 12
slice_x,slice_y=objects[11]
roi_c43=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c43)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_3.png',roi_c43) 
roi_check=roi_c43*tray1[107:133,211:262,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c43=[]
for i in range(235):
    _c43x=roi_c43*tray1[107:133,211:262,i]
    _c43.append(_c43x)
_c43xx=np.stack(_c43,axis=2)
_c43xx.shape 
y=np.resize(_c43xx,[_c43xx.shape[0]*_c43xx.shape[1],_c43xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_3.csv',y,delimiter=',')
#tray 1 @ object 13
slice_x,slice_y=objects[12]
roi_c51=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c51)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_22.png',roi_c51) 
roi_check=roi_c51*tray1[132:158,23:80,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c51=[]
for i in range(235):
    _c51x=roi_c51*tray1[132:158,23:80,i]
    _c51.append(_c51x)
_c51xx=np.stack(_c51,axis=2)
_c51xx.shape 
y=np.resize(_c51xx,[_c51xx.shape[0]*_c51xx.shape[1],_c51xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_22.csv',y,delimiter=',')
#tray 1 @ object 14
slice_x,slice_y=objects[13]
roi_c52=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c52)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_16.png',roi_c52) 
roi_check=roi_c52*tray1[133:160,83:142,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c52=[]
for i in range(235):
    _c52x=roi_c52*tray1[133:160,83:142,i]
    _c52.append(_c52x)
_c52xx=np.stack(_c52,axis=2)
_c52xx.shape 
y=np.resize(_c52xx,[_c52xx.shape[0]*_c52xx.shape[1],_c52xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_16.csv',y,delimiter=',')
#tray 1 @ object 15
slice_x,slice_y=objects[14]
roi_c53=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c53)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_10.png',roi_c53) 
roi_check=roi_c53*tray1[136:162,148:205,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c53=[]
for i in range(235):
    _c53x=roi_c53*tray1[136:162,148:205,i]
    _c53.append(_c53x)
_c53xx=np.stack(_c53,axis=2)
_c53xx.shape 
y=np.resize(_c53xx,[_c53xx.shape[0]*_c53xx.shape[1],_c53xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_10.csv',y,delimiter=',')
# tray 1 @ object 16
slice_x,slice_y=objects[15]
roi_c61=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c61)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_4.png',roi_c61) 
roi_check=roi_c61*tray1[137:163,213:265,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c61=[]
for i in range(235):
    _c61x=roi_c61*tray1[137:163,213:265,i]
    _c61.append(_c61x)
_c61xx=np.stack(_c61,axis=2)
_c61xx.shape 
y=np.resize(_c61xx,[_c61xx.shape[0]*_c61xx.shape[1],_c61xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_4.csv',y,delimiter=',')
# tray 1 @ object 17
slice_x,slice_y=objects[16]
roi_c62=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c62)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_23.png',roi_c62) 
roi_check=roi_c62*tray1[162:188,18:71,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c62=[]
for i in range(235):
    _c62x=roi_c62*tray1[162:188,18:71,i]
    _c62.append(_c62x)
_c62xx=np.stack(_c62,axis=2)
_c62xx.shape 
y=np.resize(_c62xx,[_c62xx.shape[0]*_c62xx.shape[1],_c62xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_23.csv',y,delimiter=',')
# tray 1 @ object 18
slice_x,slice_y=objects[17]
roi_c63=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c63)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_17.png',roi_c63) 
roi_check=roi_c63*tray1[163:189,77:136,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c63=[]
for i in range(235):
    _c63x=roi_c63*tray1[163:189,77:136,i]
    _c63.append(_c63x)
_c63xx=np.stack(_c63,axis=2)
_c63xx.shape 
y=np.resize(_c63xx,[_c63xx.shape[0]*_c63xx.shape[1],_c63xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_17.csv',y,delimiter=',')
# tray 1 @ object 19
slice_x,slice_y=objects[18]
roi_c71=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c71)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_11.png',roi_c71) 
roi_check=roi_c71*tray1[166:192,142:203,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c71=[]
for i in range(235):
    _c71x=roi_c71*tray1[166:192,142:203,i]
    _c71.append(_c71x)
_c71xx=np.stack(_c71,axis=2)
_c71xx.shape 
y=np.resize(_c71xx,[_c71xx.shape[0]*_c71xx.shape[1],_c71xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_11.csv',y,delimiter=',')
# tray 1 @ object 20
slice_x,slice_y=objects[19]
roi_c72=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c72)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_5.png',roi_c72) 
roi_check=roi_c72*tray1[166:192,204:262,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c72=[]
for i in range(235):
    _c72x=roi_c72*tray1[166:192,204:262,i]
    _c72.append(_c72x)
_c72xx=np.stack(_c72,axis=2)
_c72xx.shape 
y=np.resize(_c72xx,[_c72xx.shape[0]*_c72xx.shape[1],_c72xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_5.csv',y,delimiter=',')
# tray 1 @ object 21
slice_x,slice_y=objects[20]
roi_c73=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c73)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_24.png',roi_c73) 
roi_check=roi_c73*tray1[192:215,27:83,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c73=[]
for i in range(235):
    _c73x=roi_c73*tray1[192:215,27:83,i]
    _c73.append(_c73x)
_c73xx=np.stack(_c73,axis=2)
_c73xx.shape 
y=np.resize(_c73xx,[_c73xx.shape[0]*_c73xx.shape[1],_c73xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_24.csv',y,delimiter=',')
# tray 1 @ object 22
slice_x,slice_y=objects[21]
roi_c81=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c81)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_18.png',roi_c81) 
roi_check=roi_c81*tray1[194:220,88:145,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c81=[]
for i in range(235):
    _c81x=roi_c81*tray1[194:220,88:145,i]
    _c81.append(_c81x)
_c81xx=np.stack(_c81,axis=2)
_c81xx.shape 
y=np.resize(_c81xx,[_c81xx.shape[0]*_c81xx.shape[1],_c81xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_18.csv',y,delimiter=',')
# tray 1 @ object 23
slice_x,slice_y=objects[22]
roi_c82=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c82)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_12.png',roi_c82) 
roi_check=roi_c82*tray1[195:221,153:209,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c82=[]
for i in range(235):
    _c82x=roi_c82*tray1[195:221,153:209,i]
    _c82.append(_c82x)
_c82xx=np.stack(_c82,axis=2)
_c82xx.shape 
y=np.resize(_c82xx,[_c82xx.shape[0]*_c82xx.shape[1],_c82xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_12.csv',y,delimiter=',')
# tray 1 @ object 24
slice_x,slice_y=objects[23]
roi_c83=bina_img7x[slice_x,slice_y]
plt.imshow(roi_c83)
plt.imsave('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_6.png',roi_c83) 
roi_check=roi_c83*tray1[195:221,217:273,57] # checking and combine from slice position of each object
plt.imshow(roi_check)
_c83=[]
for i in range(235):
    _c83x=roi_c83*tray1[195:221,217:273,i]
    _c83.append(_c83x)
_c83xx=np.stack(_c83,axis=2)
_c83xx.shape 
y=np.resize(_c83xx,[_c83xx.shape[0]*_c83xx.shape[1],_c83xx.shape[2]])
savetxt('C:/Users/ttruong/OneDrive - Massey University/Data 2020& python code/2000 samples -11-11-2020/Hien 11-11-2020_2000 honey samples/Segmentation_11_11_20/Tray 6/s_6.csv',y,delimiter=',')
