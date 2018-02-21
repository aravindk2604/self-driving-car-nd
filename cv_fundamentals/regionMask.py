import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print out some stats
image = mpimg.imread('test.jpg')
#print('This image is: ', type(image), 'with dimensions: ', image.shape)

# Grab the x and y size and make a copy of the image 
ysize = image.shape[0]
xsize = image.shape[1]
print xsize, ysize
# Note: always make a copy rather than simply using "="
region_select = np.copy(image)

# Define a triangular region of interest
left_bottom = [0, 390]
right_bottom = [600, 390]
apex = [300, 150]

# connect these points using a straight line formula

fit_left = np.polyfit((left_bottom[0], apex[0]),(left_bottom[1], apex[1]),1 )
fit_right = np.polyfit((right_bottom[0], apex[0]),(right_bottom[1], apex[1]),1 )
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]),(left_bottom[1], right_bottom[1]),1 )

# creating meshgrid
XX, YY = np.meshgrid(np.arange(0,xsize),np.arange(0,ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
					(YY > (XX*fit_right[0] + fit_right[1])) & \
					(YY < (XX*fit_bottom[0] + fit_bottom[1]))

# make the region of interest red in color
region_select[region_thresholds] = [255,0,0]

#Display the image
plt.imshow(region_select)
plt.show()

mpimg.imsave("region-mask.jpg", region_select)