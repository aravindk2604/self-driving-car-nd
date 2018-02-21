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
color_select = np.copy(image)
line_image = np.copy(image)

# Define color selection criteria

red_threshold = 200
green_threshold = 200
blue_threshold = 190

rgb_threshold = [red_threshold, green_threshold, blue_threshold]


# Define a triangular region of interest
left_bottom = [0, 390]
right_bottom = [600, 390]
apex = [300, 150]

# connect these points using a straight line formula

fit_left = np.polyfit((left_bottom[0], apex[0]),(left_bottom[1], apex[1]),1 )
fit_right = np.polyfit((right_bottom[0], apex[0]),(right_bottom[1], apex[1]),1 )
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]),(left_bottom[1], right_bottom[1]),1 )


# Mask the pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) \
			     | (image[:,:,1] < rgb_threshold[1]) \
			     | (image[:,:,2] < rgb_threshold[2]) 



# creating meshgrid
XX, YY = np.meshgrid(np.arange(0,xsize),np.arange(0,ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
					(YY > (XX*fit_right[0] + fit_right[1])) & \
					(YY < (XX*fit_bottom[0] + fit_bottom[1]))


# mask color selection
color_select[color_thresholds] = [0,0,0]

# FInd where image is both colored right and in the region
line_image[~color_thresholds & region_thresholds] = [255,0,0]


#Display the image
plt.imshow(color_select)
plt.imshow(line_image)
plt.show()

#mpimg.imsave("region-mask-lanes.jpg", region_select)