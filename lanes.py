import cv2
import numpy as np

def average_slope_intercept(image, lines):
	left_fit = [] # Left line
	right_fit = [] # Right line
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		parameters = np.polyfit((x1,x2), (y1,y2), 1)
		# 1 is the degree of the polynomial created
		# which is linear
		# parameters variable output as (slope, intercept)
		
		slope = parameters[0]
		intercept = parameters[1]

		if slope < 0:
			left_fit.append((slope, intercept))
		else:
			right_fit.append((slope, intercept))
	left_fit_average = np.average((left_fit), axis=0)
	right_fit_average = np.average((right_fit), axis=0)
	left_line = make_coordinates(image, left_fit_average)
	right_line = make_coordinates(image, right_fit_average)
	return np.array([left_line, right_line])

def make_coordinates(image, line_parameters):
	slope, intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(y1 * (3/5))
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	return np.array([x1,y1,x2,y2])

def canny(image):
	'''
	This function reduces the noises in the image for edge detection
	'''
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(image, (5,5), 0) # (img , kernel_size, sigma_X)
	canny = cv2.Canny(image,50,150) # (img, low_threshold, high_threshold)
	return canny

def region_of_interest(image):
	height = image.shape[0]
	triangle = np.array([
		[(200,height), 
		(1100, height), 
		(550, 250)
		]])
	mask = np.zeros_like(image) # An image of same dimension but having each pixel value = 0
	cv2.fillPoly(mask, triangle, 255) # Filling the triangle with white color
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def display_lines(image, lines): 
	line_image = np.zeros_like(image)
	if lines is not None:
		for x1,y1,x2,y2 in lines:
			cv2.line(line_image, (x1,y1),(x2,y2), (255,0,0), 10) # 10 is line thickness
	return line_image

image = cv2.imread('./image/test_image.jpg')
lane_image = np.copy(image)

cap = cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
	_,frame =cap.read()  # Returns a boolean value and a frame
	canny_image = canny(frame)
	cropped_image = region_of_interest(canny_image)
	#                      (img,pixel_precision,radian_precision,threshold, ...)
	lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
	averaged_lines = average_slope_intercept(frame, lines) 
	line_image = display_lines(frame, averaged_lines)
	combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) 
	'''
	0.8 is the weight value multiplied with each pixel in lane_imgae array.
	Similar with line_image where w value is 1
	And the last parameter(1) is the bias added to each pixel
	'''
	cv2.imshow("result", combo_image)
	if cv2.waitKey(1) == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()


########<-- NOTES -->#########

'''
Gaussian Blurring(Optional when applying Canny function to the image)

Image noise is the random variation in the brightness or color of image in a pixel.
Image noise reduction helps in improving the quality of an image.

Options other than GaussianBlur are - Median blurring and bilateral blurring. 
'''

'''
Hough Space

To detect straight lines in our image we use Hough alogorithm.
'''