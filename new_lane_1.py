import cv2
import numpy as np 
import matplotlib.pyplot as plt

def make_coordinate(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1,x2, y2])

def avg_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)

    left_line = make_coordinate(image, left_fit_avg)
    right_line = make_coordinate(image, right_fit_avg)
    return np.array([left_line, right_line])

def canny_cvt(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 2.0, 2.0, 0)
    canny = cv2.Canny(blur, 5,8)
    return canny

def n_canny_cvt(image):
    n_canny_img = cv2.Canny(image, 200, 255)
    return n_canny_img

def roi(image):
    height = image.shape[0]
    width =  image.shape[1]
    poly = np.array([
        [(10,height-1), (width-1, height-1), (int(width*0.9),int(height*0.2)) , (int(width*0.15),int(height*0.2))]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, poly, 255)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img

def display_line(image, lines):
    line_image = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 1)
    return line_image

def thresh(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # ret, t_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    blur = cv2.GaussianBlur(gray, (7,7), 2.0, 0, 2.0)
    ret, t_img = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return t_img

def prespective_trans(image):
    height = image.shape[0]
    width =  image.shape[1]
    poi = np.float32([[0, int(height*(3/4))],
                    [width-1, int(height*(3/4))],
                    [0, height-1],
                    [width-1, height-1],])
    proj = np.float32([[0, 0],
                     [width-1, 0],
                     [0,height-1],
                     [width-1, height-1]])
    m = cv2.getPerspectiveTransform(poi, proj)
    w_img = cv2.warpPerspective(image, m, (width, height))
    return w_img

count = 0
def pipeline(binaryImg, image):
    if count ==0:
        w_img = np.copy(binaryImg)
        # After creating a binary Image,
        # Take a histogram of the bottom half of the image
        hist = np.sum(binaryImg[int(binaryImg.shape[0]/2):,:],axis=0)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binaryImg,binaryImg, binaryImg))*255
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        mid_point = np.int64(hist.shape[0]/2)
        leftx_base = np.argmax(hist[:mid_point])
        rightx_base = np.argmax(hist[mid_point:]) + mid_point
        
        # Choose the number of sliding windows
        nWindows = 9
        
        # Set height of windows
        window_h = np.int64(binaryImg.shape[0]/nWindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binaryImg.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set the width of the windows +/- margin
        margin = 100

        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nWindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binaryImg.shape[0] - (window+1)*window_h
            win_y_high = binaryImg.shape[0] - window*window_h
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy>= win_y_low)& (nonzeroy< win_y_high) & (nonzerox>= win_xleft_low) & (nonzerox< win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy>= win_y_low)& (nonzeroy< win_y_high) & (nonzerox>= win_xright_low) & (nonzerox< win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))
            
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binaryImg.shape[0]-1, binaryImg.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        warp_zero = np.zeros_like(binaryImg).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        # newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        
        # Combine the result with the original image
        # result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

        # y_eval=700
        # mid_x=640
        # ym_per_pix=3.0/72.0
        # xm_per_pix=3.7/650.0 
        
        # c1=(2*right_fit[0]*y_eval+right_fit[1])*xm_per_pix/ym_per_pix
        # c2=2*right_fit[0]*xm_per_pix/(ym_per_pix**2)
        
        # curvature=((1+c1*c1)**1.5)/(np.absolute(c2))
        
        # left_pos=(left_fit[0]*(y_eval**2))+(left_fit[1]*y_eval)+left_fit[2]
        # right_pos=(right_fit[0]*(y_eval**2))+(right_fit[1]*y_eval)+right_fit[2]
        
        # dx=((left_pos+right_pos)/2-mid_x)*xm_per_pix
        # if dx>0:
        #     text='Left'
        # else:
        #     text='Right'
        
        # font=cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(result,'Radius of curvature  = %.2f m'%(curvature),(20,50), font, 1,(255,255,255),2,cv2.LINE_AA)
        
        # cv2.putText(result,'Vehicle position : %.2f m %s of center'%(abs(dx), text),(20,90),
        #                 font, 1,(255,255,255),2,cv2.LINE_AA)
        
        # return result

# img = cv2.imread('/home/alok/new_lane.py/123.png')
# lane_img = np.copy(img)
# canny_img = canny_cvt(lane_img)
# r_img = roi(canny_img)
# th_img = thresh(lane_img)
# # lines = cv2.HoughLinesP(r_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# # lines = cv2.HoughLinesP(th_img, 2, np.pi/180, 100, np.array([]), minLineLength=5, maxLineGap=5)
# # avg_line = avg_slope_intercept(lane_img, lines)
# # line_img = display_line(canny_img, avg_line)
# # line_img = cv2.cvtColor(line_img, cv2.COLOR_BAYER_GR2RGB)
# # combo_img = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)
# # cv2.imshow('canny', combo_img)
# pipe = pipeline(th_img, lane_img)

# cv2.imshow('thresh', th_img)
# cv2.waitKey(0)

# .............................................

# cap = cv2.VideoCapture('/home/alok/new_lane.py/123_t.mp4')
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    canny_img = canny_cvt(frame)
    th_img = thresh(frame)
    n_canny = n_canny_cvt(th_img)
    # # lines = cv2.HoughLinesP(r_img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # lines = cv2.HoughLinesP(th_img, 2, np.pi/180, 100, np.array([]), minLineLength=5, maxLineGap=10)
    # # avg_line = avg_slope_intercept(frame, lines)
    # line_img = display_line(canny_img, lines)
    # line_img = cv2.cvtColor(line_img, cv2.COLOR_BAYER_GR2RGB)
    # combo_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)

    # pipe = pipeline(th_img, frame)
    warp = prespective_trans(th_img)
    # cv2.imshow('img', combo_img)
    cv2.imshow('thresh', warp)
    
    cv2.waitKey(1)