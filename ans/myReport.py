import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

img_titles=['Original_Image','Undistorted_Image','Original_Image','Undistorted_Image']

def mulImg_show_plt(rows=1, cols=2,titles=img_titles,save=True,*imgs ):
    """
    show multiple subimages in a figure with plt
    """
    f, subWindows = plt.subplots(rows, cols, figsize=(16, 6))
    f.tight_layout()
    for i in range(len(imgs)):
        subWindows[i].imshow(imgs[i],cmap='gray')
        subWindows[i].set_title(titles[i], fontsize=10)
        plt.subplots_adjust(left=0., right=1., top=0.95, bottom=0.,wspace =0.1, hspace =0)
    if save:
        plt.savefig('../output_images/'+titles[-1])
    return

def mulImg_show_cv(rows=1,cols=2,titles=img_titles,save=True,*imgs ):
    """
    show multiple subimages in a figure with cv2 
    """
    imgNo=len(imgs)
    img_width,img_heithg=imgs[0].shape[:2]
    subImg_w=img_width//cols
    subImg_h=img_heithg//cols
#    outImt=np.zeros((total_h,total_w,imgs[0].shape[2]),imgs[0].dtype)
#    print("width is {0},height is {1}".format(img_width,img_heithg))
    outImg=[]
    for i in range(imgNo):
        if i==0:
            outImg=cv2.resize(imgs[i],(subImg_h,subImg_w))
        elif i<cols:
            outImg=np.hstack((outImg,cv2.resize(imgs[i],(subImg_h,subImg_w))))
    cv2.imshow('outImg', outImg) 
    return outImg

images = glob.glob('../camera_cal/calibration*.jpg')
def calibCamera(images):
    """
    Camera Calibration
    """
    nx=9
    ny=6
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

mtx, dist = calibCamera(images)


# Make a list of calibration images
images2 = glob.glob('../test_images/test*.jpg')

undistort_imgs=[]
boolSave=True
for fname,fname2 in zip(images,images2):
    img1 = cv2.imread(fname)
    dst1 = cv2.undistort(img1, mtx, dist, None, mtx)
    img2 = cv2.imread(fname2)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    dst2 = cv2.undistort(img2, mtx, dist, None, mtx)
    undistort_imgs.append(dst2)
    mulImg_show_plt(1,4,img_titles,boolSave,img1,dst1,img2,dst2)
    boolSave=False
#%%
def HLS_select(img,space=2,thresh=(0,255) ):
    """
    Color Transforms:
    Transform the original image to HLS color space, and choose one space to 
    filter the pixes with the threshold.
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    col_channel=hls[:,:,space]
    binary_output = np.zeros_like(col_channel)
    binary_output[(col_channel > thresh[0]) & (col_channel < thresh[1])] = 1
    return binary_output

def gradient_direction_magnitute_thresh(img,sobel_kernel=3, dir_thresh=(0, np.pi/2),
                                        mag_thresh=(0, 255)):
    gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    angel_sobel=np.arctan2(np.abs(sobely),np.abs(sobelx))
    
    mag_sobel=np.sqrt(np.power(sobelx,2)+np.power(sobely,2))
    scaled_mag = np.uint8(255*mag_sobel/np.max(mag_sobel))
    
    grad_binary=np.zeros_like(scaled_mag)
#    grad_binary[(scaled_mag > mag_thresh[0]) & (scaled_mag <= mag_thresh[1])] = 1
#    grad_binary[(angel_sobel > dir_thresh[0]) & (angel_sobel <= dir_thresh[1])] = 1
    grad_binary[(angel_sobel >= dir_thresh[0]) & (angel_sobel < dir_thresh[1]) & (scaled_mag >= mag_thresh[0]) & (scaled_mag < mag_thresh[1])] = 1
    return grad_binary

img_titles=['Undistorted_Image','Color_Threshold (S_channel)','Gradient_Threshold','Combined_Thresholds']
boolSave=True
thres_binary =[]
for fname in undistort_imgs:
    S_img=HLS_select(fname,space=2,thresh=(200, 255))
    grad_img=gradient_direction_magnitute_thresh(fname,3,dir_thresh=(0, np.pi/4), mag_thresh=(30, 100))
    color_binary = np.dstack(( grad_img,S_img,np.zeros_like(grad_img))) *255
    combined_binary = np.zeros_like(grad_img)
    combined_binary[(S_img == 1) | (grad_img == 1)] = 1
    thres_binary.append(combined_binary)
    mulImg_show_plt(1,4, img_titles,boolSave,fname,S_img,grad_img,color_binary)
    boolSave=False
plt.show()

#%%
def perspectTransform(src,dst):
    M = cv2.getPerspectiveTransform(src, dst)
    revM=cv2.getPerspectiveTransform(dst, src)
    return M,revM
img_size = (color_binary.shape[1], color_binary.shape[0])
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
persTransMat,Minv=perspectTransform(src,dst)


img_titles=['Color_Undistorted_Image','Color_Warped Img','Binary_Warped_Img']
colframe=undistort_imgs[0]
show_src_Img=colframe.copy()
col_warped=cv2.warpPerspective(colframe, persTransMat, img_size)
show_dst_Img=col_warped.copy()
cv2.polylines(show_src_Img, np.int_([src]),True, (0,255, 0),8)
cv2.polylines(show_dst_Img, np.int_([dst]),True, (0,255, 0),8)
bina_warped=cv2.warpPerspective(thres_binary[0], persTransMat, img_size)
boolSave=True
mulImg_show_plt(1,3, img_titles,boolSave, show_src_Img,show_dst_Img,bina_warped)
#cv2.imwrite("warpedBianry.jpg",bina_warped*255)
#%%

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
#        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
#        (win_xleft_high,win_y_high),(0,255,0), 2) 
#        cv2.rectangle(out_img,(win_xright_low,win_y_low),
#        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def search_around_poly(binary_warped,left_fit,right_fit,left2right_distance):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Choose the width of the margin around the previous polynomial to search
    margin = 100
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +left_fit[2] - margin)) \
                      & (nonzerox < (left_fit[0]*(nonzeroy**2) +left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +right_fit[2] - margin)) \
                      & (nonzerox < (right_fit[0]*(nonzeroy**2) +right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty, out_img

def top_foot(binary_warped,fit):
    y_floor=binary_warped.shape[0]-1
    x_floor=fit[0]*y_floor**2 + fit[1]*y_floor + fit[2]
    x_top=fit[2]
    return x_top,x_floor

global left_fit_pre,right_fit_pre,left2right_distance,unchange_left_cnt,unchange_right_cnt
unchange_left_cnt=0
unchange_right_cnt=0
def fit_polynomial(binary_warped, run_times=1):
    # Find our lane pixels first
    global left_fit_pre,right_fit_pre,left2right_distance,unchange_left_cnt,unchange_right_cnt
    if run_times==1:
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
    else:
        leftx, lefty, rightx, righty, out_img =  search_around_poly(binary_warped,
                                                                    left_fit_pre,right_fit_pre,
                                                                   left2right_distance)
#    print("length of leftx,rightx is {0},{1}".format(len(leftx),len(rightx)))
    if run_times==1 or len(leftx)>400:
        left_fit_now = np.polyfit(lefty, leftx, 2)
        unchange_left_cnt=0
    else:
        left_fit_now=left_fit_pre
        unchange_left_cnt+=1
    if run_times==1 or len(rightx)>400:
        right_fit_now = np.polyfit(righty, rightx, 2)
        unchange_right_cnt=0
    else:
        right_fit_now=right_fit_pre
        unchange_right_cnt+=1
    if unchange_left_cnt>3 or unchange_right_cnt>3:
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
        unchange_left_cnt=0
        unchange_right_cnt=0
        left_fit_now = np.polyfit(lefty, leftx, 2)
        right_fit_now = np.polyfit(righty, rightx, 2)
    
    imgWid=binary_warped.shape[1]-1
    left_top,left_foot = top_foot(binary_warped,left_fit_now)
    right_top,right_foot = top_foot(binary_warped,right_fit_now)
    if run_times >1:
        # smooth the lanes between left and right on current frame
        if (right_top-left_top)>left2right_distance*0.6 or \
            (right_foot-left_foot)>left2right_distance*0.6:
            if left_top>imgWid*3/5 or left_foot>imgWid*3/5 or left_foot <=0:
                left_fit_now=[right_fit_now[0],right_fit_now[1],right_fit_now[2]-left2right_distance]
            elif right_top<imgWid*2/5 or right_foot<imgWid*2/5 or right_foot>=imgWid:
                right_fit_now=[left_fit_now[0],left_fit_now[1],left_fit_now[2]+left2right_distance]
        if (right_top-left_top)>(right_foot-left_foot)*1.3:
            if len(leftx)> len(rightx):
                right_fit_now=[left_fit_now[0],left_fit_now[1],left_fit_now[2]+left2right_distance]
            else:
                left_fit_now=[right_fit_now[0],right_fit_now[1],right_fit_now[2]-left2right_distance]

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit_now[0]*ploty**2 + left_fit_now[1]*ploty + left_fit_now[2]
        right_fitx = right_fit_now[0]*ploty**2 + right_fit_now[1]*ploty + right_fit_now[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    y_eval=binary_warped.shape[0]-1
    left2right_distance=right_fit_now[0]*y_eval**2 + right_fit_now[1]*y_eval + right_fit_now[2]- \
    left_fit_now[0]*y_eval**2 - left_fit_now[1]*y_eval - left_fit_now[2]
    left_fit_pre=left_fit_now
    right_fit_pre=right_fit_now
    return out_img, left_fitx,right_fitx,ploty

def real_curvature_centerPosition(lefty, leftx,righty, rightx,y_eval,x_center):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # curvature of the lane
    real_left_fit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    real_right_fit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    center_fit=(real_left_fit+real_right_fit)/2
    center_curverad = (1+(2*center_fit[0]*y_eval*ym_per_pix+center_fit[1])**2)**(3/2)/abs(2*center_fit[0])  
    # position of the vehicle with respect to center
    left_lanex=real_left_fit[0]*(y_eval*ym_per_pix)**2 + real_left_fit[1]*y_eval*ym_per_pix + real_left_fit[2]
    right_lanex=real_right_fit[0]*(y_eval*ym_per_pix)**2 + real_right_fit[1]*y_eval*ym_per_pix + real_right_fit[2]
    lane_centerx=(left_lanex+right_lanex)/2
    vehicle_centerx=x_center*xm_per_pix
    return center_curverad, lane_centerx-vehicle_centerx

def draw_result(out_img, left_fitx,right_fitx,ploty, use_plt=True):
    # calculate curvature
    x_center=out_img.shape[1]//2
    real_curverad,real_deviation = real_curvature_centerPosition(ploty, left_fitx,ploty, right_fitx,out_img.shape[0]-1,x_center)
    str_curvature="Radius of Curvature = {0:.1f}(m)".format(real_curverad)
    str_deviation="Vechicle is {:.2f}m left of center".format(real_deviation) \
    if real_deviation>0 \
    else "Vechicle is {:.2f}m right of center".format(-real_deviation)
    # draw graph
    global left_fit_pre,right_fit_pre
    if use_plt:
        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow',label=r'$\alpha =\frac{1}{2}\ln(\frac{1-\varepsilon}{\varepsilon })$')
        plt.annotate(r'$f_L(y) = {0:.5f}y^2+{1:.5f}y+{2:.5f}$'
                     .format(left_fit_pre[0],left_fit_pre[1],left_fit_pre[2]),
                     xy = (left_fitx[600],600), xytext = (left_fitx[600]-200,550),\
                     arrowprops = dict(facecolor = "w", headlength = 10, headwidth = 4, width = 2),\
                     color='w',fontsize=9)
        plt.annotate(r'$f_R(y) = {0:.5f}y^2+{1:.5f}y+{2:.5f}$'
                     .format(right_fit_pre[0],right_fit_pre[1],right_fit_pre[2]),
                     xy = (right_fitx[100],100), xytext = (right_fitx[100]-600,200),\
                     arrowprops = dict(facecolor = "w", headlength = 10, headwidth = 4, width = 2),\
                     color='w',fontsize=9)
        plt.imshow(out_img)
        plt.savefig('../output_images/'+'lane_polynomial.jpg')
        plt.show()
        
    else:
        # opencv ways
        left_pts = np.dstack((np.array(left_fitx),np.array(ploty))).squeeze()
        right_pts = np.dstack((np.array(right_fitx),np.array(ploty))).squeeze()
        left_pts = left_pts.reshape((-1,1,2)).astype(np.int32)
        right_pts = right_pts.reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(out_img,[left_pts],False,(0,255,255),5)
        cv2.polylines(out_img,[right_pts],False,(0,255,255),5)
    return str_curvature,str_deviation,out_img


def reproject2Real(colframe,bina_warped,img_size,left_fitx,right_fitx,ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(bina_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size) 
    # Combine the result with the original image
    result = cv2.addWeighted(colframe, 1, newwarp, 0.3, 0)
    return result

out_img, left_fitx,right_fitx,ploty=fit_polynomial(bina_warped)
str_curvature,str_deviation,out_img=draw_result(out_img, left_fitx,right_fitx,ploty)
print("{0}\n{1}".format(str_curvature,str_deviation))
img_titles=['original_Image','final_effect']
boolSave=True
mulImg_show_plt(1,2, img_titles,boolSave, colframe,
                reproject2Real(colframe,bina_warped,img_size,left_fitx,right_fitx,ploty))
plt.text(350,50,str_curvature, color='w')
plt.text(350,100,str_deviation, color='w')
plt.savefig('../output_images/'+'final_effect')
plt.show()

#%%

def showTextOnImg(colImg,str_curvature,str_deviation):
#    str_curvature="Radius of Curvature = {0:.1f}(m)".format(real_curverad)
#    str_deviation="Vechicle is {:.2f}m left of center".format(real_deviation) \
#        if real_deviation>0 else "Vechicle is {:.2f}m right of center".format(-real_deviation)
    textImg=cv2.putText(colImg,str_curvature,(100,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
    textImg=cv2.putText(textImg,str_deviation,(100,100),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
    return textImg

cap=cv2.VideoCapture("../project_video.mp4")
#cap=cv2.VideoCapture("challenge_video.mp4")
#cap=cv2.VideoCapture("harder_challenge_video.mp4")
#cap=cv2.VideoCapture("record.avi")
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/4))
fps =cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('D','I','B',' ')
out = cv2.VideoWriter('../output.avi',fourcc, fps, size,True)
run_time=1
while(cap.isOpened()):  
    ret, frame = cap.read()
    if not ret: break
    frame= cv2.undistort(frame, mtx, dist, None, mtx)
    #color threshold
    S_img=HLS_select(frame,space=2,thresh=(100, 255))
    #gradient threshold
    grad_img=gradient_direction_magnitute_thresh(frame,3,dir_thresh=(0, np.pi/4), mag_thresh=(30, 100))
    color_binary = np.dstack(( np.zeros_like(grad_img),S_img,grad_img)) *255
    #perspect transformation
    combined_binary = np.zeros_like(grad_img)
    combined_binary[(S_img == 1) | (grad_img == 1)] = 1
    bina_warped=cv2.warpPerspective(combined_binary, persTransMat, img_size)
    #detect lanes & calculate curvature
    out_img, left_fitx,right_fitx,ploty=fit_polynomial(bina_warped,run_time)
    str_curvature,str_deviation,out_img=draw_result(out_img, left_fitx,right_fitx,ploty,False)
    run_time +=1
    # reproject to real images
    real_result=reproject2Real(frame,bina_warped,img_size,left_fitx,right_fitx,ploty)
    real_result=showTextOnImg(real_result,str_curvature,str_deviation)
    outsave= mulImg_show_cv(1,4,img_titles,False,frame,color_binary,out_img,real_result) 
    out.write(outsave)
    k = cv2.waitKey(3)  
    # ESC
    if (k & 0xff == 27):  
        break  
else:
    print("Can't open video")
cap.release()
out.release()
cv2.destroyAllWindows()

#%%

# transform movie from *.avi to *.mp4
from moviepy.editor import VideoFileClip
myclip = VideoFileClip("../output.avi")
print ("fps is {}, duration is {}, end is {}".format(myclip.fps, myclip.size,myclip.end) )
myclip.write_videofile("../output.mp4", fps=myclip.fps) # export as video
myclip.close

# remove *.avi for saving storage
import os
oldFile="../output.avi"
if os.path.exists(oldFile): 
    os.remove(oldFile)
    print("Now removed %s"%oldFile)
else:
    print('no such file:%s'%oldFile)