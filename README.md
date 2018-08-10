## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: ./examples/undistort_output.png "Undistorted Chessboard"
[image2]: ./output_images/Undistorted_Image.png "Road Transformed"
[image3]: ./output_images/Combined_Thresholds.png "Binary Example"
[image4]: ./output_images/Binary_Warped_Img.png "Warp Example"
[image5]: ./output_images/lane_polynomial.jpg "Fit Visual"
[image6]: ./output_images/final_effect.png "Output"
[video1]: ./output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
**All relative codes are located in the file `ans/myReport.py`.**

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 42-90 of the file called `ans/myReport.py`. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
Just like the chessboard image, I also applied the distortion correction to the test image using the `cv2.undistort` function and got the previous result, in which we could see the difference of the hood.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 92-132 in `ans/myReport.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspectTransform()`, which appears in lines 135-150 in the file `myReport.py` (ans/myReport.py).  The `perspectTransform()` function takes as inputs source (`src`) and destination (`dst`) points, and outputs the transformation matrix.  I chose the hardcode the source and destination points in the following manner:

```python
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
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. In additon, I also transformed the thresholded binary image to get the warped counterpart one `bina_warped`.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used a function called `fit_polynomial` (lines 278-344) to generate the line points of the left and the right lanes. Then the `draw_result` function (lines 362-398) took lane points as input and fit them with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the function `real_curvature_centerPosition` (lines 346-360 in myReport.py). Firtst I calculated the meters per pixel in either x or y dimension, which combining with the lane line points would produce the real-world points location. Then I could get the 2nd order polynomia of the real-world lane lines with the function `np.polyfit()`. The curvature formula is $$ R_curve=\frac{[1+(\frac{dx}{dy})^2]^{\frac{3}{2}}}{|\frac{d^2x}{dy^2}|} $$, which would generate the real-world curvature of the lane.
The position of vehicle was assumed as the middle bottom point of the image. The lane center could be calculated with the bottom end points of both the left and the right lane lines. After comparing the vehicle position and the lane center, the position of the vehicle with respect to center was calculated out.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 401-417 in the function `reproject2Real()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 
My general pipeline is: 
Original color image> Undistorttion> (HL)S Color space threthold> Gradient direction and magnitute thresh> Warp perspective> Fit polynomial lines> Calculate curvature and vehicle center position> Inverse warp to real road.
Errors mostly occured in the `Fit polynomial lines` section because number of the binary points used for polynomial fit were sometimes either too many or too few.
- Too many points: when some other vehicle or some big shadow passed nearby the lane, the vehicle's (or the shaddow's) points polluted the original lane points, which made the output lane polynomial devious. To get rid of that, I used the lane polynomial of the previous frame to decrease the searching scope of current frame, which made the searching scope a narrow ribbon.
- Too few points: when there were abrasion on the lanes, I couldn't get enough widespread points to generate the polynomial line with the correct direction. To avoid that, I set the left and the right lane lines coresponding to each other.I calculated the distance between the two lanes of the previous frame. And if one lane line of current frame was lost (Judged by the position of the top or bottom end of each line), the lost lane line should be generated by the corresponding lane with moving a distance.