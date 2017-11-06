# **Finding Lane Lines on the Road** 

---

## **Goals**
* Make a pipeline that finds lane lines on the road
* Self review of the project 

---

## **Pipeline**
The core idea of the pipeline is identifying the lane markings, approximating them with straight lines and superimposing them on the original image. The pipeline can therefore be thought of two sub-pipelines. One for identifying the lane markings, and the other for approximating it.

Identification starts off by blurring the image to get rid of noise, and filtering colors that we care about. In this case, it is white and yellow. Once the required colors are extracted, the image is converted to grayscale and passed though Canny edge detection algorithm. At this point in time, we are left with edges formed by yellow and white colors in the image. Once we have the edges, we then apply a region mask where the lane markings are most likely to exist. Hough transform is applied on top of the region mask to identify lane line segments.

Once the line segments are identified, they are split into left and right lanes segments by exploiting the idea that they are on either side of the mid-point of image. Left and right lanes are then separately fitted with a straight line using the segment endpoints. The fit is then used to interpolate the lane lines in the region of interest. The interpolated lane lines are then superimposed on the origin image.

For video, there could be sudden variation in the fit between frames. We smooth this by exponentially smoothing fits across multiple frames. Context variable passed in to the image pipeline is used for this smoothing.

---

## **Limitations**
The pipeline adheres to the requirement of the project rubric but it has many known limitations:

# Generalization
The pipeline does not generalize well. The pipeline does not exploit fundamental properties of lane markings that are more robust. Lanes in a image should be identified independent of the position of the car. The current pipeline implementation has very strong dependency on the car being in the middle of the lane. The implementation is mostly hand tuned for 3 video scenarios provided as examples. 

    1. The approximated lane lines do not work well with curved lanes.
    2. Pipeline will break if the position of camera is changed.
    3. Lighting changes can impact correctness of lane identification.
    4. There is hard coding of many constants that manually tuned from example that will not generalize.
    5. More complex lanes will not be identified.
    6. Lane identification will break when car is changing lanes.
    7. Absence of lanes are not handled.

# Performance
The current implementation does not focuss much on performance of the solution. The performance of the pipeline is probably not good to be used for any practical self-driving experiements. Because performance was not a criteria for project evaluation, it was not given focus.

# Possible improvements
There are many possible directions for improvement.

    1. Lane identification cast as a machine learning problem will generalize much better.
    2. Fitting higher degree polynomials to the line segments should ideally take care of curves, but a simple polyfit with higher degree did not do the trick.
    3. Speed of execution can be improved by exploiting more parallelism from the cores. 
    4. Using image sequences in videos, one can combine predictions of lane positions from previous frames with lane identified in current frame to get a better estimate of lanes (Kalman filters may be?) 
