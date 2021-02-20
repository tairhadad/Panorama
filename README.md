# Panorama
Create a panoramic image

Libraries nedded:
cv2 ,datetime ,numpy, argparse

=============================================================================

Code explenation:

to run the code please use the terminal and run the following line:
> Panorama.py path_left_img path_right_img path_output

The code first import the images than, start the preproceesing step on the images - smaller them in 30%, and change the height of the images to be equal

later, find the keypoints for every images using SIFT algorithem 
and connect the keypoints between the images with knnMatch function. 
this function returned all the matches  and i was selecting only the best matches. 
 
the next step is creating a homograpy matrix with the keypoints we found. 
the final step is to wrap the images.

============================================================================

Enjoy :)
