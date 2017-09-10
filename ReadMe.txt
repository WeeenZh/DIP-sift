Features:

- Identify traffic signs

- Trademark identification



Process:

- The position of the identification card is detected by the red circle of the signage

- Extract the primary logo of the tag

- extract the main feature points

- matches the standard identity feature points in the data set

- Compare the average distance of matching for each standard logo

- the most recent recognition result as a sign in the scene



Code:

- set bgr color space threshold, separate display red pixels, background black

- Image smoothing

-hough circle detection

- if there is a circle, enter scene 1, sign recognition

- traverse all circles, cut circle inside the rectangle roi (region of interest)

- Extract the feature points within the roi (SURF Detector)

- Calculate the matching of the feature points within the roi with the standard identity (FLANN matcher)

- distance less than max_dist - (max_dist - min_dist) / 1.30 added to good_matches (excellent match)

- Calculate the average distance for all matches

- the end of the circle, the average distance of the smallest as the most similar objects

-

- if there is no circle, scene 2, logo recognition

- big problem with the scene 1, the whole image as roi

- Calculate the most similar objects

- Computational transformation matrix (findHomography)

- Draws the connection transformation matrix at each corner (a rectangular box that lists the matching logo

- End the return





Need to do:

Add a standard identifier to the data set: obj_xxx.jpg

Make traffic logo scene: the logo ps to the complex background



12-21

progress:

- Added logo recognition, positioning
