#include <cv.h>    
#include <highgui.h>    
#include <math.h> 
#include"opencv2/nonfree/nonfree.hpp"
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

void readme();

/** @function main */
int main()
{
	/*if (argc != 3)
	{
		readme(); return -1;
	}*/

	Mat img_scene = imread("sce_street.jpg");
	/*cv::adaptiveThreshold(img_scene, img_scene, 255, CV_ADAPTIVE_THRESH_MEAN_C, 
		CV_THRESH_BINARY_INV,16,5);*/
//————————检测标志圆边缘以及圆心————————————————————

	IplImage* RedCircle = cvLoadImage("sce_street.jpg", -1); ;
	CvScalar s;
	for (int i = 0; i < RedCircle->height; i++)
	{
		for (int j = 0; j < RedCircle->width; j++)
		{
			s = cvGet2D(RedCircle, i, j); // 获得像素值
			if (s.val[0]<80 && s.val[1]<80 && s.val[2]>110)
				//注意这里的012对应的是bgr，范围的意思是防止光线的明暗影响，可以适当放宽，另外你也可以选择其他的颜色空间，可以直接取消明暗影响，比如HSV
			{
				s.val[0] = 0;
				s.val[1] = 0;
				s.val[2] = 255;
			}  //如果满足条件就设置为红色
			else
			{
				s.val[0] = 0;
				s.val[1] = 0;
				s.val[2] = 0;
			} //如果不满足就设置为黑色
			cvSet2D(RedCircle, i, j, s);   //设置像素
		}
	}
	cvNamedWindow("image", 1);
	cvShowImage("image", RedCircle);  //显示出来
	//cvWaitKey();

	/*std::vector<cv::Mat>mv(img_scene.channels()); 
	/// Read image ( same size, same type )
	split(img_scene, mv);
	/// Create Windows
	//imshow("BG", mv[0]);
	//imshow("G", mv[1]);
	//imshow("R", mv[2]);
	GaussianBlur(mv[2], mv[2], Size(9, 9), 2, 2);
	//cvWaitKey();*/

	IplImage* GrayCircle = cvCreateImage(cvGetSize(RedCircle), IPL_DEPTH_8U, 1);//用原图像指针创建新图像
	cvCvtColor(RedCircle, GrayCircle, CV_BGR2GRAY);
	cvShowImage("RedCircle", RedCircle);
	cvShowImage("GrayCircle", GrayCircle);
	//cvWaitKey();
	cvSmooth(GrayCircle, GrayCircle, CV_GAUSSIAN, 5, 5);//平滑处理  
	vector<Vec3f> circles;
	HoughCircles((Mat)GrayCircle, circles, CV_HOUGH_GRADIENT, 1.3, 
		50,//不同圆之间最小距离 
		80, 
		80, 
		15, 
		165);//hough圆变换  
//GrayCircle = cvLoadImage("correct_Img_1.jpg",1);  
	if (circles.size() > 0) {
		for (int i = 0; i < circles.size(); i++)
		{
			CvPoint center;//圆心  
			center.x = circles[i][0];
			center.y = circles[i][1];

			int radius = circles[i][2];//半径  
			cvCircle(RedCircle, center, radius, Scalar(0, 255, 255), 1, 8, 0);//绘制圆  
			cvCircle(RedCircle, center, 3, Scalar(0, 255, 0), 3, 8, 0);//绘制圆心  
			std::cout << "圆心为:" << circles[i][0] << "," << circles[i][1] << "   直径为：" << 2 * circles[i][2] << "\n";//为了保证精度，以原值输出  
		}
		cvSaveImage("out.jpg", GrayCircle);
		cvNamedWindow("hough圆检测", 1);
		cvShowImage("hough圆检测", RedCircle);
		cvWaitKey();
		//————————识别圆内标识————————————————————
		std::vector<KeyPoint> keypoints_object, keypoints_scene;
		FlannBasedMatcher matcher;
		std::vector< DMatch > matches;//this vector is not stable

		for (int CircleIterator = 0; CircleIterator < circles.size(); CircleIterator++)
		{
			//int CircleIterator = 0;
			std::vector<string> objstring;
			objstring.push_back("noman");
			objstring.push_back("stop");
			objstring.push_back("nocar");
			objstring.push_back("nowhistling");
			objstring.push_back("noswimming");

			float SmallestAverageDistance = 10;
			int NumOfClosestObj = 0;

			for (int objItrator = 0; objItrator < objstring.size(); objItrator++) {
				Mat img_object;
				img_object = imread("obj_" + objstring[objItrator] + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
				//imshow("img_object", img_object);
				//cvWaitKey();

				if (!img_object.data || !img_scene.data)
				{
					std::cout << " --(!) Error reading images " << std::endl; return -1;
				}

				//-- Step 1: Detect the keypoints using SURF Detector
				int minHessian = 600;

				SurfFeatureDetector detector(minHessian);


				Mat sceneROI(img_scene, Rect(
					circles[CircleIterator][0] - 0.7*circles[CircleIterator][2],
					circles[CircleIterator][1] - 0.7*circles[CircleIterator][2],
					1.4*circles[CircleIterator][2],
					1.4*circles[CircleIterator][2])
					);//SET roi of the scene
				IplImage* img = cvCreateImage(cvGetSize(RedCircle), IPL_DEPTH_8U, 1);

				detector.detect(img_object, keypoints_object);
				detector.detect(sceneROI, keypoints_scene);
				//imshow("sceneROI", sceneROI);
				//cvWaitKey();

				//-- Step 2: Calculate descriptors (feature vectors)
				SurfDescriptorExtractor extractor;

				Mat descriptors_object, descriptors_scene;

				extractor.compute(img_object, keypoints_object, descriptors_object);
				extractor.compute(sceneROI, keypoints_scene, descriptors_scene);

				//-- Step 3: Matching descriptor vectors using FLANN matcher

				matcher.match(descriptors_object, descriptors_scene, matches);

				double max_dist = 0; double min_dist = 100;

				//-- Quick calculation of max and min distances between keypoints
				for (int i = 0; i < descriptors_object.rows; i++)
				{
					double dist = matches[i].distance;
					if (dist < min_dist) min_dist = dist;
					if (dist > max_dist) max_dist = dist;
				}

				printf("-- Max dist : %f \n", max_dist);
				printf("-- Min dist : %f \n", min_dist);

				//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
				std::vector< DMatch > good_matches;

				for (int i = 0; i < descriptors_object.rows; i++)
				{
					if (matches[i].distance < max_dist - (max_dist - min_dist) / 1.30)
					{
						good_matches.push_back(matches[i]);
					}
				}

				Mat img_matches;
				drawMatches(img_object, keypoints_object, sceneROI, keypoints_scene,
					good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
					vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

				//-- Localize the object
				std::vector<Point2f> obj;
				std::vector<Point2f> scene;
				float AverageDistance = 0;

				for (int i = 0; i < good_matches.size(); i++)
				{
					//-- Get the keypoints from the good matches
					obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
					scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
					AverageDistance += good_matches[i].distance;
					printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
				}
				AverageDistance = AverageDistance / good_matches.size();
				std::cout << objstring[objItrator] + " AverageDistance:" << AverageDistance << std::endl;
				if ((AverageDistance - SmallestAverageDistance) < 0) {
					SmallestAverageDistance = AverageDistance;
					NumOfClosestObj = objItrator;//最相似的物体
				}

				//if (obj.size() < 4) { printf("not so similiar!"); }
				//else {
				//	Mat H = findHomography(obj, scene, CV_RANSAC);

				//	//-- Get the corners from the image_1 ( the object to be "detected" )
				//	std::vector<Point2f> obj_corners(4);
				//	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_object.cols, 0);
				//	obj_corners[2] = cvPoint(img_object.cols, img_object.rows); obj_corners[3] = cvPoint(0, img_object.rows);
				//	std::vector<Point2f> scene_corners(4);

				//	perspectiveTransform(obj_corners, scene_corners, H);

				//	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
				//	line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
				//	line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
				//	line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
				//	line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
				//}
				//-- Show detected matches
				if (objItrator == objstring.size() - 1) {
					string text = objstring[NumOfClosestObj];
					Point pt(circles[CircleIterator][0], circles[CircleIterator][1]);
					Scalar color = CV_RGB(0, 255, 255);
					putText(img_scene, text, pt, CV_FONT_HERSHEY_DUPLEX, 1.0f, color);
					imshow("img_scene", img_scene);
					cvWaitKey();
				}
				//imshow("Good Matches & Object detection", img_matches);
				keypoints_object.clear();
				matches.clear();
			}
		}
	}
	else {
		std::vector<KeyPoint> keypoints_object, keypoints_scene;
		FlannBasedMatcher matcher;
		std::vector< DMatch > matches;//this vector is not stable

		//int CircleIterator = 0;
		std::vector<string> objstring;
		objstring.push_back("cola");
		objstring.push_back("heineken");
		objstring.push_back("pepsi");
		objstring.push_back("spirit");
		objstring.push_back("xuebi");
		objstring.push_back("KFC");
		objstring.push_back("STARBUCKS");

		float SmallestAverageDistance = 10;
		int NumOfClosestObj = 0;

		for (int objItrator = 0; objItrator < objstring.size(); objItrator++) {
			Mat img_object;
			img_object = imread("obj_" + objstring[objItrator] + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
			//imshow("img_object", img_object);
			//cvWaitKey();

			if (!img_object.data || !img_scene.data)
			{
				std::cout << " --(!) Error reading images " << std::endl; return -1;
			}

			//-- Step 1: Detect the keypoints using SURF Detector
			int minHessian = 600;

			SurfFeatureDetector detector(minHessian);
			IplImage* img = cvCreateImage(cvGetSize(RedCircle), IPL_DEPTH_8U, 1);

			detector.detect(img_object, keypoints_object);
			detector.detect(img_scene, keypoints_scene);
			//imshow("sceneROI", sceneROI);
			//cvWaitKey();

			//-- Step 2: Calculate descriptors (feature vectors)
			SurfDescriptorExtractor extractor;

			Mat descriptors_object, descriptors_scene;

			extractor.compute(img_object, keypoints_object, descriptors_object);
			extractor.compute(img_scene, keypoints_scene, descriptors_scene);

			//-- Step 3: Matching descriptor vectors using FLANN matcher

			matcher.match(descriptors_object, descriptors_scene, matches);

			double max_dist = 0; double min_dist = 100;

			//-- Quick calculation of max and min distances between keypoints
			for (int i = 0; i < descriptors_object.rows; i++)
			{
				double dist = matches[i].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}

			printf("-- Max dist : %f \n", max_dist);
			printf("-- Min dist : %f \n", min_dist);

			//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
			std::vector< DMatch > good_matches;

			for (int i = 0; i < descriptors_object.rows; i++)
			{
				if (matches[i].distance < max_dist - (max_dist - min_dist) / 1.30)
				{
					good_matches.push_back(matches[i]);
				}
			}

			Mat img_matches;
			drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

			//-- Localize the object
			std::vector<Point2f> obj;
			std::vector<Point2f> scene;
			float AverageDistance = 0;

			for (int i = 0; i < good_matches.size(); i++)
			{
				//-- Get the keypoints from the good matches
				obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
				scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
				AverageDistance += good_matches[i].distance;
				printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
			}
			AverageDistance = AverageDistance / good_matches.size();
			std::cout << objstring[objItrator] + " AverageDistance:" << AverageDistance << std::endl;
			if ((AverageDistance - SmallestAverageDistance) < 0) {
				SmallestAverageDistance = AverageDistance;
				NumOfClosestObj = objItrator;//最相似的物体
			}

			//-- Show detected matches
			if (objItrator == objstring.size() - 1) {
				if (obj.size() < 6) { printf("no similiar things found!"); }
				else {
					img_object = imread("obj_" + objstring[NumOfClosestObj] + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
					//imshow("img_object", img_object);
					//cvWaitKey();

					if (!img_object.data || !img_scene.data)
					{
						std::cout << " --(!) Error reading images " << std::endl; return -1;
					}

					//-- Step 1: Detect the keypoints using SURF Detector
					int minHessian = 600;

					SurfFeatureDetector detector(minHessian);
					IplImage* img = cvCreateImage(cvGetSize(RedCircle), IPL_DEPTH_8U, 1);

					detector.detect(img_object, keypoints_object);
					detector.detect(img_scene, keypoints_scene);
					//imshow("sceneROI", sceneROI);
					//cvWaitKey();

					//-- Step 2: Calculate descriptors (feature vectors)
					SurfDescriptorExtractor extractor;

					Mat descriptors_object, descriptors_scene;

					extractor.compute(img_object, keypoints_object, descriptors_object);
					extractor.compute(img_scene, keypoints_scene, descriptors_scene);

					//-- Step 3: Matching descriptor vectors using FLANN matcher

					matcher.match(descriptors_object, descriptors_scene, matches);

					double max_dist = 0; double min_dist = 100;

					//-- Quick calculation of max and min distances between keypoints
					for (int i = 0; i < descriptors_object.rows; i++)
					{
						double dist = matches[i].distance;
						if (dist < min_dist) min_dist = dist;
						if (dist > max_dist) max_dist = dist;
					}

					printf("-- Max dist : %f \n", max_dist);
					printf("-- Min dist : %f \n", min_dist);

					//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
					std::vector< DMatch > good_matches;

					for (int i = 0; i < descriptors_object.rows; i++)
					{
						if (matches[i].distance < max_dist - (max_dist - min_dist) / 1.30)
						{
							good_matches.push_back(matches[i]);
						}
					}

					Mat img_matches;
					drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
						good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
						vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

					//-- Localize the object
					std::vector<Point2f> obj;
					std::vector<Point2f> scene;
					float AverageDistance = 0;

					for (int i = 0; i < good_matches.size(); i++)
					{
						//-- Get the keypoints from the good matches
						obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
						scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
						AverageDistance += good_matches[i].distance;
						printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
					}

					Mat H = findHomography(obj, scene, CV_RANSAC);

					//-- Get the corners from the image_1 ( the object to be "detected" )
					std::vector<Point2f> obj_corners(4);
					obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_object.cols, 0);
					obj_corners[2] = cvPoint(img_object.cols, img_object.rows); obj_corners[3] = cvPoint(0, img_object.rows);
					std::vector<Point2f> scene_corners(4);

					perspectiveTransform(obj_corners, scene_corners, H);

					//-- Draw lines between the corners (the mapped object in the scene - image_2 )
					line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
					line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
					line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
					line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
					
				string text = objstring[NumOfClosestObj];
				Scalar color = CV_RGB(0, 255, 255);
				putText(img_matches, text, scene_corners[0] + Point2f(img_object.cols-100, 50), CV_FONT_HERSHEY_DUPLEX, 1.0f, color);

				imshow("img_scene", img_scene);

				imshow("Good Matches & Object detection", img_matches);
				cvWaitKey();
				}
			}

			keypoints_object.clear();
			matches.clear();
		}
	
	}
	return 0;
}

/** @function readme */
void readme()
{
	std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl;
}