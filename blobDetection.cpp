#include <iostream>
#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>



cv::Mat read_img(std::string filename);
cv::Mat apply_bilateral_filter(cv::Mat img_i);
void save_img(cv::Mat img, std::string filename);
void display_img(std::string filename);

cv::SimpleBlobDetector::Params set_parameters_BD();
std::vector<cv::KeyPoint> detect_blobs(cv::Mat img);
cv::Mat draw_blobs(cv::Mat img_i, std::vector<cv::KeyPoint> keypoints);

std::string new_filename(std::string filename);

double norm(cv::Matx31d vector){
  double x = vector(0,0);
  double y = vector(1,0);
  double z = vector(2,0);
  double norm_sqr = x*x + y*y + z*z;

  return std::sqrt(norm_sqr);
}

int main(int argc, char** argv){
  // Read image and apply filter
  std::string filename = argv[1];
  cv::Mat img_original = read_img(filename);
  cv::Mat img = apply_bilateral_filter(img_original); // bilateral is better for edge detection

  // Detect and draw blobs on image
  std::vector<cv::KeyPoint> keypoints = detect_blobs(img); // To change BlobDetector parameters, change function: set_parameters_BD()
  cv::Mat img_with_keypoints = draw_blobs(img, keypoints);

  // Store center of blobs in vector
  std::vector<cv::Point2f> pts;
  for(int i=0; i<keypoints.size(); i++) { pts.push_back( keypoints[i].pt ); }
  
  // Fit Ellipse:
  cv::RotatedRect box = cv::fitEllipse(pts);
  cv::ellipse(img_with_keypoints, box, cv::Scalar(0,0,255), 3, cv::LINE_AA); // 3 is thickness
  cv::circle(img_with_keypoints, box.center, 10, cv::Scalar(0,255,0), 10, cv::LINE_AA); // 10 is radius and 10 is thickness
  // circle is for drawing center
  
  // Save and display new image
  std::string filename_output = new_filename(filename);
  save_img(img_with_keypoints, filename_output);
  //display_img(filename_output);

  // FINDING CAMERA POSITION:
  
  int n_LEDs = 6;
  double Diameter = 0.3453; // meters [m]

  // Setting real world points: LEDs. Origin will be ellipse center.
  std::vector<cv::Point3d> pts_3D;
  for (int i = 0; i < n_LEDs; i++) {
      double angle = i * (2 * M_PI) / n_LEDs;
      double X = (Diameter / 2) * std::cos(angle);
      double Y = (Diameter / 2) * std::sin(angle);
      double Z = 0;
      pts_3D.push_back(cv::Point3d(X, Y, Z));
  }
  // 7th point shall be ellipse center (origin)
  pts_3D.push_back(cv::Point3d(0, 0, 0));
  pts.push_back(box.center);

  // Camera matrix:
  double fx = 3554.84;  // Focal length in x direction
  double fy = 3529.99646;  // Focal length in y direction
  double cx = 4506.97897;  // Principal point x coordinate
  double cy = 3192.566;  // Principal point y coordinate
  
  cv::Matx33d K ( fx, 0, cx,
		  0, fy, cy,
		  0, 0, 1);
  

  // Solve for the extrinsic parameters using OpenCV
  cv::Matx31d rotation_vector, translation_vector;
  bool success = cv::solvePnP(pts_3D, pts, K, cv::Mat(), rotation_vector, translation_vector); //make sure there is correspondence bet'n 2d and 3d points

  if (success) {
      // Convert rotation vector to rotation matrix
      cv::Matx33d R;
      cv::Rodrigues(rotation_vector, R);

      // Compute covariance matrix of rotation and translation vectors
      cv::Mat Jacobian;
      std::vector<cv::Point2d> probe;
      cv::projectPoints(pts_3D, rotation_vector, translation_vector, K, cv::Mat(), probe, Jacobian);
      cv::Mat Sigma = cv::Mat(Jacobian.t() * Jacobian, cv::Rect(0, 0, 6, 6)).inv();

      // Compute standard deviations:
      cv::Mat std_dev;
      cv::sqrt(Sigma.diag(), std_dev);
      // Rotation vector and rotation matrix:
      cv::Matx31d dev_rvec( std_dev.at<double>(0, 0),
			    std_dev.at<double>(0, 1),
			    std_dev.at<double>(0, 2) );
      cv::Matx33d dev_R;
      cv::Rodrigues(dev_rvec, dev_R);
      
      // Translation vector:
      cv::Matx31d dev_tvec( std_dev.at<double>(0, 3),
			    std_dev.at<double>(0, 4),
			    std_dev.at<double>(0, 5) );
      
      // Calculate distance from center of mPMT to camera (with translation vector)
      double distance = norm(translation_vector);
      // Distance uncertainty with translation vector:
      double delta_distance = (1/distance) * std::sqrt(
                                                       std::pow( translation_vector(0, 0) * dev_tvec(0,0), 2) +
                                                       std::pow( translation_vector(0, 1) * dev_tvec(0, 1), 2) +
                                                       std::pow( translation_vector(0, 2) * dev_tvec(0, 2), 2)
                                                       );
      
      // Translation vector in Real World coordinates (camera position)
      cv::Matx31d tvec_rw = -R.t() * translation_vector;
      // Uncertainty camera position
      cv::Matx31d dev_tvec_rw( 0.,
			      0.,
			      0.);
      
      for (int i = 0; i < tvec_rw.rows; i++) {
          double uncer_i = 0.;
          for (int j = 0; j < tvec_rw.rows; j++) {
              uncer_i += std::pow( translation_vector(i,0) * dev_R(j, i), 2);
              uncer_i += std::pow( R(j, i) * dev_tvec(j,0), 2);
          }
          uncer_i = std::sqrt(uncer_i);
          dev_tvec_rw(i, 0) = uncer_i;
      }

      // Distance calculation using camera position (Real World coordinates)
      double distance_rw = norm(tvec_rw); 

      // Distance uncertainty with real world vector:
      double delta_distance_rw = (1 / distance_rw) * std::sqrt(
                                                               std::pow(dev_tvec_rw(0, 0) * dev_tvec_rw(0, 0), 2) +
                                                               std::pow(dev_tvec_rw(0, 1) * dev_tvec_rw(0, 1), 2) +
                                                               std::pow(dev_tvec_rw(0, 2) * dev_tvec_rw(0, 2), 2)
                                                               );

      // Print results

      std::cout << "\nTranslation vector:\n" << translation_vector << std::endl;
      std::cout << "Translation vector standard deviations:\n" << dev_tvec << std::endl;

      std::cout << "\nRotation vector:\n" << rotation_vector << std::endl;
      std::cout << "Rotation vector standard deviations:\n" << dev_rvec << std::endl;
      
      std::cout << "\nRotation matrix:\n" << R << std::endl;
      std::cout << "Rotation matrix standard deviations:\n" << dev_R << std::endl;
      
      std::cout << "\nCamera position (Real World Coordinates):\n" << tvec_rw << std::endl;
      std::cout << "Camera position uncertainties:\n" << dev_tvec_rw << std::endl;
      
      std::cout << "\nDistance from camera to center of mPMT (calculated with translation vector): " << distance << " [m]" << std::endl;
      std::cout << "Uncertainty distance (tvec): " << delta_distance << " [m]" << std::endl;
      
      std::cout << "\nDistance from center of mPMT to camera (calculated with camera position RW): " << distance_rw << " [m]\n";
      std::cout << "Uncertainty distance (camera position RW): " << delta_distance_rw << " [m]" << std::endl;
  }
  else {
      std::cerr << "\nsolvePnP failed to find a solution." << std::endl;
  }

  return 0;
}





cv::Mat read_img(std::string filename)
{
  std::cout << "filename = " << filename << "\n";
  cv::Mat img = cv::imread(filename);
  if( img.empty() )
    {
      std::cerr << "Couldn't open image. Exiting..." << filename << "\n";
      exit(1);
    }
  else
    {
      return img;
    }
}



cv::Mat apply_bilateral_filter(cv::Mat img_i)
{
  cv::Mat img_f = img_i.clone();
  cv::bilateralFilter(img_i,img_f,5,75,75);
  return img_f;
}



cv::SimpleBlobDetector::Params set_parameters_BD()
{
  cv::SimpleBlobDetector::Params params;

  // By color. This filter compares the intensity of a binary image at the center of a blob to blobColor. If they differ, the blob is filtered out. 
  // Use blobColor = 0 to extract dark blobs and blobColor = 255 to extract light blobs.
  params.blobColor = 255;
  
  // Change thresholds
  params.minThreshold = 5;
  params.maxThreshold = 255;
  
  // Filter by Area.
  params.filterByArea = true;
  params.minArea = 12; // before(=400) couldn't detect blobs in simulated LED
  
  // Filter by Circularity
  params.filterByCircularity = true;
  params.minCircularity = 0.1;
  
  // Filter by Convexity
  params.filterByConvexity = false;
  params.minConvexity = 0.87;
  
  // Filter by Inertia
  params.filterByInertia = false;
  params.minInertiaRatio = 0.01;

  return params;
}



std::vector<cv::KeyPoint> detect_blobs(cv::Mat img)
{
  cv::SimpleBlobDetector::Params params = set_parameters_BD(); // Parameters can be set inside this function
  cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
  
  std::vector<cv::KeyPoint> keypoints;
  detector->detect( img, keypoints );
  std::cout << "n Keypoints = " << keypoints.size() << "\n";
  return keypoints;
}



cv::Mat draw_blobs(cv::Mat img_i, std::vector<cv::KeyPoint> keypoints)
{
  cv::Mat img_f = img_i.clone();
  
  // Flag ensures the size of the circle corresponds to the size of blob
  cv::drawKeypoints( img_i, keypoints, img_f, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  return img_f;
}



void save_img(cv::Mat img, std::string filename)
{
  if( img.empty() )
    {
      std::cout << "Couldn't open image " << filename << "\n";
      exit(1);
    }
  else
    {
      cv::imwrite(filename, img);
      std::cout << "Output saved as " << filename << "\n";
    }
}



void display_img(std::string filename)
{
  std::string cmd = "okular " + filename;
  std::cout << "\nExecuting okular:\n\n";

  const char* command = cmd.c_str();
  std::system( command );
}


std::string new_filename(std::string filename)
{
    std::string new_filename;
    size_t Pos = filename.find_last_of(".");
    if (Pos != std::string::npos) {
        // Insert "_output" before the extension
        new_filename = filename.substr(0, Pos) + "_output" + filename.substr(Pos);
    }
    else {
        // If there's no extension, simply append "_output"
        new_filename = filename + "_output";
    }
    return new_filename;
}
