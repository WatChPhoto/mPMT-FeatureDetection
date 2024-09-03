#include <iostream>
#include <fstream>
#include <cmath>
#include <TFile.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include "Configuration.hpp"
#include "featureFunctions.hpp"
#include "PMTIdentified.hpp"
#include "hough_ellipse.hpp"

using std::string;
using std::vector;
using namespace cv;

cv::Mat read_img(std::string filename);
cv::Mat apply_bilateral_filter(cv::Mat img_i);
void save_img(cv::Mat img, std::string filename);
void display_img(std::string filename);

cv::SimpleBlobDetector::Params set_parameters_BD();
std::vector<cv::KeyPoint> detect_blobs(cv::Mat img);
cv::Mat draw_blobs(cv::Mat img_i, std::vector<cv::KeyPoint> keypoints);
void order_blobs(std::vector<cv::Point2f> & pts, cv::Point2d center);
void draw_index(cv::Mat img, std::vector<cv::Point2f> pts);
cv::Mat set_camera_matrix();
std::vector<cv::Point3d> set_realworld_pts(std::vector<cv::Point2f>& pts, cv::Point2d ellipse_center, int n_LEDs, double Diameter);
double calc_vec_magnitude(cv::Mat vec);
double calc_uncer_vec_magnitude(cv::Mat vec, cv::Mat uncer_vec);
cv::Mat calc_dev_rmat_brute(cv::Mat rvec, cv::Mat dev_rvec);
cv::Mat calc_uncer_camera_position(cv::Mat tvec, cv::Mat rmat, cv::Mat dev_tvec, cv::Mat dev_rmat);
void print_results(cv::Mat tvec, cv::Mat dev_tvec, cv::Mat rvec, cv::Mat dev_rvec, cv::Mat rmat, cv::Mat dev_rmat, cv::Mat tvec_rw, cv::Mat dev_tvec_rw, double distance, double delta_distance, double distance_rw, double delta_distance_rw);

std::string new_filename(std::string filename);

std::vector< PMTIdentified > slow_ellipse_detection(const std::vector< cv::Vec3f > blobs, Mat& image_houghellipse,
    bool write_image, const std::string& infname);

int main(int argc, char** argv){
  // Read image and apply filter
  std::string filename = argv[1];
  cv::Mat img_original = read_img(filename);
  cv::Mat img = apply_bilateral_filter(img_original); // bilateral is better for edge detection

  // Detect and draw blobs on image
  std::vector<cv::KeyPoint> keypoints = detect_blobs(img); // To change BlobDetector parameters, change function: Config.txt or function set_parameters_BD()
  cv::Mat img_with_keypoints = draw_blobs(img, keypoints);

  // Store center of blobs in a vector (NOT ORDERED INDEX YET)
  std::vector< cv::Vec3f > blobs; // format (x,y,radious)
  for (int i = 0; i < keypoints.size(); i++) { 
      cv::Vec3f blob = cv::Point3f(keypoints[i].pt.x, keypoints[i].pt.y, keypoints[i].size/2.);
      blobs.push_back(blob);
  }

  // HOUGH ELLIPSE DETECTION:
  //TFile* fout = new TFile("FindBoltLocation.root", "RECREATE"); // Uncomment these lines for ROOT histograms of Hough Space
  std::vector< PMTIdentified > mPMTs = slow_ellipse_detection(blobs, img, true, filename);
  //fout->Write(); // Uncomment these lines for ROOT histograms of Hough Space
  //fout->Close(); // Uncomment these lines for ROOT histograms of Hough Space

  for (const PMTIdentified& pmt : mPMTs) {

      // Store blobs of the current ellipse inside a vector
      std::vector<cv::Point2f> pts;
      for (const Vec3f& led : pmt.bolts) {
          float xx = led[0]; float yy = led[1];
          cv::Point2f point = cv::Point2f(xx, yy);
          pts.push_back(point);
      }
      // Store center of the current mPMT
      cv::Point2f centerPMT = cv::Point2f(pmt.circ.get_xy().x, pmt.circ.get_xy().y);
      
      // Order blobs counter-clockwise, and draw their index
      order_blobs(pts, centerPMT);
      std::string filename_output = new_filename(filename);
      cv::Mat img_output = read_img(filename_output);
      draw_index(img_output, pts);
      save_img(img_output, filename_output);

      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // FINDING CAMERA POSITION:
      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      int n_LEDs = (int)config::Get_int("mPMT_nLEDs");
      double Diameter = (double)config::Get_double("mPMT_Diameter");

      // Camera matrix:
      cv::Mat K = set_camera_matrix(); // Parameters can be changed from Config.txt file or inside function

      // Setting real world points: LEDs. Origin will be ellipse center.
      std::vector<cv::Point3d> pts_3D = set_realworld_pts(pts, centerPMT, n_LEDs, Diameter);

      // Solve for the extrinsic parameters using OpenCV
      cv::Mat rvec, tvec;
      bool success = cv::solvePnP(pts_3D, pts, K, cv::Mat(), rvec, tvec);
      if (!success) {
          std::cerr << "\nsolvePnP failed to find a solution for extrinsic parameters. Exiting...\n";
          exit(1);
      }

      // Compute covariance matrix of extrinsic parameters
      cv::Mat Jacobian;
      std::vector<cv::Point2d> probe;
      cv::projectPoints(pts_3D, rvec, tvec, K, cv::Mat(), probe, Jacobian);
      cv::Mat SigmaSquared = cv::Mat(Jacobian.t() * Jacobian, cv::Rect(0, 0, 6, 6)).inv();

      // Standard deviations of translation and rotation vectors
      cv::Mat std_dev;
      cv::sqrt(SigmaSquared.diag(), std_dev);
      cv::Mat dev_tvec = (cv::Mat_<double>(3, 1) << std_dev.at<double>(3, 0), std_dev.at<double>(4, 0), std_dev.at<double>(5, 0));
      cv::Mat dev_rvec = (cv::Mat_<double>(3, 1) << std_dev.at<double>(0, 0), std_dev.at<double>(1, 0), std_dev.at<double>(2, 0));

      // Rotation matrix and its standard deviation
      cv::Mat rmat;
      cv::Rodrigues(rvec, rmat);
      cv::Mat dev_rmat = calc_dev_rmat_brute(rvec, dev_rvec);

      // Translation vector in Real World coordinates (camera position)
      cv::Mat tvec_rw = -1 * rmat.t() * tvec;
      cv::Mat dev_tvec_rw = calc_uncer_camera_position(tvec, rmat, dev_tvec, dev_rmat);

      // Calculate distance from center of mPMT to camera (with translation vector)
      double distance = calc_vec_magnitude(tvec);
      double delta_distance = calc_uncer_vec_magnitude(tvec, dev_tvec);

      // Distance calculation using camera position (Real World coordinates)
      double distance_rw = calc_vec_magnitude(tvec_rw);
      double delta_distance_rw = calc_uncer_vec_magnitude(tvec_rw, dev_tvec_rw);

      // Print photogrammetry results for current mPMT:
      std::cout << "\n======================================================================================\n";
      std::cout << "\nPhotogrammetry results for mPMT at pixel " << centerPMT << " with LEDs:" << std::endl;
      for (int ii = 0; ii < n_LEDs; ii++) {
          std::cout << "LED_" << ii << " at pixel " << pts[ii] << std::endl;
      }
      print_results(tvec, dev_tvec, rvec, dev_rvec, rmat, dev_rmat, tvec_rw, dev_tvec_rw, distance, delta_distance, distance_rw, delta_distance_rw);

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
  params.blobColor = (int)config::Get_int("blob_color"); //255;
  
  // Change thresholds
  params.minThreshold = (float)config::Get_double("blob_minThreshold"); //5;
  params.maxThreshold = (float)config::Get_double("blob_maxThreshold"); //255;
  
  // Filter by Area.
  params.filterByArea = (bool)((int)config::Get_int("blob_filterByArea")); //true;
  params.minArea = (float)config::Get_double("blob_minArea"); //12; // before(=400) couldn't detect blobs in simulated LED
  
  // Filter by Circularity
  params.filterByCircularity = (bool)( (int)config::Get_int("blob_filterByCircularity") ); //true;
  params.minCircularity = (float)config::Get_double("blob_minCircularity"); //0.1;
  
  // Filter by Convexity
  params.filterByConvexity = (bool)((int)config::Get_int("blob_filterByConvexity")); // false;
  params.minConvexity = (float)config::Get_double("blob_minConvexity"); // 0.87;
  
  // Filter by Inertia
  params.filterByInertia = (bool)((int)config::Get_int("blob_filterByInertia")); //false;
  params.minInertiaRatio = (float)config::Get_double("blob_minInertiaRatio"); //0.01;
  
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
      std::cerr << "Couldn't open image " << filename << "\n";
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


void order_blobs(std::vector<cv::Point2f> & pts, cv::Point2d center)
{
    // Struct to store position and angle entangled
    struct Blob {
        cv::Point2d position;
        double angle;
    };

    // Lambda function to compare angle of blobs
    auto compareBlobs = [](Blob a, Blob b) {
        return a.angle > b.angle;
    };

    // Store positions and angles
    std::vector<Blob> blobs;
    for (int i = 0; i < pts.size(); i++) { // All blobs are taken into account. Careful with noise from outside ellipse
        cv::Point2d pos = pts[i];
        double angle = std::atan2(pos.y - center.y, pos.x - center.x);
        if (angle < 0) { angle += 2 * M_PI; }
        blobs.push_back({ pos, angle });
    }

    // Order blobs
    sort(blobs.begin(), blobs.end(), compareBlobs);

    // Rewrite pts with ordered_pts
    for (int i = 0; i < pts.size(); i++) {
        pts[i] = blobs[i].position;
    }
}


void draw_index(cv::Mat img, std::vector<cv::Point2f> pts)
{
    for (int i = 0; i < pts.size(); i++) {
        cv::putText(img, std::to_string(i), pts[i], cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(255, 255, 255), 2);
    }
}


cv::Mat set_camera_matrix()
{
    double fx = (double)config::Get_double("camera_fx");  // Focal length in x direction
    double fy = (double)config::Get_double("camera_fy");  // Focal length in y direction
    double cx = (double)config::Get_double("camera_cx");  // Principal point x coordinate
    double cy = (double)config::Get_double("camera_cy");  // Principal point y coordinate

    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    return K;
}


std::vector<cv::Point3d> set_realworld_pts(std::vector<cv::Point2f>& pts, cv::Point2d ellipse_center, int n_LEDs, double Diameter)
{
    std::vector<cv::Point3d> pts_3D;

    // LEDs are fixed every 60° for n=6 in the mPMT
    for (int i = 0; i < n_LEDs; i++) {
        double angle = i * (2 * M_PI) / n_LEDs;
        double X = (Diameter / 2) * std::cos(angle);
        double Y = (Diameter / 2) * std::sin(angle);
        double Z = 0;
        pts_3D.push_back(cv::Point3d(X, Y, Z));
    }
    
    // Last point shall be ellipse center (origin)
    pts_3D.push_back(cv::Point3d(0, 0, 0));
    pts.push_back(ellipse_center);
    
    return pts_3D;
}


cv::Mat calc_dev_rmat_brute(cv::Mat rvec, cv::Mat dev_rvec)
{
    int N = 10000; // # of perturbations to rvec
    cv::RNG rng(123); // Random Number Generator with seed
    
    // Apply N perturbations, calculate N perturbed rmat and store them in a vector
    std::vector<cv::Mat> pert_rmats;
    for (int i = 0; i < N; i++) {
        cv::Mat pert_rvec = rvec.clone();
        for (int row = 0; row < 3; row++) { pert_rvec.at<double>(row,0) += rng.gaussian( dev_rvec.at<double>(row,0) ); }
        cv::Mat pert_rmat;
        cv::Rodrigues(pert_rvec, pert_rmat);
        pert_rmats.push_back(pert_rmat);
    }

    // Calculate mean
    cv::Mat mean_rmat = (cv::Mat_<double>(3, 3) << 0., 0., 0., 0., 0., 0., 0., 0., 0.);
    for (int i = 0; i < N; i++) { mean_rmat += pert_rmats[i]; }
    mean_rmat /= N;

    // Calculate std dev
    cv::Mat var_rmat = (cv::Mat_<double>(3, 3) << 0., 0., 0., 0., 0., 0., 0., 0., 0.);
    for (int i = 0; i < N; i++) {
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                var_rmat.at<double>(row,col) += std::pow(pert_rmats[i].at<double>(row, col) - mean_rmat.at<double>(row, col), 2);
            }
        }
    }
    var_rmat /= N - 1;
    cv::Mat dev_rmat;
    cv::sqrt(var_rmat, dev_rmat);

    /*
    // Print difference of mean with respect to Rodrigues_rmat
    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);
    std::cout << rmat - mean_rmat << "\n";
    */

    return dev_rmat;
}


double calc_vec_magnitude(cv::Mat vec)
{
    if (vec.rows != 3 || vec.cols != 1) {
        std::cerr << "Error: calculating magnitude of a vector with rows != 3 or cols != 1. Exiting...\n";
        exit(1);
    }

    double magnitude = 0.;
    for (int i = 0; i < 3; i++) {
        magnitude += std::pow(vec.at<double>(i, 0), 2);
    }
    magnitude = std::sqrt( magnitude );

    return magnitude;
}


double calc_uncer_vec_magnitude(cv::Mat vec, cv::Mat uncer_vec)
{
    if (vec.rows != 3 || vec.cols != 1 || uncer_vec.rows != 3 || uncer_vec.cols != 1) {
        std::cerr << "Error: calculating magnitude uncertainty of a vector with rows != 3 or cols != 1. Exiting...\n";
        exit(1);
    }

    double magnitude = calc_vec_magnitude(vec);
    double uncer_magnitude = 0.;
    for (int i = 0; i < 3; i++) {
        uncer_magnitude += std::pow(vec.at<double>(i, 0) * uncer_vec.at<double>(i, 0), 2);
    }
    uncer_magnitude = std::sqrt(uncer_magnitude) / magnitude;

    return uncer_magnitude;
}


cv::Mat calc_uncer_camera_position(cv::Mat tvec, cv::Mat rmat, cv::Mat dev_tvec, cv::Mat dev_rmat)
{
    cv::Mat dev_cp = (cv::Mat_<double>(3, 1) << 0., 0., 0.);
    for (int i = 0; i < 3; i++) {
        double uncer_i = 0.;
        for (int j = 0; j < 3; j++) {
            uncer_i += std::pow(tvec.at<double>(i, 0) * dev_rmat.at<double>(j, i), 2);
            uncer_i += std::pow(rmat.at<double>(j, i) * dev_tvec.at<double>(j, 0), 2);
        }
        uncer_i = std::sqrt(uncer_i);
        dev_cp.at<double>(i, 0) = uncer_i;
    }
    return dev_cp;
}


void print_results(cv::Mat tvec, cv::Mat dev_tvec, cv::Mat rvec, cv::Mat dev_rvec, cv::Mat rmat, cv::Mat dev_rmat, cv::Mat tvec_rw, cv::Mat dev_tvec_rw, double distance, double delta_distance, double distance_rw, double delta_distance_rw)
{
    std::cout << "\nTranslation vector:\n" << tvec << std::endl;
    std::cout << "Translation vector standard deviations:\n" << dev_tvec << std::endl;

    std::cout << "\nRotation vector:\n" << rvec << std::endl;
    std::cout << "Rotation vector standard deviations:\n" << dev_rvec << std::endl;

    std::cout << "\nRotation matrix:\n" << rmat << std::endl;
    std::cout << "Rotation matrix standard deviations:\n" << dev_rmat << std::endl;

    std::cout << "\nCamera position (Real World Coordinates):\n" << tvec_rw << std::endl;
    std::cout << "Camera position uncertainties:\n" << dev_tvec_rw << std::endl;

    std::cout << "\nDistance from camera to center of mPMT (calculated with translation vector): " << distance << " [m]" << std::endl;
    std::cout << "Uncertainty distance (tvec): " << delta_distance << " [m]" << std::endl;

    std::cout << "\nDistance from center of mPMT to camera (calculated with camera position RW): " << distance_rw << " [m]" << std::endl;
    std::cout << "Uncertainty distance (camera position RW): " << delta_distance_rw << " [m]\n" << std::endl;
}


std::vector< PMTIdentified > slow_ellipse_detection( const std::vector< cv::Vec3f > blobs, Mat& image_houghellipse, 
			     bool write_image, const std::string& infname){ 

  bool do_ellipse_hough = (bool)config::Get_int( "do_ellipse_hough" );
  if ( do_ellipse_hough ){

    unsigned nbins_bb = (unsigned)config::Get_int("ellipse_hough_nbins_bb");
    unsigned nbins_ee = (unsigned)config::Get_int("ellipse_hough_nbins_ee");
    unsigned nbins_phiphi = (unsigned)config::Get_int("ellipse_hough_nbins_phiphi");
    unsigned nbins_x = (unsigned)config::Get_int("ellipse_hough_nbins_x");
    unsigned nbins_y = (unsigned)config::Get_int("ellipse_hough_nbins_y");
    //# above number of bins multiply as short (2 bytes per bin)
    //# therefore eg. 2 x 40 x 10 x 10 x 2300 x 1300 = 23.8 GB !!!
    float bbmin = (float)config::Get_double("ellipse_hough_bbmin");
    float bbmax = (float)config::Get_double("ellipse_hough_bbmax");
    float eemin = (float)config::Get_double("ellipse_hough_eemin");
    float eemax = (float)config::Get_double("ellipse_hough_eemax");
    float phiphimin = (float)config::Get_double("ellipse_hough_phphimin");
    float phiphimax = (float)config::Get_double("ellipse_hough_phiphimax");
    float xmin = (float)config::Get_double("ellipse_hough_xmin");
    float xmax = (float)config::Get_double("ellipse_hough_xmax");
    float ymin = (float)config::Get_double("ellipse_hough_ymin");
    float ymax = (float)config::Get_double("ellipse_hough_ymax");

    ///===========================================================
    /// Begin ellipse hough transfrom stuff
    EllipseHough h ( nbins_bb, bbmin, bbmax, 
		     nbins_ee, eemin, eemax,
		     nbins_phiphi, phiphimin, phiphimax,
		     nbins_x, xmin, xmax,
		     nbins_y, ymin, ymax );

    h.set_minhits( config::Get_int("ellipse_hough_minhits") );
    h.set_threshold( config::Get_int("ellipse_hough_threshold") );
    h.set_distance_factor( config::Get_double("ellipse_hough_drscale") ); // This was not called initially

    std::vector< xypoint > data;
    for ( unsigned i=0 ; i < blobs.size(); ++i ){
      //float radius = blobs[i][2];
      int blobx = blobs[i][0];
      int bloby = blobs[i][1];
      data.push_back( xypoint( blobx , bloby ) );
    }
    HoughEllipseResults hers = h.find_ellipses( data );	  
	  
    /// take hough resutls and fill vector of PMTIdentified info
    std::vector< PMTIdentified > ellipse_pmts;
    int n_ellipses = 0;
    for ( const HoughEllipseResult& her : hers ){
      n_ellipses++;
      ellipse_st pmtloc{ her.e };
      std::vector< Vec3f > boltlocs;
      std::vector< float > dists;
      for ( const xypoint& xy : her.data ){
	boltlocs.push_back( Vec3f( xy.x, xy.y, 3 ) );
	dists.push_back( her.e.dmin( xy ) );
      }
      ellipse_pmts.push_back( PMTIdentified( pmtloc, boltlocs, dists, her.peakval ) );
    }
    std::cout << "Number of ellipses detected: " << n_ellipses << std::endl;
   
    /// collect blobs that were put onto PMTs
    std::vector< Vec3f > blobs_on_pmts;
    for ( const HoughEllipseResult& her : hers ){
      for ( const xypoint& xy : her.data ){
	for ( const Vec3f& b : blobs ){
	  if ( fabs( b[0]-xy.x )< 1.0 && fabs( b[1]-xy.y ) < 1.0 ){
	    blobs_on_pmts.push_back( b );
	    break;
	  }
	}
      }
    }
   

    std::cout<<"========================== Before Pruning PMTS ===================================="<<std::endl;
    for  (const PMTIdentified & pmt : ellipse_pmts) {
      print_pmt_ellipse( std::cout, pmt );
      //std::cout<<pmt<<std::endl;
    }
    if ( write_image ){
      Mat image_before = image_houghellipse.clone();

      for ( const PMTIdentified& her : ellipse_pmts ){
	// Draw ellipse
	Size axes(  int(her.circ.get_a()), int(her.circ.get_b()) );
	Point center( int(her.circ.get_xy().x), int(her.circ.get_xy().y) );
	cv::ellipse( image_before, center, axes, RADTODEG( her.circ.get_phi() ), 0., 360,  Scalar (0, 0, 255), 10 );

	// Draw blobs again (on top of drawn ellipse)
	Scalar my_color( 0, 255, 255 );
	for ( const Vec3f& xyz : her.bolts ){
	  cv::circle( image_before, Point( xyz[0], xyz[1] ), 3, my_color, 1, 0 );
	  //image_ellipse.at<Scalar>( xy.x, xy.y ) = my_color;
	}
      }
      
      // annotate with bolt numbers and angles
      overlay_bolt_angle_boltid( ellipse_pmts, image_before );
      
      //string outputname = build_output_filename ( infname , "houghellipse_before");
      string outputname = new_filename(infname);
      std::cout<<"\nWriting image "<<outputname<<std::endl;
      imwrite (outputname, image_before );
    }
  return ellipse_pmts;
  }
  
  std::cout << "do_ellipse_hough = 0. Exiting...\n";
  exit(1);
  std::vector< PMTIdentified > null;
  return null;
}
