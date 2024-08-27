#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <TFile.h>
#include "Configuration.hpp"
#include "featureFunctions.hpp"
#include <opencv2/features2d.hpp>	//Blob

#include "PMTIdentified.hpp"

#include<cmath>

#include "hough_ellipse.hpp"

using std::string;
using std::vector;
using namespace cv;

void slow_ellipse_detection( const std::vector< cv::Vec3f > blobs, Mat& image_houghellipse, 
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
    for ( const HoughEllipseResult& her : hers ){ // Doesn't get inside for
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

      for ( const PMTIdentified& her : ellipse_pmts ){ // Doesn't get inside for
	//if ( her.bolts.size() < 9 ) continue;
	Size axes(  int(her.circ.get_a()), int(her.circ.get_b()) );
	Point center( int(her.circ.get_xy().x), int(her.circ.get_xy().y) );
	cv::ellipse( image_before, center, axes, RADTODEG( her.circ.get_phi() ), 0., 360,  Scalar (0, 0, 255), 10 );
	
	//std::cout << " this part is working \n";
	
	Scalar my_color( 0, 255, 255 );
	for ( const Vec3f& xyz : her.bolts ){
	  cv::circle( image_before, Point( xyz[0], xyz[1] ), 3, my_color, 1, 0 );
	  //image_ellipse.at<Scalar>( xy.x, xy.y ) = my_color;
	}
      }
      
      // annotate with bolt numbers and angles
      overlay_bolt_angle_boltid( ellipse_pmts, image_before );	  
      
      string outputname = build_output_filename ( infname , "houghellipse_before");
      std::cout<<"Writing image "<<outputname<<std::endl;
      imwrite (outputname, image_before );
    }

  }
}


SimpleBlobDetector::Params set_parameters_BD()
{
  SimpleBlobDetector::Params params;

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


std::vector<KeyPoint> detect_blobs(Mat img)
{
  SimpleBlobDetector::Params params = set_parameters_BD(); // Parameters can be set inside this function
  Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
  
  std::vector<KeyPoint> keypoints;
  detector->detect( img, keypoints );
  std::cout << "n Keypoints = " << keypoints.size() << "\n";
  return keypoints;
}


int main(int argc, char **argv) {
  TFile * fout = new TFile ("FindBoltLocation.root", "RECREATE");
  /* Code to draw one ellipse to play around
  float nxpix = (float)config::Get_double("ellipse_hough_xmax");
  float nypix = (float)config::Get_double("ellipse_hough_ymax");
  cv::Mat img = cv::Mat(nypix,nxpix,CV_8UC3,cv::Scalar(0,0,0));
  
  // Draw 6 blobs contained in one ellipse
  cv::Point2f center = cv::Point2f(nxpix/2., nypix/2.);
  //double ee = std::sqrt(1 - std::pow(bb / aa, 2));
  double ee = 0.5;
  int bb = nxpix/8; // nypix / 10;
  int aa = (int)(bb / std::sqrt(1 - ee*ee)); // nxpix / 8;
  cv::Size axes_size = cv::Size(bb, aa); // 93 y 128 para (750, 500)
  double angle = 35;
  double start_angle = 0.; double end_angle = 360.;
  int thickness = 5;
  // General ellipse
  //cv::ellipse(img, center, axes_size, angle, start_angle, end_angle, cv::Scalar(255,0,0), thickness);
  // Drawing 6 blobs
  for (int i = 0; i < 6; i++){
      double current_angle = start_angle + (360. / 6.) * i;
      cv::ellipse(img, center, axes_size, angle, current_angle+1., current_angle-1., cv::Scalar(0, 255, 0), thickness);
  }
  std::cout << "Semimajor axis (a) = " << aa << "\tSemiminor axis (b) = " << bb << "\tEccentricity = " << ee << std::endl;
  */
  
  std::string filename = std::string( argv[1] );//"ellipse.png";
  //cv::imwrite(filename, img);
  Mat img;
  imread(filename, img);
  
  // Detect blobs and store them inside vector
  std::vector<cv::Vec3f> blobs;
  std::vector<KeyPoint> keypoints = detect_blobs(img);
  for(int i = 0; i<keypoints.size(); i++){
    float x = keypoints[i].pt.x;
    float y = keypoints[i].pt.y;
    float r = keypoints[i].size / 2.0;
    blobs.push_back( Vec3f(x, y, r) );
    cv::circle(img,cv::Point2i(x,y),r,cv::Scalar(0,255,0),2);
    std::cout<<"("<<x<<","<<y<<")"<<std::endl;
  }
  
  //vector < Vec3f > blobs = {Vec3f(100,100,3),Vec3f(110,150,3), Vec3f(90,100,3)}; //(x,y,r)
  
  slow_ellipse_detection( blobs, img, true, filename); 

  fout->Write ();
  fout->Close ();
  
  return 0;
}
