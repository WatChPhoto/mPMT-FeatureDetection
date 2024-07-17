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
    h.set_minhits( config::Get_double("ellipse_hough_drscale") );

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
    for ( const HoughEllipseResult& her : hers ){
      ellipse_st pmtloc{ her.e };
      std::vector< Vec3f > boltlocs;
      std::vector< float > dists;
      for ( const xypoint& xy : her.data ){
	boltlocs.push_back( Vec3f( xy.x, xy.y, 3 ) );
	dists.push_back( her.e.dmin( xy ) );
      }
      ellipse_pmts.push_back( PMTIdentified( pmtloc, boltlocs, dists, her.peakval ) );
    }

        

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
	//if ( her.bolts.size() < 9 ) continue;
	Size axes(  int(her.circ.get_a()), int(her.circ.get_b()) );
	Point center( int(her.circ.get_xy().x), int(her.circ.get_xy().y) );
	ellipse( image_before, center, axes, RADTODEG( her.circ.get_phi() ), 0., 360,  Scalar (255, 102, 255), 2 );
	

	Scalar my_color( 0, 0, 255 );
	for ( const Vec3f& xyz : her.bolts ){
	  circle( image_before, Point( xyz[0], xyz[1] ), 3, my_color, 1, 0 );
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

int main(int argc, char **argv) {
  std::string filename = std::string(argv[1]);
  vector < Vec3f > blobs = {Vec3f(100,100,3),Vec3f(110,150,3), Vec3f(90,100,3)}; //(x,y,r)
  cv::Mat image_houghellipse = cv::imread(filename);
  
  slow_ellipse_detection( blobs, image_houghellipse, true, argv[1]); 

  return 0;
}
