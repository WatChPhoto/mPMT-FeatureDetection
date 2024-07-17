#include "hough_ellipse.hpp"

#include <sstream>
#include <fstream>
#include <cstring>
#include <thread>
#include <mutex>
#include "TDirectory.h"
#include "TEllipse.h"
#include "TMarker.h"


//const double pi = std::acos(-1);
float RADTODEG( float rad ){ return rad*180/pi; }

void process_mem_usage(float& vm_usage, float& resident_set)
{
  vm_usage     = 0.0;
  resident_set = 0.0;

  // the two fields we want
  unsigned long vsize;
  long rss;
  {
    std::string ignore;
    std::ifstream ifs("/proc/self/stat", std::ios_base::in);
    ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
	>> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
	>> ignore >> ignore >> vsize >> rss;
  }

  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;
}



unsigned num_ellipses( const HoughEllipseResults& hrs) {
  unsigned nc=0;
  for ( const HoughEllipseResult& hr : hrs ){
    if ( hr.eltype == HoughEllipse ) ++nc;
  }
  return nc;
}

std::ostream& operator<<( std::ostream& os, const HoughEllipseResults& hrs ){
  for ( const HoughEllipseResult& hr : hrs ){
    os << hr;
  }
  return os;
}

std::ostream& operator<<( std::ostream& os, const HoughEllipseResult& hr ){
  if ( hr.eltype == HoughEllipse ){
    os<<"Hough Ellipse: (xc,yc)= ( "<<hr.e.get_xy().x<<", "<<hr.e.get_xy().y
      <<" )  bb="<<hr.e.get_b()
      <<" eccentricity="<<hr.e.get_e()
      <<" phi="<<hr.e.get_phi()
      <<" hough-peak="<<hr.peakval<<std::endl;
  } else {
    os<<"Hough Unused Hits::"<<std::endl;
  }
  os << "  Nhits="<<hr.data.size()<<std::endl;
  for ( const xypoint& xy : hr.data ){
    os << "    (x,y)= ( "<<xy.x<<", "<<xy.y<<" )"<<std::endl;
  }
  return os;
}



int EllipseHough::get_bbin  ( double bb ) const {
  float fbbwid = (fbbmax-fbbmin)/fNbb; 
  int bbin = (bb-fbbmin)/fbbwid;
  if (bbin<0 || bbin>= fNbb ) return -1;
  return bbin;
}

int EllipseHough::get_ebin  ( double ee ) const {
  float feewid = (feemax-feemin)/fNee; 
  int eein = (ee-feemin)/feewid;
  if (eein<0 || eein>= fNee ) return -1;
  return eein;
}

int EllipseHough::get_phibin( double phi ) const {
  float fphiwid = (fphimax-fphimin)/fNphi; 
  int phibin = (phi-fphimin)/fphiwid;
  if (phibin<0 || phibin>= fNee ) return -1;
  return phibin;
}

int EllipseHough::get_xbin  ( double x ) const {
  float fxbwid = (fxmax-fxmin)/fNx; 
  int xbin = (x-fxmin)/fxbwid;
  if (xbin<0 || xbin>= fNx ) return -1;
  return xbin;
}
int EllipseHough::get_ybin  ( double y ) const{
  float fybwid = (fymax-fymin)/fNy; 
  int ybin = (y-fymin)/fybwid;
  if (ybin<0 || ybin>= fNy ) return -1;
  return ybin;
}

float EllipseHough::get_b_frombin( int bin ) const{
  float fbbwid = (fbbmax-fbbmin)/fNbb; 
  return fbbmin + (bin+0.5)*fbbwid;
}

float EllipseHough::get_e_frombin( int bin ) const{
  float feewid = (feemax-feemin)/fNee; 
  return feemin + (bin+0.5)*feewid;
}

float EllipseHough::get_phi_frombin( int bin ) const{
  float fphiwid = (fphimax-fphimin)/fNphi; 
  return fphimin + (bin+0.5)*fphiwid;
}


float EllipseHough::get_x_frombin( int bin ) const {
  float fxbwid = (fxmax-fxmin)/fNx; 
  return fxmin + (bin+0.5)*fxbwid;
}

float EllipseHough::get_y_frombin( int bin ) const {
  float fybwid = (fymax-fymin)/fNy; 
  return fymin + (bin+0.5)*fybwid;
}


EllipseHough::EllipseHough( unsigned nbins_bb     , float bbmin   , float bbmax     ,
			    unsigned nbins_ee     , float eemin   , float eemax    ,
			    unsigned nbins_phiphi , float phiphimin, float phiphimax,
			    unsigned nbins_x      , float xmin    , float xmax     ,
			    unsigned nbins_y      , float ymin    , float ymax     ) :
  fNbb( nbins_bb )     , fbbmin( bbmin )     , fbbmax( bbmax ),
  fNee( nbins_ee )     , feemin( eemin )     , feemax( eemax ),
  fNphi( nbins_phiphi ), fphimin( phiphimin ), fphimax( phiphimax ), 
  fNx( nbins_x ), fxmin( xmin ), fxmax( xmax ),
  fNy( nbins_y ), fymin( ymin ), fymax( ymax )					   
{
  static unsigned instance_count=0;
  ++instance_count;
  std::ostringstream os;

  TDirectory * curdir = gDirectory;
  houghdir = gDirectory->mkdir( (std::string("ehough_")+std::to_string(instance_count)).c_str() );
  houghdir->cd();

  float dbb = (bbmax-bbmin)/nbins_bb;
  float dee = (eemax-eemin)/nbins_ee;
  float dphi = (phiphimax-phiphimin)/nbins_phiphi;

  fTransformed = new unsigned short****[ fNbb ]  ;
  fE = new ellipse_st**[ fNbb ];
  for ( unsigned ibb=0; ibb<fNbb; ++ibb){
    fTransformed[ ibb ] = new unsigned short***[ fNee ];
    fE[ ibb ] = new ellipse_st*[ fNee ];
    float bbm = bbmin + dbb*ibb;
    float bbp = bbmin + dbb*(ibb+1);
    float bb = (bbm+bbp)/2;

    for ( unsigned iee=0; iee<fNee; ++iee ){
      fTransformed[ibb][iee] = new unsigned short**[fNphi];
      fE[ibb][iee] = new ellipse_st[fNphi];
      float eem = eemin + dee*iee;
      float eep = eemin + dee*(iee+1);
      float ee  = (eem+eep)/2;

      for ( unsigned iphi=0; iphi<fNphi; ++iphi ){
	fTransformed[ibb][iee][iphi] = new unsigned short*[fNx];
	float phim = phiphimin + dphi*iphi;
	float phip = phiphimin + dphi*(iphi+1);
	float phi  = (phim+phip)/2;
	fE[ibb][iee][iphi].set_bephi( bb, ee, phi );

	for ( unsigned ix=0; ix<fNx; ++ix ){
	  fTransformed[ibb][iee][iphi][ix] = new unsigned short[fNy];
	}
      }
    }
  }
  curdir->cd();

  float vmuse =0., memuse=0.;
  process_mem_usage( vmuse, memuse );
  std::cout<<"Done. vmuse = "<<vmuse/1024/1024<<" GB, memuse = "<<memuse/1024/1024<<" GB "<<std::endl;
}

EllipseHough::~EllipseHough(){
  // cleanup!!!

  for ( unsigned ibb=0; ibb<fNbb; ++ibb){
    for ( unsigned iee=0; iee<fNee; ++iee ){
      for ( unsigned iphi=0; iphi<fNphi; ++iphi ){
	for ( unsigned ix=0; ix<fNx; ++ix ){
	  if ( fTransformed[ibb][iee][iphi][ix] ) delete [] fTransformed[ibb][iee][iphi][ix];
	}
	if ( fTransformed[ibb][iee][iphi] ) delete []  fTransformed[ibb][iee][iphi];
      }
      if ( fTransformed[ibb][iee] ) delete [] fTransformed[ibb][iee];
      if ( fE[ibb][iee] ) delete [] fE[ibb][iee];
    }
    if ( fTransformed[ ibb ] ) delete []  fTransformed[ ibb ];
    if ( fE[ ibb ] ) delete [] fE[ ibb ];
  }

  if ( fTransformed ) delete [] fTransformed;
  if ( fE ) delete [] fE;

}


const HoughEllipseResults& EllipseHough::find_ellipses( const std::vector< xypoint >& data ){
  //float fxbwid = (fxmax-fxmin)/fNx; 
  //float fybwid = (fymax-fymin)/fNy;

  std::cout<<"EllipseHough::find_ellipses on data with "<<data.size()<<" points"<<std::endl;
  fresults.clear();
  std::vector< xypoint > unused_hits = data;
  bool done = false;
  while ( unused_hits.size() > minhits && !done ){
    hough_transform( unused_hits );
    float vmuse =0., memuse=0.;
    process_mem_usage( vmuse, memuse );
    std::cout<<" vmuse = "<<vmuse/1024/1024<<" GB, memuse = "<<memuse/1024/1024<<" GB "<<std::endl;
    
    std::vector< HoughEllipseResult > hrs = find_maximum( unused_hits );
    
    process_mem_usage( vmuse, memuse );
    std::cout<<" vmuse = "<<vmuse/1024/1024<<" GB, memuse = "<<memuse/1024/1024<<" GB "<<std::endl;

    unsigned nfound=0;
    for ( HoughEllipseResult & hr : hrs ){
      if ( hr.peakval > threshold ) {
	hr.eltype = HoughEllipse;
	++nfound;
      }
      
      std::cout<<"Find ellipse "<<fresults.size()+1<<std::endl;
      std::cout<< hr <<std::endl;
      
      
      fresults.push_back( hr );
      //save_hough_histo( fresults.size(), hr );
      //plot_candidate( fresults.size(), hr );
    }
    
    if ( nfound == 0 ) done = true;
      
    process_mem_usage( vmuse, memuse );
    std::cout<<" vmuse = "<<vmuse/1024/1024<<" GB, memuse = "<<memuse/1024/1024<<" GB "<<std::endl;
  }
  

  return fresults;
}


void EllipseHough::zero_hough_counts(){
  size_t shortsized = sizeof( unsigned short );
  for ( unsigned ibb=0; ibb<fNbb; ++ibb){
    for ( unsigned iee=0; iee<fNee; ++iee ){
      for ( unsigned iphi=0; iphi<fNphi; ++iphi ){
	for ( unsigned ix=0; ix<fNx; ++ix ){
	  memset( fTransformed[ibb][iee][iphi][ix], 0, fNy*shortsized );
	}
      }
    }
  } 
}


void hough_transform_phibin( EllipseHough& eh, unsigned iphi, const std::vector< xypoint >& data ){ 
  
  float bbwid = (eh.fbbmax-eh.fbbmin)/eh.fNbb;
  
  for ( const xypoint& xy : data ){
    for ( unsigned ibb=0; ibb<eh.fNbb; ++ibb){
      for ( unsigned iee=0; iee<eh.fNee; ++iee ){	
	ellipse_st * elli = &eh.fE[ibb][iee][iphi]; // has set everything but x and y
	elli->set_xy( xy );
	
	float bb = elli->get_b();
	// pick number of angles based on bb
	unsigned nang = unsigned( 2 * bb / bbwid );
	float   dtheta = 2*pi/nang;
	for ( unsigned itheta = 0; itheta<nang; ++itheta ){
	  xypoint ab = elli->xy( itheta*dtheta );
	  
	  float a = ab.x;
	  float b = ab.y;
	    // calculate bin
	  int xbin = eh.get_xbin( a );
	  int ybin = eh.get_ybin( b );
	  if ( xbin<0 || ybin<0 ) continue;
	  
	  unsigned short weight = 1;
	  for ( int xx=-1; xx<2; ++xx ){
	    for ( int yy=-1; yy<2; ++yy ){
	      weight = 4-2*abs(xx)-2*abs(yy)+abs(xx*yy);
	      int curx = xbin + xx;
	      int cury = ybin + yy;
	      if ( curx >=0 && curx < eh.fNx &&
		   cury >=0 && cury < eh.fNy ){
		  /*		  std::cout<<"ibb= "<<ibb<<"  "
				  <<"iee= "<<iee<<"  "
				  <<"iphi= "<<iphi<<"  "
				  <<"icurx= "<<curx<<"  "
				  <<"icury= "<<cury<<std::endl;*/
		eh.fTransformed[ ibb ][ iee ][ iphi ][ curx ][ cury ] += weight ;
	      }
	    }
	  }
	}
      }
    }
  }
}



void EllipseHough::hough_transform( const std::vector< xypoint >& data ){

  std::cout<<"EllipseHough::hough_transform call on data with "<<data.size()<<" points"<<std::endl;
  
  std::vector< std::thread > phithreads;
  
  zero_hough_counts();
  
  float vmuse =0., memuse=0.;
  process_mem_usage( vmuse, memuse );
  std::cout<<"histos-reset vmuse = "<<vmuse/1024/1024<<" GB, memuse = "<<memuse/1024/1024<<" GB "<<std::endl;
  for ( unsigned iphi=0; iphi<fNphi; ++iphi ){
    phithreads.push_back( std::thread( hough_transform_phibin, std::ref(*this), iphi, std::ref( data ) ) );  
  }
  for ( unsigned iphi=0; iphi<fNphi; ++iphi ){
    phithreads[iphi].join();
  }
}


  


void find_maximum_phibin( EllipseHough& eh, unsigned iphi, binindices_st& best ){
  std::mutex m;
  best.pkval = 0;
  for ( unsigned ibb=0; ibb<eh.fNbb; ++ibb){
    for ( unsigned iee=0; iee<eh.fNee; ++iee ){
      for ( unsigned ix=0; ix<eh.fNx; ++ix ){
	for ( unsigned iy=0; iy<eh.fNy; ++iy ){
	  m.lock();
	  if ( eh.fTransformed[ibb][iee][iphi][ix][iy] > best.pkval ){
	    best = binindices_st( ibb, iee, iphi, ix, iy, 		  
				  eh.fTransformed[ibb][iee][iphi][ix][iy] ) ;	
	  }
	  m.unlock();
	}
      }
    }
  }
}

void find_maximum_xyslice( EllipseHough& eh, 
			   unsigned ixmin, unsigned ixmax,
			   unsigned iymin, unsigned iymax,
			   binindices_st& best ){
  //std::mutex m;
  best.pkval = 0;
  unsigned curval;
  for ( unsigned iphi=0; iphi<eh.fNphi; ++iphi ){
    for ( unsigned ibb=0; ibb<eh.fNbb; ++ibb){
      for ( unsigned iee=0; iee<eh.fNee; ++iee ){
	for ( unsigned ix=ixmin; ix<ixmax; ++ix ){
	  for ( unsigned iy=iymin; iy<iymax; ++iy ){
	    curval = eh.fTransformed[ibb][iee][iphi][ix][iy];
	    if ( curval > best.pkval ){
	      best.ibb = ibb;
	      best.iee = iee;
	      best.iphi = iphi;
	      best.ix = ix;
	      best.iy = iy;
	      best.pkval = curval;
	    }
	  }
	}
      }
    }
  }
}




//void EllipseHough::find_maximum( std::vector< xypoint >& hits, std::vector< HoughEllipseResult > & result ){
std::vector< HoughEllipseResult> EllipseHough::find_maximum( std::vector< xypoint >& hits ){
  std::vector< binindices_st > passed_threshold;
  std::vector< HoughEllipseResult > results;
  
  std::vector< std::thread > slice_threads;

  const int slice_size = 50;
  const int xslices = fNx/slice_size;
  const int yslices = fNy/slice_size;
  std::vector< std::pair< xypoint, xypoint> > slices;
  for ( unsigned ix=0; ix<xslices; ++ix ){
    unsigned ixmin = ix*slice_size;
    unsigned ixmax = (ix+1)*slice_size;
    if (ix == xslices-1 ) ixmax = fNx-1;
    for ( unsigned iy=0; iy<yslices; ++iy ){
      unsigned iymin = iy*slice_size;
      unsigned iymax = (iy+1)*slice_size;
      if (iy == yslices-1 ) iymax = fNy-1;
      slices.push_back( std::pair<xypoint,xypoint>( xypoint( ixmin, iymin), xypoint( ixmax, iymax ) ) );
    }
  }

  // loop over the hough transform array to find the peak
  // and store the "best" circle center and radius
  for ( unsigned islice=0; islice<slices.size(); ++islice ){
    passed_threshold.push_back( binindices_st() );
  }

  for ( unsigned islice=0; islice<slices.size(); ++islice ){
    //std::cout<<"find_maximum: starting thread "<<islice<<std::endl;
    unsigned ixmin = slices[islice].first.x;
    unsigned iymin = slices[islice].first.y;
    unsigned ixmax = slices[islice].second.x;
    unsigned iymax = slices[islice].second.y;
    slice_threads.push_back( std::thread( find_maximum_xyslice, std::ref(*this), 
					  ixmin, ixmax, iymin, iymax, std::ref( passed_threshold[islice]) ) );

    if ( islice % 10 == 9 ){
      for ( unsigned i=0; i<slice_threads.size(); ++i ){
	//std::cout<<"waiting for thread...."<<std::endl;
	slice_threads[i].join();
      }
      slice_threads.clear();
      //std::cout<<"threads up to "<<islice<<" done."<<std::endl;
    }
    //find_maximum_phibin( std::ref(*this), iphi, std::ref( passed_threshold[iphi]) );
    //std::cout<<"find_maximum: thread "<<iphi<<" started "<<std::endl;
  }


  for ( unsigned i=0; i<slice_threads.size(); ++i ){
    //std::cout<<"waiting for thread...."<<std::endl;
    slice_threads[i].join();
  }
  slice_threads.clear();
  //std::cout<<"threads up to "<<slices.size()<<" done."<<std::endl;

 
  // find the hits that are associated with the ellipse and add them
  // to the result, otherwise add them to list of unused_hits
  // use bin sizes in xc, yc as threshold distance for hit to be from circle
  float fxbwid = (fxmax-fxmin)/fNx;
  float fybwid = (fymax-fymin)/fNy;
  float rthres = 3;//drscaling * std::sqrt( fxbwid*fxbwid + fybwid*fybwid );
  
  
  std::sort( passed_threshold.begin(), passed_threshold.end() );
  std::cout<<"found "<<passed_threshold.size()<<" bins above threshold"<<std::endl;
  unsigned pkmax = 0;
  if ( passed_threshold.size() > 0 ) pkmax = passed_threshold.back().pkval;
  while ( passed_threshold.size() > 0 ){
    binindices_st bi = passed_threshold.back();
    passed_threshold.pop_back();
    
    if ( bi.pkval < pkmax/2 || bi.pkval < threshold ) break;

    float x = get_x_frombin( bi.ix );
    float y = get_y_frombin( bi.iy );
    bool alreadyfound=false;
    for ( const HoughEllipseResult& hr : results ) {
      if ( fabs( hr.e.get_xy().x - x ) < hr.e.get_b() &&
	   fabs( hr.e.get_xy().x - x ) < hr.e.get_b() ) {
	alreadyfound=true;
	//std::cout<<"skipping this bin, already found "<<hr.e<<std::endl;
	break;
      }
    }
    if (alreadyfound) continue;
    std::cout<<passed_threshold.size()<<" bins remain to be checked "<<std::endl;

    HoughEllipseResult curbest( ellipse_st( get_b_frombin( bi.ibb ),
					    get_e_frombin( bi.iee  ),
					    get_phi_frombin( bi.iphi ),
					    xypoint( x, y ) ),
				bi.pkval );

    std::cout<<"curbest.e "<<curbest.e<<" pk="<<curbest.peakval<<" / "<<threshold<<std::endl;  

    std::vector< xypoint > unused_hits;
 
    for ( xypoint xy : hits ){
      if ( fabs( xy.x - curbest.e.get_xy().x ) > 2*curbest.e.get_b() ||
	   fabs( xy.y - curbest.e.get_xy().y ) > 2*curbest.e.get_b() ) {
	unused_hits.push_back( xy );
	continue;
      }
      float dr  = curbest.e.dmin( xy );
      if ( fabs( dr ) < rthres ){
	curbest.data.push_back( xy );
      } else {
	unused_hits.push_back( xy );
      }
    }
    
    if ( curbest.data.size() >= minhits ) {
      hits = unused_hits;
      results.push_back( curbest );
      std::cout<<"New ellipse "<<results.size()<<" with "<<curbest.data.size()<<" points "
	       <<" --- "<<hits.size()<<" remaining hits to match"
	       <<std::endl;
    }
  }
      
  return results;
}

  
void EllipseHough::save_hough_histo( unsigned num, const HoughEllipseResult& hr ){
  TDirectory* curdir = gDirectory;
  houghdir->cd();
  
  std::string hname = "hough_space_" + std::to_string( num );

  std::string htitle = 
    std::string( "Hough Space bb = ") + std::to_string( get_b_frombin( hr.ibb ) ) +
    std::string( "ee = ") + std::to_string( get_b_frombin( hr.iee ) ) +
    std::string( "phi = ") + std::to_string( get_b_frombin( hr.iphi ) ) +
    " ; xc; yc ";

  std::cout<<"save_hough_histo: "<<htitle<<std::endl;
  TH2S* savehist = new TH2S( hname.c_str(), htitle.c_str(),
			     fNx, fxmin, fxmax,
			     fNy, fymin, fymax ) ;

  for ( unsigned ix=0; ix<fNx; ++ix ){
    for ( unsigned iy=0; iy<fNy; ++iy ){
      savehist->SetBinContent( ix+1, iy+1, 
			       fTransformed[hr.ibb][hr.iee][hr.iphi][ ix ] [iy ] );
    }
  }
  savehist->SetDirectory( houghdir );
  curdir->cd();
  std::cout<<"saved."<<std::endl;
}

void EllipseHough::plot_candidate( unsigned num, const HoughEllipseResult & hr ){
  TDirectory* curdir = gDirectory;
  houghdir->cd();
  
  std::string hname = std::string("hcircle_")+std::to_string(num);

  std::string htitle = 
    std::string( "Ellipse candidate bb = ") + std::to_string( get_b_frombin( hr.ibb ) ) +
    std::string( "ee = ") + std::to_string( get_b_frombin( hr.iee ) ) +
    std::string( "phi = ") + std::to_string( get_b_frombin( hr.iphi ) ) +
    " ; xc; yc ";

  std::cout<<"plot_candidate: "<<htitle<<std::endl;
  TH2S* hplot = new TH2S( hname.c_str(), htitle.c_str(),
			     fNx, fxmin, fxmax,
			     fNy, fymin, fymax ) ;
  hplot->SetMarkerStyle( 7 );
  hplot->SetDirectory( houghdir );

  for ( const xypoint& xy : hr.data ){
    hplot->Fill( xy.x, xy.y );
  }

  TEllipse *el = new TEllipse( hr.e.get_xy().x, hr.e.get_xy().y, hr.e.get_b(), 
			       hr.e.get_a(), 0., 360., RADTODEG( hr.e.get_phi() ) );
  el->SetFillStyle(0);
  hplot->GetListOfFunctions()->Add( el );
  TMarker *mark = new TMarker( hr.e.get_xy().x, hr.e.get_xy().y, 2 );
  hplot->GetListOfFunctions()->Add( mark );

  curdir->cd();
  std::cout<<"plotted."<<std::endl;

}

