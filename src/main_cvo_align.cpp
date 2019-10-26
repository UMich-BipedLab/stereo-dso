#include "Cvo/rkhs_se3.hpp"
#include "FullSystem/CvoTrackingPoints.h"
#include "util/PointConverters.hpp"
#include "boost/filesystem.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>

using namespace std;
using namespace boost::filesystem;



int main(int argc, char *argv[]) {
  // list all files in current directory.
  //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
  int num_class= 0;
  string n_class_str;
  if (argc > 5) {
    n_class_str = argv[5];
    num_class = stoi(n_class_str);
  }
  
  
  path p (argv[1] );
  std::ofstream output_file(argv[2]);
  int start_frame = stoi(argv[3]);
  int num_frames = stoi(argv[4]);
  
  
  vector<string> files;
  // cycle through the directory
  int total_num = 0;
  for(auto & p : boost::filesystem::directory_iterator( p ) )
  {
    // If it's not a directory, list it. If you want to list directories too, just remove this check.
    if (is_regular_file(p.path())) {
      // assign current file name to current_file and echo it out to the console.
      string current_file = p.path().string();
      files.push_back(string(argv[1]) + "/" + to_string(total_num) + ".txt" );
      total_num += 1;
      //cout <<"reading "<< current_file << endl; 

    }
  }

  std::cout<<"Just read file names\n";
  //sort(files.begin(), files.end());
  vector<vector<dso::CvoTrackingPoints> > all_pts(files.size());
  vector<vector<dso::CvoTrackingPoints>> downsampled(files.size());
  int i = 0;
  for (auto &f:  files) {
    std::cout<<"Reading "<<f<<std::endl;
    dso::read_cvo_pointcloud_from_file<dso::CvoTrackingPoints>(f, all_pts[i]);

    int counter = 0;
    for (auto && p: all_pts[i]) {
      counter++;
      //if (//counter % sample_freq != 0 ||
      //    fabs(p.dI_xy[0]) < 5 && fabs(p.dI_xy[1] )< 5 ||
      if (    p.local_coarse_xyz.norm() > 60 ||
              fabs(p.local_coarse_xyz(1)) > 2)
        continue;
      
      dso::CvoTrackingPoints p_new;
      p_new = p;
      downsampled[i].push_back(p_new);
    }
    
    i ++;
    if (i == num_frames) break;
  }
  std::cout<<"Just reading  names\n";
  
  cvo::rkhs_se3 cvo_align;
  Eigen::Affine3f init_guess;
  init_guess.matrix().setIdentity();
  init_guess.matrix()(2, 3) = -0.75;

  for (int i = start_frame; i< downsampled.size()-2 ; i++) {
    if (i > start_frame)
      init_guess = cvo_align.get_transform().inverse();
    //    init_guess.matrix()(2,3) = -2.5;
    //init_guess.setIdentity();
    std::cout<<"\n=============================================\nat"<<i<<"\n iter";
    cvo_align.set_pcd(downsampled[0+i], downsampled[1+i], init_guess, true);
    cvo_align.align();
    init_guess= cvo_align.get_accum_transform();
    Eigen::Matrix4f result = init_guess.matrix();
    std::cout<<"\n The inner product between "<<i <<" and "<< i+1 <<" is "<<cvo_align.inner_product()<<"\n";
    std::cout<<"Transform is \n";
    std::cout<<cvo_align.get_transform().matrix() <<"\n\n";
     output_file << result(0,0)<<" "<<result(0,1)<<" "<<result(0,2)<<" "<<result(0,3)<<" "
                <<result(1,0)<<" " <<result(1,1)<<" "<<result(1,2)<<" "<<result(1,3)<<" "
                <<result(2,0)<<" " <<result(2,1)<<" "<<result(2,2)<<" "<<result(2,3);
     //output_file << result.block<3,4>(0,0);
    output_file<<"\n";

  }
  

  output_file.close();

  return 0;
}
