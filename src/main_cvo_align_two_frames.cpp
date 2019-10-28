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

void load_file_name(string assoc_pth, vector<string> &vstrFirstID, \
                    vector<string> &vstrSecondID, std::vector<Eigen::Matrix3f> &list_init_guess);

int main(int argc, char *argv[]) {
    // list all files in current directory.
    //You could put any file path in here, e.g. "/home/me/mwah" to list that directory
    int num_class= 0;
    string n_class_str;
    if (argc > 5) {
        n_class_str = argv[5];
        num_class = stoi(n_class_str);
    }
  
    std::string pth (argv[1]);
    std::string assoc (argv[2]);
    std::string pcd_folder (argv[3]);
    std::ofstream output_file(argv[4]);
  
    std::string assoc_pth = pth+assoc;
    std::vector<string> vstrFirstID;
    std::vector<string> vstrSecondID;
    std::vector<Eigen::Matrix3f> list_init_guess;

    load_file_name(assoc_pth, vstrFirstID, vstrSecondID, list_init_guess);


    for(int i=0; i<vstrFirstID.size(); ++i){
        
        std::string fst_pth = pth+pcd_folder+vstrFirstID[i];
        std::string scd_pth = pth+pcd_folder+vstrSecondID[i];

        vector<dso::CvoTrackingPoints> fst_points;
        vector<dso::CvoTrackingPoints> scd_points;
        dso::read_cvo_pointcloud_from_file<dso::CvoTrackingPoints>(fst_pth, fst_points);
        dso::read_cvo_pointcloud_from_file<dso::CvoTrackingPoints>(scd_pth, scd_points);
        
        Eigen::Affine3f init_guess;
        init_guess.matrix() = list_init_guess[i];

        cvo::rkhs_se3 cvo_align;
        
        // init_guess.matrix().setIdentity();
        // init_guess.matrix()(2, 3) = -0.75;

        // init_guess = cvo_align.get_transform().inverse();
        std::cout<<"\n=============================================\nat"<<i<<"\n iter";
        cvo_align.set_pcd(fst_points, scd_points, init_guess, true);
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

void load_file_name(string assoc_pth, vector<string> &vstrFirstID, \
                    vector<string> &vstrSecondID, std::vector<Eigen::Matrix3f> &list_init_guess){
    std::ifstream fAssociation;
    fAssociation.open(assoc_pth.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            string first;
            ss >> first;
            vstrFirstID.push_back(first);
            string second;
            ss >> second;
            vstrSecondID.push_back(second);
            
            Eigen::Matrix3f init_guess = Eigen::Matrix3f::Identity();
            ss >> init_guess(0,0) >> init_guess(0,1) >> init_guess(0,2) >> init_guess(0,3)
               >> init_guess(1,0) >> init_guess(1,1) >> init_guess(1,2) >> init_guess(1,3)
               >> init_guess(2,0) >> init_guess(2,1) >> init_guess(2,2) >> init_guess(2,3);
            
            list_init_guess.push_back(init_guess);

        }
    }
    fAssociation.close();
}