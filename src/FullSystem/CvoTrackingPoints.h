#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>



namespace dso {
  typedef Eigen::Vector3f Vec3f;
 
  // a simplified structrure holding tracking data
  struct CvoTrackingPoints {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int u,v;
    float idepth;
    Vec3f rgb;
    float dI_xy[2];
    Vec3f local_coarse_xyz;
    int num_semantic_classes;
    Eigen::VectorXf semantics;

    void printCvoInfo() {
      std::cout<<"Print PointHessian at "
               <<local_coarse_xyz[0]<<","
               <<local_coarse_xyz[1]<<","
               <<local_coarse_xyz[2]<<"\n";
      printf("RGB: %d,%d,%d, dI_xy: %.3f %.3f\n", (int)rgb[0], (int)rgb[1], (int)rgb[2], dI_xy[0], dI_xy[1]  );
      std::cout<<num_semantic_classes << " semantics classes: ";
      for (int i = 0; i != num_semantic_classes; i++) {
        std::cout<<semantics(i)<<", ";
      }
      std::cout<<"\n";
    }


   
    void initializeFromImageWithoutDepth(int w, int h, float uf, float vf,
                                         float * image_rgb,
                                         Eigen::Vector3f * dI_ixy,
                                         int num_classes, float * semantic_distribution) {

      this->u = int(round(uf));
      this->v = int(round(vf));
      this->num_semantic_classes = num_classes;
      if (num_classes > 0) {
        this->semantics.resize(num_classes);
      } 
          
      std::vector<std::vector<int> > uv_surrounded = {{int(uf),int(vf)},
                                                      {int(uf)+1 , int(vf)+1},
                                                      {int(uf)+1, int(vf)},
                                                      {int(uf),int(vf)+1}};
      int counter = 0;
      for (auto && offset : uv_surrounded) {
        int u_ = offset[0];
        int v_ = offset[1];
        if ( !(u_ > -1 && u_ < w && v_ > -1 && v_ < h))
          continue;
        rgb(0) += image_rgb[3*( w * v_ + u_)];
        rgb(1) += image_rgb[3*( w * v_ + u_)+1];
        rgb(2) += image_rgb[3*( w * v_ + u_)+2];
        this->dI_xy[0] += dI_ixy[w * v_ + u_][1];
        this->dI_xy[1] = dI_ixy[w * v_ + u_][2];
        if (num_classes > 0) {
          memcpy(this->semantics.data(), &(semantic_distribution [num_classes * (w * v_ + u_) ])  , num_classes * sizeof(float));
        } 
        counter += 1;
      }
      if (counter) {
        rgb = rgb.eval() / counter;
        dI_xy[0] = dI_xy[0] / counter;
        dI_xy[1] = dI_xy[1] / counter;
        semantics = semantics.eval() / counter;
      }
    }

    CvoTrackingPoints & operator=(const CvoTrackingPoints & p) {
      this->u = p.u;
      this->v = p.v;
      idepth = p.idepth;
      rgb = p.rgb;
      dI_xy[0] = p.dI_xy[0];
      dI_xy[1] = p.dI_xy[1];
      local_coarse_xyz = p.local_coarse_xyz;
      num_semantic_classes = p.num_semantic_classes;
      if (num_semantic_classes) {
        semantics.resize(num_semantic_classes);
        memcpy(semantics.data(), p.semantics.data(), num_semantic_classes * sizeof(float));
      }
      return *this;
    }
  };

  template <typename Pnt>
  inline 
  void write_cvo_pointcloud_to_file(std::string filename, std::vector<Pnt> & points) {
    std::ofstream outfile(filename);
    int num_points = points.size();
    if (num_points == 0)
      return;
    int num_class = points[0].num_semantic_classes;
    outfile << num_points << " ";
    outfile << num_class<< "\n";
    for (int j = 0; j < num_points; j++) {
      auto & p = points[j];
      outfile << p.u << " "<< p.v<<" ";
      outfile << p.idepth<<" ";
      outfile << p.rgb(0) <<" " << p.rgb(1) <<" "<< p.rgb(2)<< " ";
      outfile << p.dI_xy[0]<<" "<<p.dI_xy[1]<<" ";
      outfile << p.local_coarse_xyz(0) <<" "<< p.local_coarse_xyz(1) <<" "<< p.local_coarse_xyz(2)<<" ";
      if (num_class > 0) {
        for ( int i = 0; i < num_class; i++) {
          outfile << p.semantics(i) << " ";
        }
      }
      outfile << "\n";

    }
    
    outfile.close();
    
    
    
  }

  template <typename Pnt>
  inline
  void read_cvo_pointcloud_from_file(std::string filename, std::vector<Pnt> & points){
    std::ifstream infile(filename);
    int num_points;
    int num_class = 0;
    infile >> num_points;
    infile >> num_class;
    points.clear();
    points.resize(num_points);
    for (int j = 0; j < num_points; j++) {
      auto & p = points[j];
      infile >> p.u >> p.v;
      infile >> p.idepth;
      infile >> p.rgb(0) >> p.rgb(1) >> p.rgb(2);
      infile >> p.dI_xy[0]>>p.dI_xy[1];
      infile >> p.local_coarse_xyz(0) >> p.local_coarse_xyz(1) >> p.local_coarse_xyz(2);
      p.num_semantic_classes = num_class;
      if (num_class > 0) {
        for ( int i = 0; i < num_class; i++) {
          infile >> p.semantics(i);
        }
      }
    }
    
    infile.close();
    
    
  }


  template <typename Pnt>
  void Pnt_to_CvoPoint(const Pnt & in, CvoTrackingPoints & out) {
    out.rgb = in.rgb;
    out.local_coarse_xyz = in.local_coarse_xyz;
    out.num_semantic_classes = in.num_semantic_classes;
    out.u = in.u;
    out.v = in.v;
    out.dI_xy[0] = in.dI_xy[0];
    out.dI_xy[1] = in.dI_xy[1];
    out.idepth = in.idepth;
    if (in.num_semantic_classes) 
      out.semantics = in.semantics;
  }

}
