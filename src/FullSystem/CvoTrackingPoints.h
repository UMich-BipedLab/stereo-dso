#pragma once
#include <Eigen/Dense>
#include <iostream>
#include "util/Pnt.h"

namespace dso {
  typedef Eigen::Vector3f Vec3f;

  // a simplified structrure holding tracking data
  struct CvoTrackingPoints {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int u,v;
    float idepth;
    Vec3f rgb;
    Vec3f local_coarse_xyz;
    Eigen::VectorXf semantics;
    int num_semantic_classes;
    
    void printCvoInfo() {
      std::cout<<"Print PointHessian at "
               <<local_coarse_xyz(0)<<","
               <<local_coarse_xyz(1)<<","
               <<local_coarse_xyz(2)<<"\n";
      printf("RGB: %d,%d,%d\n", (int)rgb(0), (int)rgb(1), (int)rgb(2));
      std::cout<<num_semantic_classes << " semantics classes: ";
      for (int i = 0; i != num_semantic_classes; i++) {
        std::cout<<semantics(i)<<", ";
      }
      std::cout<<"\n";
    }
  };

  void Pnt_to_CvoPoint(const Pnt & in, CvoTrackingPoints & out) {
    out.rgb = in.rgb;
    out.local_coarse_xyz = in.local_coarse_xyz;
    out.num_semantic_classes = in.num_semantic_classes;
    out.u = in.u;
    out.v = in.v;
    out.idepth = in.idepth;
    if (in.num_semantic_classes) 
      out.semantics = in.semantics;
  }

}
