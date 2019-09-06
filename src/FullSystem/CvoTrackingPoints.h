#pragma once
#include <Eigen/Dense>
#include <iostream>
namespace dso {
  
  struct CvoTrackingPoints {

    Vec3f color;
    Vec3f local_coarse_xyz;
    Eigen::VectorXf semantics;
    int numSemanticClasses;
    
    void printCvoInfo() {
      std::cout<<"Print PointHessian at "
               <<local_coarse_xyz(0)<<","
               <<local_coarse_xyz(1)<<","
               <<local_coarse_xyz(2)<<"\n";
      printf("RGB: %d,%d,%d\n", (int)color(0), (int)color(1), (int)color(2));
      std::cout<<numSemanticClasses << " semantics classes: ";
      for (int i = 0; i != numSemanticClasses; i++) {
        std::cout<<semantics(i)<<", ";
      }
      std::cout<<"\n";
    }
  };
  
}
