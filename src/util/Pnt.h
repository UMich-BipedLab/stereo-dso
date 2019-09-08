#pragma once
#include <Eigen/Dense>
#include "NumType.h"
namespace dso {
  struct Pnt
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    // index in jacobian. never changes (actually, there is no reason why).
    float u,v;

    Vec3f rgb;
    Vec3f local_coarse_xyz;
    int num_semantic_classes;
    VecXf semantics;
    
    // idepth / isgood / energy during optimization.
    float idepth;
    bool isGood;
    Vec2f energy;		// (UenergyPhotometric, energyRegularizer)
    bool isGood_new;
    float idepth_new;
    Vec2f energy_new;

    float iR;
    float iRSumNum;

    float lastHessian;
    float lastHessian_new;

    // max stepsize for idepth (corresponding to max. movement in pixel-space).
    float maxstep;

    // idx (x+y*w) of closest point one pyramid level above.
    int parent;
    float parentDist;

    // idx (x+y*w) of up to 10 nearest points in pixel space.
    int neighbours[10];
    float neighboursDist[10];

    float my_type;
    float outlierTH;
  };
}
