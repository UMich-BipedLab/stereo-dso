#pragma once
 
#include "util/NumType.h"
#include "util/Pnt.h"
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "util/settings.h"
#include "util/FrameShell.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"
#include "HessianBlocks.h"
#include "util/globalCalib.h"
#include "Cvo/rkhs_se3.hpp"
#include "Cvo/data_type.h"

namespace dso
{
  struct Pnt;
  
  class CvoTracker {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    CvoTracker();
    CvoTracker(const CalibHessian & Hcalib, int w, int h);
    ~CvoTracker();
 
    // two frame tracking: compute the transform between the
    // current reference frame to the newFrameHessian
    bool trackNewestCvo ( // input: new frame (left)
                         FrameHessian * newFrameHessian,
                         const std::vector<Pnt> & ptsWithDepth,
                         // output: constant motion
                         SE3 &lastToNew_output);
                         //AffLight &aff_g2l_out);

    // when a new keyframe is selected, upate the CvoTracker
    // to use the new reference frame
    void setCurrRef(FrameHessian * ref,
                    const std::vector<Pnt> & ptsWithDepth) {
      currRef = ref;
      refPointsWithDepth = ptsWithDepth;
    }


    // getters
    FrameHessian * getCurrRef() {return currRef;}
    int getCurrRefId() {return currRef? currRef->shell->id : -1;}
    double getLastResiduals() {return lastResiduals;}
    Vec3 getLastFlowIndicators() {return lastFlowIndicators;}
    const std::vector<Pnt> & getRefPointsWithDepth() {return refPointsWithDepth;}

  private:
    cvo::rkhs_se3 * cvo_align;

    FrameHessian * currRef;

    // intrinsic matrix
    Mat33f K;
    Mat33f Ki;
    float fx;
    float fy;
    float fxi;
    float fyi;
    float cx;
    float cy;
    float cxi;
    float cyi;

    // size of the frame
    int w;
    int h;


    // for CvoTracking
    // the raw selected high gradient points for the current reference frame
    std::vector<Pnt> refPointsWithDepth;
    //int numPointsWithStaticDepth;



    // computer residual
    Vec6 calcRes(FrameHessian * newFrame, const SE3 & refToNew, const Vec2 & affLL, float cutOffThreshold );
    // latest tracking residual
    double lastResiduals;       // track residual
    Vec3 lastFlowIndicators;    // optical flow indicator 
    
  };
  
}
