#include "CvoTracker.h"
#include <vector>

namespace dso {
  CvoTracker::CvoTracker() {
    cvo_align = new cvo::rkhs_se3;
  }

  CvoTracker::CvoTracker(CalibHessian * Hcalib,
                         int width,
                         int height)
    : currRef(nullptr),
      lastResiduals(0),
      w(width),
      h(height),
      cvo_align(new cvo::rkhs_se3)
  {
    
  }

  ~CvoTracker::CvoTracker() {
    delete cvo_align;
  }

  static void fhToPcd() {
    
  }
  
  bool CvoTracker::trackNewCvo(FrameHessian * newFrame,
                               Pnt * ptsWithDepth,  // from the new frame
                               int numPtsWithDepth, // from the new frame
                               // outputs
                               SE3 & lastToNew_output) {
    
    
  }

  Vec6 CvoTracker::calcRes(const SE3 & refToNew ,
                           float cutOffThreshold) {
    
  }
  
}