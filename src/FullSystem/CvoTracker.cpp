#include "CvoTracker.h"
#include <vector>
#include "../Cvo/data_type.h"

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

  bool CvoTracker::trackNewCvo(FrameHessian * newFrame,
                               const std::vector<Pnt> & newPtsWithDepth,
                               // outputs
                               SE3 & lastToNew_output) {
    if (currRef == nullptr || refPointsWithDepth.size()) {
      std::cout<<"Ref frame not setup in CvoTracker!\n";
      return false;
    }

    cvo_align->set_pcd(refPointsWithDepth, newPtsWithDepth);

    cvo_align->align();

    Eigen::Matrix4f cvo_out_eigen = cvo_align->get_transform();

    SE3 cvo_out_se3( cvo_out_eigen.linear().cast<double>();
                     cvo_out_eigen.translation().cast<double>() );
    
    lastToNew_output = cvo_out_se3;
    
  }

  

  Vec6 CvoTracker::calcRes(const SE3 & refToNew ,
                           float cutOffThreshold) {
    
  }
  
}
