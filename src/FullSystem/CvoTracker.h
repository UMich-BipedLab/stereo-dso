#pragma once
  
#include "util/NumType.h"
#include "util/Pnt.h"
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "FullSystem/ImmaturePoint.h"
#include "util/settings.h"
#include "util/FrameShell.h"
#include "util/ImageAndExposure.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"
#include "HessianBlocks.h"
#include "util/globalCalib.h"
#include "Cvo/rkhs_se3.hpp"
#include "Cvo/data_type.h"
#include "CvoTrackingPoints.h"
namespace dso
{
  struct Pnt;

  class CvoTracker {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    CvoTracker();
    CvoTracker(//const CalibHessian & Hcalib,
               int w, int h);
    ~CvoTracker();
 
    // two frame tracking: compute the transform between the
    // current reference frame to the newFrameHessian
    bool trackNewestCvo ( // input: new frame (left)
                         FrameHessian * newFrameHessian,
                         ImageAndExposure * newImage,
                         const std::vector<Pnt> & ptsWithDepth,
                         bool isSequential,
                         const SE3 & refTolast,
                         // output: 
                         SE3 &lastToNew_output,
                         double & lastResiduals,
                         Vec3 & lastFlowIndicators,
                         float & align_inner_prod) const;
                         //AffLight &aff_g2l_out);

    void setIntrinsic( CalibHessian & HCalib);
    

    // when a new keyframe is selected, upate the CvoTracker
    // to use the new reference frame
    template <typename Pt>
    void setCurrRef(FrameHessian * ref,
                    ImageAndExposure * img,
                    const std::vector<Pt> & ptsWithDepth);

    template <typename Pt>
    void setSequentialSource(FrameHessian * source,
                             ImageAndExposure * imgSource,
                             const std::vector<Pt> &ptsSource);

    // when a new keyframe is selected, upate the CvoTracker
    // to use the new reference frame
    // here we use the active points of ref frame intead of all detected high gradient points
    void setCurrRef(FrameHessian * ref,
                    ImageAndExposure * img);
    void setCurrRefPts( std::vector<ImmaturePoint *> & immature_pts);
    

    // computer residual
    Vec6 calcRes(FrameHessian * newFrame,
                 const SE3 & refToNew,
                 const Vec2 & affLL,
                 float cutOffThreshold ) const;


    // getters
    FrameHessian * getCurrRef() {return currRef;}
    int getCurrRefId() {return currRef? currRef->shell->incoming_id : -1;}
    //    double getLastResiduals() {return lastResiduals;}
    //Vec3 getLastFlowIndicators() {return lastFlowIndicators;}
    const std::vector<CvoTrackingPoints> & getRefPointsWithDepth() {return refPointsWithDepth;}

  private:
    cvo::rkhs_se3 * cvo_align;

    // intrinsic matrix
    Mat33f K;
    Mat33f Ki;
    float fx, fy, cx, cy;

    // size of the frame
    int w;
    int h;

    // for sequential tracking (instead of frame to keyframe tracking)
    FrameHessian * seqSourceFh;
    std::vector<CvoTrackingPoints> seqSourcePoints;

    // for CvoTracking
    // the raw selected high gradient points for the current reference frame
    FrameHessian * currRef;
    float * refImage;
    std::vector<CvoTrackingPoints> refPointsWithDepth;
    //int numPointsWithStaticDepth;



    // latest tracking residual
    //double lastResiduals;       // track residual
    //Vec3 lastFlowIndicators;    // optical flow indicator 
    
  };
  
}
