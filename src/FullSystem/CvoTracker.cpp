#include "CvoTracker.h"
#include <vector>



namespace dso {
  CvoTracker::CvoTracker() {
    cvo_align = new cvo::rkhs_se3;
  }

  CvoTracker::CvoTracker(const CalibHessian & Hcalib,
                         int width,
                         int height)
    : currRef(nullptr),
      lastResiduals(0),
      w(width),
      h(height),
      cvo_align(new cvo::rkhs_se3)
  {
    
  }

  CvoTracker::~CvoTracker() {
    delete cvo_align;
  }

  bool CvoTracker::trackNewestCvo(FrameHessian * newFrame,
                                  const std::vector<Pnt> & newPtsWithDepth,
                                  // outputs
                                  SE3 & lastToNew_output
                                  ) {
    if (currRef == nullptr || refPointsWithDepth.size()) {
      std::cout<<"Ref frame not setup in CvoTracker!\n";
      return false;
    }

    lastFlowIndicators.setConstant(1000);

    cvo_align->set_pcd(w, h, currRef, refPointsWithDepth, newFrame, newPtsWithDepth);

    cvo_align->align();

    Eigen::Affine3f cvo_out_eigen = cvo_align->get_transform();

    SE3 cvo_out_se3( cvo_out_eigen.linear().cast<double>(),
                     cvo_out_eigen.translation().cast<double>() );
    
    lastToNew_output = cvo_out_se3;

    // compute the residuals and optical flow
    Vec6 residuals  = calcRes(newFrame, lastToNew_output,
                              Vec2(1,0),  setting_coarseCutoffTH);
    lastFlowIndicators = residuals.segment<3>(2);
    lastResiduals = sqrtf((float)residuals(0) / residuals(1));

    return true;

  }

  

  Vec6 CvoTracker::calcRes(FrameHessian * newFrame,
                           const SE3 & refInNew , // Here refInNew * p_ref = p_new
                           const Vec2 & affLL,
                           float cutoffTH
                           ) {

    bool debugPlot = true;
    
    float E = 0;
    int numTermsInE = 0;
    int numTermsInWarped = 0;
    int numSaturated=0;

    Eigen::Vector3f* dINew = newFrame->dIp[0]; // the intensity gradients of the new frame 

    Mat33f RKi = (refInNew.rotationMatrix().cast<float>() * Ki );
    // printf("the Ki is:\n %f,%f,%f\n %f,%f,%f\n %f,%f,%f\n -----\n",Ki[lvl](0,0), Ki[lvl](0,1), Ki[lvl](0,2), Ki[lvl](1,0), Ki[lvl](1,1), Ki[lvl](1,2), Ki[lvl](2,0), Ki[lvl](2,1), Ki[lvl](2,2) );
    Vec3f t = (refInNew.translation()).cast<float>();
    // printf("the t is:\n %f, %f, %f\n", t(0),t(1),t(2));
    //Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l).cast<float>();


    float sumSquaredShiftT=0;
    float sumSquaredShiftRT=0;
    float sumSquaredShiftNum=0;

    float maxEnergy = 2*setting_huberTH*cutoffTH-setting_huberTH*setting_huberTH;	// energy for r=setting_coarseCutoffTH.


    MinimalImageB3* resImage = 0;
   
    if(debugPlot)
    {
      resImage = new MinimalImageB3(w,h);
      resImage->setConst(Vec3b(255,255,255));
    }
   
    int nl = refPointsWithDepth.size();
    //	printf("the num of the points is: %d \n", nl);
    for(int i=0;i<nl;i++)
    {
      auto & p = refPointsWithDepth[i];
      float depth = p.idepth ;
      float x =  p.u;
      float y =  p.v;

      // points in the new frame
      Vec3f pt = RKi * Vec3f(x, y, 1) + t* depth;
      float u = pt[0] / pt[2];
      float v = pt[1] / pt[2];
      float Ku = fx * u + cx;
      float Kv = fy * v + cy;
      float new_idepth = depth/pt[2];
      // printf("Ku & Kv are: %f, %f; x and y are: %f, %f\n", Ku, Kv, x, y);


      // what does this do ???
      if( i%32==0)
      {
        // translation only (positive)
        Vec3f ptT = Ki * Vec3f(x, y, 1) + t*depth;
        float uT = ptT[0] / ptT[2];
        float vT = ptT[1] / ptT[2];
        float KuT = fx * uT + cx;
        float KvT = fy * vT + cy;

        // translation only (negative)
        Vec3f ptT2 = Ki * Vec3f(x, y, 1) - t*depth;
        float uT2 = ptT2[0] / ptT2[2];
        float vT2 = ptT2[1] / ptT2[2];
        float KuT2 = fx * uT2 + cx;
        float KvT2 = fy * vT2 + cy;

        //translation and rotation (negative)
        Vec3f pt3 = RKi * Vec3f(x, y, 1) - t*depth;
        float u3 = pt3[0] / pt3[2];
        float v3 = pt3[1] / pt3[2];
        float Ku3 = fx * u3 + cx;
        float Kv3 = fy * v3 + cy;

        //translation and rotation (positive)
        //already have it.

        sumSquaredShiftT += (KuT-x)*(KuT-x) + (KvT-y)*(KvT-y);
        sumSquaredShiftT += (KuT2-x)*(KuT2-x) + (KvT2-y)*(KvT2-y);
        sumSquaredShiftRT += (Ku-x)*(Ku-x) + (Kv-y)*(Kv-y);
        sumSquaredShiftRT += (Ku3-x)*(Ku3-x) + (Kv3-y)*(Kv3-y);
        sumSquaredShiftNum+=2;
      }

      if(!(Ku > 2 && Kv > 2 && Ku < w-3 && Kv < h-3 && new_idepth > 0)) continue;

      float refColor = currRef->dIp[0][ int(round( x + y * this->w)) ](0);
      Vec3f hitColor = getInterpolatedElement33(dINew, Ku, Kv, w);
      if(!std::isfinite((float)hitColor[0])) continue;

      // photometric residual
      float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
      //Huber weight 
      float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);


      if(fabs(residual) > cutoffTH)
      {
        if(debugPlot)
          resImage->setPixel4(x, y, Vec3b(0,0,255));
        E += maxEnergy;
        numTermsInE++;
        numSaturated++;
      }
      else
      {
        if(debugPlot)
          resImage->setPixel4(x, y, Vec3b(residual+128,residual+128,residual+128));

        E += hw *residual*residual*(2-hw);
        numTermsInE++;

        /*
        buf_warped_idepth[numTermsInWarped] = new_idepth; // inverted depth
        buf_warped_u[numTermsInWarped] = u;               // normed u
        buf_warped_v[numTermsInWarped] = v;               // normed v
        buf_warped_dx[numTermsInWarped] = hitColor[1];    // gradient x
        buf_warped_dy[numTermsInWarped] = hitColor[2];    // gradient y
        buf_warped_residual[numTermsInWarped] = residual; // redisual
        buf_warped_weight[numTermsInWarped] = hw;         // huber weight in the error function
        buf_warped_refColor[numTermsInWarped] = lpc_color[i]; // 
        numTermsInWarped++;
        */
      }
    }
    /*
    while(numTermsInWarped%4!=0)
    {
      buf_warped_idepth[numTermsInWarped] = 0;
      buf_warped_u[numTermsInWarped] = 0;
      buf_warped_v[numTermsInWarped] = 0;
      buf_warped_dx[numTermsInWarped] = 0;
      buf_warped_dy[numTermsInWarped] = 0;
      buf_warped_residual[numTermsInWarped] = 0;
      buf_warped_weight[numTermsInWarped] = 0;
      buf_warped_refColor[numTermsInWarped] = 0;
      numTermsInWarped++;
    }
    buf_warped_n = numTermsInWarped;
    */

    if(debugPlot)
    {
      IOWrap::displayImage("RES", resImage, false);
      IOWrap::waitKey(0);
      delete resImage;
    }

    Vec6 rs;
    rs[0] = E;
    rs[1] = numTermsInE;
    rs[2] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);
    rs[3] = 0;
    rs[4] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);
    rs[5] = numSaturated / (float)numTermsInE;

    return rs;

  }
  
}
