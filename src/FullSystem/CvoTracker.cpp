#include "CvoTracker.h"
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "util/debugVisualization.hpp"
#include "util/settings.h"
#include "util/PointConverters.hpp"
namespace dso {

  
  CvoTracker::CvoTracker() {
    cvo_align = new cvo::rkhs_se3;
  }

  CvoTracker::CvoTracker(//const CalibHessian & Hcalib,
                         int width,
                         int height,
                         bool is_inner_prod)
    : currRef(NULL),
      seqSourceFh(NULL),
      refImage(new float [width * height] ),
      w(width),
      h(height),
      cvo_align(new cvo::rkhs_se3),
      using_inner_product_residual(is_inner_prod)
  {
  }

  CvoTracker::~CvoTracker() {
    delete cvo_align;
    delete refImage;
  }

  template <typename Pt>
  void badPointFilter(const std::vector<Pt> & input,
                             std::vector<dso::CvoTrackingPoints> & output ) {
    int counter = 0;
    output.resize(input.size());
    for (int i = 0; i < input.size(); i++) {
      //if(rand()/(float)RAND_MAX > setting_desiredPointDensity * 3.5 / input.size())
      //  continue;
      
      if ( input[i].idepth > 0.011 
           //&& input[i].local_coarse_xyz(1) < setting_CvoDepthMax
           && input[i].local_coarse_xyz(1) < setting_CvoHeightMax 
 	  && input[i].local_coarse_xyz.norm() < 60 )  {
        //refPointsWithDepth[counter] = ptsWithDepth[i];
        if (input[i].num_semantic_classes){
          int semantic_class;
          input[i].semantics.maxCoeff(&semantic_class);
          if (classToIgnore.find( semantic_class ) != classToIgnore.end())
            continue;
        }
        Pnt_to_CvoPoint<Pt>(input[i], output[counter]);

        //Vec3f hsv;
        //RGBtoHSV(output[counter].rgb.data(), hsv.data());
        //output[counter].rgb = hsv;
        //output[counter].rgb = 
        counter++;
      }
    }
    output.resize(counter);
    
    
  }

  template 
  void badPointFilter<dso::CvoTrackingPoints>(const std::vector<dso::CvoTrackingPoints> & input,
                                                     std::vector<dso::CvoTrackingPoints> & output );
  template 
  void badPointFilter<dso::Pnt>(const std::vector<dso::Pnt> & input,
                                       std::vector<dso::CvoTrackingPoints> & output );

  
  void CvoTracker::setIntrinsic( CalibHessian & HCalib)
  {
    fx = HCalib.fxl();
    fy = HCalib.fyl();
    cx = HCalib.cxl();
    cy = HCalib.cyl();


    K  << fx, 0.0, cx,
      0.0, fy, cy,
      0.0, 0.0, 1.0;
    Ki = K.inverse();

  }

  template <typename Pt>
  void CvoTracker::setCurrRef(FrameHessian * ref,
                              ImageAndExposure * img,
                              const std::vector<Pt> & ptsWithDepth) {
    
    currRef = ref;
    if (img)
      memcpy(refImage, img->image, h * w * sizeof(float));

    badPointFilter<Pt>(ptsWithDepth, refPointsWithDepth);

    if (img && ref)
      save_img_with_projected_points("ref"+ std::to_string(ref->frameID) + ".png",
                                     img->image_rgb, 3,
                                     w, h, K, ptsWithDepth, true);
    std::cout<<"Cvo: set ref frame. # of point is "<<refPointsWithDepth.size()<<std::endl;
  }

  template  void CvoTracker::setCurrRef<dso::Pnt>(FrameHessian * ref,
                                                  ImageAndExposure * img,
                                                  const std::vector<dso::Pnt> & ptsWithDepth);
  template  void CvoTracker::setCurrRef<dso::CvoTrackingPoints>(FrameHessian * ref,
                                                                ImageAndExposure * img,
                                                                const std::vector<dso::CvoTrackingPoints> & ptsWithDepth);


  template <typename Pt>
  void CvoTracker::setSequentialSource(FrameHessian * source,
                                       ImageAndExposure * imgSource,
                                       const std::vector<Pt> &ptsSource) {
    
    seqSourceFh = source;
    badPointFilter<Pt>(ptsSource, seqSourcePoints);
    std::cout<<"Cvo: set cvo seq source frame. # of point is "<<seqSourcePoints.size()<<", frame id is "<<seqSourceFh->shell->incoming_id<<std::endl;
  }

  template void CvoTracker::setSequentialSource<Pnt>(FrameHessian * source, ImageAndExposure * imgSource,
                                                     const std::vector<Pnt> &ptSource);
  template void CvoTracker::setSequentialSource<CvoTrackingPoints>(FrameHessian * source,
                                                                   ImageAndExposure * imgSource,
                                                                   const std::vector<CvoTrackingPoints> & pts);
  

  void CvoTracker::setCurrRef(FrameHessian * ref,
                              ImageAndExposure * img) {

    currRef = ref;
    if (img)
      memcpy(refImage, img->image, h * w * sizeof(float));

    int counter = 0;
    refPointsWithDepth.resize(ref->pointHessians.size() );
    for (int i = 0; i < ref->pointHessians.size() ; i++) {
      //if(rand()/(float)RAND_MAX > setting_desiredPointDensity * 1.5 / ptsWithDepth.size())
      //  continue;
      auto && ph = *(ref->pointHessians[i]);
      /*
      if (ptsWithDepth[i].local_coarse_xyz(2) < 35 &&
         ptsWithDepth[i].local_coarse_xyz.norm() < 100) {
        refPointsWithDepth[counter] = ptsWithDepth[i];
        counter++;
      }
      */
    }
    refPointsWithDepth.resize(counter);

    //if (img)
    //  save_img_with_projected_points("ref"+ std::to_string(ref->frameID) + ".png", img->image, w, h, K, ptsWithDepth, true);
    std::cout<<"Cvo: set ref frame. # of point is "<<refPointsWithDepth.size()<<std::endl;
  }

  

  void CvoTracker::setCurrRefPts(std::vector<ImmaturePoint *> & immature_pts) {
    refPointsWithDepth.resize(immature_pts.size());
    int counter = 0;
    for (int i = 0; i < immature_pts.size(); i++) {
      auto & ip = *immature_pts[i];
      if ( (ip.lastTraceStatus == IPS_GOOD || ip.lastTraceStatus == IPS_SKIPPED)
           && std::isfinite(ip.idepth_max)
           &&  ip.lastTracePixelInterval < 8
           && (ip.idepth_max + ip.idepth_min) * 0.5 < 1 ) {
        CvoTrackingPoints new_cvo_p;
        new_cvo_p = ip.cvoTrackingInfo;
        //Vec3f hsv;
        //RGBtoHSV(new_cvo_p.rgb.data(), hsv.data());
        //new_cvo_p.rgb = hsv;
        
        new_cvo_p.idepth = (ip.idepth_max + ip.idepth_min) * 0.5;
        new_cvo_p.local_coarse_xyz  = Ki * Vec3f (ip.u, ip.v, 1) / (new_cvo_p.idepth);

        if (new_cvo_p.local_coarse_xyz.norm() > 100 ) continue;
        if (new_cvo_p.num_semantic_classes){
          int semantic_class;
          new_cvo_p.semantics.maxCoeff(&semantic_class);
          if (classToIgnore.find( semantic_class ) != classToIgnore.end())
            continue;
        }

        refPointsWithDepth.push_back(new_cvo_p);
        counter++;
      }
    }
    refPointsWithDepth.resize(counter);
    
  }  
  
  bool CvoTracker::trackNewestCvo(FrameHessian * newFrame,
                                  ImageAndExposure * newImage,
                                  const std::vector<Pnt> & newPtsWithDepth,
                                  bool isSequential,
                                  bool using_init_guess,
                                  const SE3 & refTolast,
                                  // outputs
                                  SE3 & lastToNew_output,
                                  double & lastResiduals,
                                  Vec3 & lastFlowIndicators,
                                  float & align_inner_prod_frame2frame,
                                  float & align_inner_prod_ref2newest
                                  ) const {

    FrameHessian * source_frame = isSequential? seqSourceFh: currRef;
    auto & source_points = isSequential?  seqSourcePoints : refPointsWithDepth;
    
    if ( source_points.size() == 0) {
      std::cout<<"Ref frame not setup in CvoTracker!\n";
      return false;
    }

    // filter out invalid points
    std::vector<CvoTrackingPoints> newValidPts;
    badPointFilter<dso::Pnt>(newPtsWithDepth, newValidPts);

    if (newImage && newImage->num_classes)
      visualize_semantic_image("cvo_new.png",newImage->image_semantics, newImage->num_classes, w, h);
    //if (newImage)
    //  save_img_with_projected_points("new" + std::to_string(newFrame->shell->incoming_id)  + ".png", newImage->image, 1,
    //                                 w, h, K, newValidPts, false);

  //save_points_as_color_pcd("new.pcd", newValidPts);
    //save_points_as_hsv_pcd("ref.pcd", refPointsWithDepth );

    lastFlowIndicators.setConstant(1000);
    std::cout<<"Start cvo align...target frame is "<<newFrame->shell->incoming_id<<", source frame is "<<source_frame->shell->incoming_id<<"\n using init is "<<using_init_guess<< std::endl;

    // feed the initial value and the two pcds into the cvo library
    Eigen::Affine3f init_guess;
    if (using_init_guess) {
      init_guess.linear() = lastToNew_output.inverse().rotationMatrix().cast<float>();
      init_guess.translation() = lastToNew_output.inverse().translation().cast<float>();
    }
    cvo_align->set_pcd<CvoTrackingPoints>( source_points, newValidPts,init_guess, using_init_guess);

    // core: align two pointcloud!
    cvo_align->align();
    float align_inner_prod_last2newest = align_inner_prod_frame2frame =  cvo_align->inner_product();
    
    // output
    Eigen::Affine3f cvo_out_eigen = cvo_align->get_transform();
    SE3 cvo_out_se3( cvo_out_eigen.linear().cast<double>(),
                     cvo_out_eigen.translation().cast<double>() );
    lastToNew_output = cvo_out_se3;
    if (isSequential) {
      // change it back to currRefToNew
      // lastToNew_output is the transform from last frame to the current frame
      lastToNew_output = refTolast * lastToNew_output;
    }

    // compute the residuals and optical flow w.r.t the ref frame
    SE3 refInNew = lastToNew_output.inverse();
    Vec6 residuals  = calcRes(newFrame, refInNew,
                              Vec2(0,0),  setting_coarseCutoffTH * 30);
    lastFlowIndicators = residuals.segment<3>(2);
    lastResiduals = sqrtf((float)residuals(0) / residuals(1));
    
    // shall we reject the alignment this time??
    //bool is_tracking_good = std::isfinite(lastResiduals) && ( align_inner_prod > 0.0015 :  lastResiduals < CvoTrackingMaxResidual) ;
    bool is_tracking_good = std::isfinite(align_inner_prod_last2newest) && !std::isnan(align_inner_prod_last2newest) &&  align_inner_prod_last2newest > setting_CvoFrameToFrameMinInnerProduct;
    if ( !is_tracking_good || align_inner_prod_last2newest <  setting_CvoFrameToFrameMinInnerProduct ) {
    //if (!std::isfinite(lastResiduals) || lastResiduals < 0.001 ) {
      std::cout<<"[Cvo] Tracking not good, inner product is "<<align_inner_prod_last2newest<<std::endl;
      static int inf_count = 0;
      std::string new_name = "new_fail" + std::to_string(inf_count) + "_frame"+to_string(newFrame->shell->incoming_id)+".pcd";
      std::string ref_name = "ref_fail" + std::to_string(inf_count) + "_frame"+to_string(source_frame->shell->incoming_id)+".pcd";
      save_points_as_color_pcd<CvoTrackingPoints>(new_name, newValidPts);
      save_points_as_color_pcd<CvoTrackingPoints>(ref_name, source_points );
      inf_count += 1;
    }

    if (seqSourceFh  && currRef != seqSourceFh)
      align_inner_prod_ref2newest = getInnerProductRefNewest(newValidPts,lastToNew_output );
    else
      align_inner_prod_ref2newest = align_inner_prod_last2newest;

    std::cout<<"Cvo_align ends. transform: \n"<<cvo_out_eigen.matrix()<<"\n Cvo refToCurr residual "<<lastResiduals<<", cvo inner product is "<<align_inner_prod_last2newest<<", cvo inner product between ref and newest is "<<align_inner_prod_ref2newest<<std::endl;

    return is_tracking_good;
    //return lastResiduals > 0.001;
  }

  float CvoTracker::getInnerProductRefNewest(const std::vector<CvoTrackingPoints> & newest_pts,
                                             const Eigen::Affine3f & ref2newest) const {
    return cvo_align->inner_product<CvoTrackingPoints>(refPointsWithDepth, newest_pts, ref2newest );
    
  }  

  float CvoTracker::getInnerProductRefNewest(const std::vector<CvoTrackingPoints> & newest_pts,
                                             const SE3 & ref2newest) const {
    Eigen::Affine3f transform;
    transform.linear() = ref2newest.rotationMatrix().cast<float>();
    transform.translation() = ref2newest.translation().cast<float>();

    return cvo_align->inner_product<CvoTrackingPoints>(refPointsWithDepth, newest_pts, transform );
    
  }  

  
  Vec6 CvoTracker::calcRes(FrameHessian * newFrame,
                           const SE3 & refInNew , // Here refInNew * p_ref = p_new
                           const Vec2 & affLL,
                           float cutoffTH
                           ) const  {

    bool debugPlot = false;

    //SE3 refInNew = refInNew.inverse();
    
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
      resImage = new MinimalImageB3(w,h,3);
      resImage->setConst(255);
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
      // float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
      float residual = hitColor[0] - ( refColor);


      //std::cout<<"refColor "<<refColor<<", hitColor "<<hitColor[0]<<", residual "<<residual<<std::endl;
      
      //Huber weight 
      float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);


      if(fabs(residual) > cutoffTH)
      {
        if(debugPlot)  {
          resImage->setPixel4(x, y, colorRed);
        }
        E += maxEnergy;
        numTermsInE++;
        numSaturated++;
      } 
      else
      {
        if(debugPlot) {
          float res_to_show = (residual < 127)? residual + 128 : 255;
          resImage->setPixel4(x, y, (uint8_t)res_to_show+128);
        }

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
      IOWrap::displayImageB3("RES", resImage, false);
      IOWrap::waitKey(200);
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
