/**
 * This file is part of DSO.
 * 
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/CvoTracker.h"
#include "FullSystem/CvoTrackingPoints.h"
#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"
#include "util/debugVisualization.hpp"
#include <cmath>
#include <opencv/cv.h>
#include <limits>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace dso
{
  int FrameHessian::instanceCounter=0;
  int PointHessian::instanceCounter=0;
  int CalibHessian::instanceCounter=0;



  FullSystem::FullSystem()
  {

    int retstat =0;
    if(setting_logStuff)
    {

      retstat += system("rm -rf logs");
      retstat += system("mkdir logs");

      retstat += system("rm -rf mats");
      retstat += system("mkdir mats");

      calibLog = new std::ofstream();
      calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
      calibLog->precision(12);

      numsLog = new std::ofstream();
      numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
      numsLog->precision(10);

      coarseTrackingLog = new std::ofstream();
      coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
      coarseTrackingLog->precision(10);

      eigenAllLog = new std::ofstream();
      eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
      eigenAllLog->precision(10);

      eigenPLog = new std::ofstream();
      eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
      eigenPLog->precision(10);

      eigenALog = new std::ofstream();
      eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
      eigenALog->precision(10);

      DiagonalLog = new std::ofstream();
      DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
      DiagonalLog->precision(10);

      variancesLog = new std::ofstream();
      variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
      variancesLog->precision(10);


      nullspacesLog = new std::ofstream();
      nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
      nullspacesLog->precision(10);
    }
    else
    {
      nullspacesLog=0;
      variancesLog=0;
      DiagonalLog=0;
      eigenALog=0;
      eigenPLog=0;
      eigenAllLog=0;
      numsLog=0;
      calibLog=0;
    }

    assert(retstat!=293847);



    selectionMap = new float[wG[0]*hG[0]];

    coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
    coarseTracker = new CoarseTracker(wG[0], hG[0]);
    coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
    coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
    pixelSelector = new PixelSelector(wG[0], hG[0]);

    statistics_lastNumOptIts=0;
    statistics_numDroppedPoints=0;
    statistics_numActivatedPoints=0;
    statistics_numCreatedPoints=0;
    statistics_numForceDroppedResBwd = 0;
    statistics_numForceDroppedResFwd = 0;
    statistics_numMargResFwd = 0;
    statistics_numMargResBwd = 0;

    lastCoarseRMSE.setConstant(100);

    currentMinActDist=2;
    initialized=false;


    ef = new EnergyFunctional();
    ef->red = &this->treadReduce;

    isLost=false;
    initFailed=false;

    useCvo = true; // use cvo alignment instead of direct image alignment
    isCvoSequential = true;
    isCvoInnerProd = false;
    cvoTracker = new CvoTracker(wG[0], hG[0], isCvoInnerProd);
    cvoTracker_forNewKF = new CvoTracker( wG[0], hG[0], isCvoInnerProd);
    lastFrame = NULL;
    
    needNewKFAfter = -1;

    linearizeOperation=true;
    runMapping=true;
    mappingThread = boost::thread(&FullSystem::mappingLoop, this);
    lastRefStopID=0;



    minIdJetVisDebug = -1;
    maxIdJetVisDebug = -1;
    minIdJetVisTracker = -1;
    maxIdJetVisTracker = -1;
  }

  FullSystem::~FullSystem()
  {
    blockUntilMappingIsFinished();

    if(setting_logStuff)
    {
      calibLog->close(); delete calibLog;
      numsLog->close(); delete numsLog;
      coarseTrackingLog->close(); delete coarseTrackingLog;
      //errorsLog->close(); delete errorsLog;
      eigenAllLog->close(); delete eigenAllLog;
      eigenPLog->close(); delete eigenPLog;
      eigenALog->close(); delete eigenALog;
      DiagonalLog->close(); delete DiagonalLog;
      variancesLog->close(); delete variancesLog;
      nullspacesLog->close(); delete nullspacesLog;
    }

    delete[] selectionMap;

    for(FrameShell* s : allFrameHistory)
      delete s;
    for(FrameHessian* fh : unmappedTrackedFrames)
      delete fh;

    delete coarseDistanceMap;
    delete coarseTracker;
    delete coarseTracker_forNewKF;
    delete cvoTracker;
    delete cvoTracker_forNewKF;
    delete coarseInitializer;
    delete pixelSelector;
    delete ef;
  }

  void FullSystem::setOriginalCalib(VecXf originalCalib, int originalW, int originalH)
  {

  }


  void FullSystem::setGammaFunction(float* BInv)
  {
    if(BInv==0) return;

    // copy BInv.
    memcpy(Hcalib.Binv, BInv, sizeof(float)*256);


    // invert.
    for(int i=1;i<255;i++)
    {
      // find val, such that Binv[val] = i.
      // I dont care about speed for this, so do it the stupid way.

      for(int s=1;s<255;s++)
      {
        if(BInv[s] <= i && BInv[s+1] >= i)
        {
          Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
          break;
        }
      }
    }
    Hcalib.B[0] = 0;
    Hcalib.B[255] = 255;
  }

  void FullSystem::printResult(std::string file)
  {
    boost::unique_lock<boost::mutex> lock(trackMutex);
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

    std::ofstream myfile;
    myfile.open (file.c_str());
    myfile << std::setprecision(15);
    int i = 0;

    Eigen::Matrix<double,3,3> last_R = (*(allFrameHistory.begin()))->camToWorld.so3().matrix();
    Eigen::Matrix<double,3,1> last_T = (*(allFrameHistory.begin()))->camToWorld.translation().transpose();

    for(FrameShell* s : allFrameHistory)
    {
      if(!s->poseValid)
      {
        myfile<< last_R(0,0) <<" "<<last_R(0,1)<<" "<<last_R(0,2)<<" "<<last_T(0,0)<<" "<<
          last_R(1,0) <<" "<<last_R(1,1)<<" "<<last_R(1,2)<<" "<<last_T(1,0)<<" "<<
          last_R(2,0) <<" "<<last_R(2,1)<<" "<<last_R(2,2)<<" "<<last_T(2,0)<<"\n";
        continue;
      }

      if(setting_onlyLogKFPoses && s->marginalizedAt == s->id)
      {
        myfile<< last_R(0,0) <<" "<<last_R(0,1)<<" "<<last_R(0,2)<<" "<<last_T(0,0)<<" "<<
          last_R(1,0) <<" "<<last_R(1,1)<<" "<<last_R(1,2)<<" "<<last_T(1,0)<<" "<<
          last_R(2,0) <<" "<<last_R(2,1)<<" "<<last_R(2,2)<<" "<<last_T(2,0)<<"\n";
        continue;
      }

      const Eigen::Matrix<double,3,3> R = s->camToWorld.so3().matrix();
      const Eigen::Matrix<double,3,1> T = s->camToWorld.translation().transpose();

      last_R = R;
      last_T = T;

      myfile<< R(0,0) <<" "<<R(0,1)<<" "<<R(0,2)<<" "<<T(0,0)<<" "<<
        R(1,0) <<" "<<R(1,1)<<" "<<R(1,2)<<" "<<T(1,0)<<" "<<
        R(2,0) <<" "<<R(2,1)<<" "<<R(2,2)<<" "<<T(2,0)<<"\n";

      //		myfile << s->timestamp <<
      //			" " << s->camToWorld.translation().transpose()<<
      //			" " << s->camToWorld.so3().unit_quaternion().x()<<
      //			" " << s->camToWorld.so3().unit_quaternion().y()<<
      //			" " << s->camToWorld.so3().unit_quaternion().z()<<
      //			" " << s->camToWorld.so3().unit_quaternion().w() << "\n";
      i++;
    }
     myfile.close();
  }


  Vec5 FullSystem::trackNewCvo(// inputs: current l/r frames
                               FrameHessian* fh, FrameHessian* fh_right,
                               ImageAndExposure * img_left, ImageAndExposure * img_right, 
                               // outputs
                               std::vector<Pnt> & fhPtsWithDepth) {
    assert(allFrameHistory.size() > 1);

    // show original images
    for(IOWrap::Output3DWrapper* ow : outputWrapper)
    {
      ow->pushStereoLiveFrame(fh,fh_right);
    }

    // generate interest feature points of the new frame
    coarseInitializer->setFirstStereo(&Hcalib, fh,fh_right, img_left, img_right);
    std::vector<Pnt> ptsWithStaticDepth(coarseInitializer->points[0],
                                        coarseInitializer->points[0] + coarseInitializer->numPoints[0]);
    // last keyframe, current reference frame for tracker
    FrameHessian* lastRef = cvoTracker->getCurrRef();

    // outputs of CVO tracker 
    double achievedRes = std::numeric_limits<double>::max(); // a random init  val, used for outputs' residual
    AffLight aff_last_2_l = AffLight(0,0);
    float cvo_align_inner_product = 0;
    //float track_res = 0.0;
      
    // three outputs of the align
    SE3 lastRef_2_fh = SE3();
    AffLight aff_g2l = AffLight(0,0);
    Vec3 flowVec = Vec3(0.0,0.0,0.0); // output, used for the optical flow

    // we can use previous transformations to initialize the current relative transforamtion
    SE3 sprelast_2_slast;
    SE3 lastRef_2_slast; // pose of previous frame and the keyframe
    if (allFrameHistory.size() > 2) {
      FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];
      FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];
      {	// lock on global pose consistency!
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        sprelast_2_slast = sprelast->camToWorld.inverse() * slast->camToWorld;  // last in prelast
        lastRef_2_slast = lastRef->shell->camToWorld.inverse() * slast->camToWorld; // last Ref to last
        aff_last_2_l = slast->aff_g2l; // illumination. not useful yet for cvo tracking
      }

      lastRef_2_fh = isCvoSequential? sprelast_2_slast : lastRef_2_slast * sprelast_2_slast; // init value!
      
      //lastRef_2_fh = lastRef_2_slast;
      //std::cout<<"Using init value\n"
      //         <<"sprelast_2_last\n"<<sprelast_2_slast.matrix()
      //         <<"\nlastRef_2_slast \n"<<lastRef_2_slast.matrix()
      //         <<"\nlastRef_2_fh "<<lastRef_2_fh.matrix()
      //         <<"\n";
    } else {
      // first frame alignment
      lastRef_2_fh = SE3(Eigen::Matrix<double, 3, 3>::Identity(), Eigen::Matrix<double,3,1>::Zero() );
      lastRef_2_fh.translation()(2) = -0.75;
    }

    
    // setup stereo matching for each new pair of frames, to get the raw depth values
    bool isTrackingSuccessful = cvoTracker->trackNewestCvo(fh,
                                                           img_left,
                                                           ptsWithStaticDepth,
                                                           isCvoSequential,
                                                           lastRef_2_slast,
                                                           lastRef_2_fh, // ref to currs
                                                           achievedRes,
                                                           flowVec,
                                                           cvo_align_inner_product
                                                          ); //
      
    if (!isTrackingSuccessful) {
      printf("\nBIG ERROR! Cvo Tracking failed! Use some const motion guesses\n\n\n");
      std::vector<SE3> lastRef_2_fh_tries;
      lastRef_2_fh_tries.push_back(lastRef_2_slast * SE3::exp(sprelast_2_slast.log()*0.5)); // assume half motion.
      lastRef_2_fh_tries.push_back(lastRef_2_slast * sprelast_2_slast * SE3::exp(sprelast_2_slast.log()*0.5)); // assume1.5 motion.
      lastRef_2_fh_tries.push_back(lastRef_2_slast * sprelast_2_slast * sprelast_2_slast );	// assume double motion (frame skipped)
      lastRef_2_fh_tries.push_back(lastRef_2_slast); // assume zero motion.

      //lastRef_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.
      double min_residual = std::isnan(achievedRes) ? std::numeric_limits<double>::max() : achievedRes;
      if (std::isnan(cvo_align_inner_product)) cvo_align_inner_product = 0;
      for (auto && transform : lastRef_2_fh_tries) {
        SE3 curr_init_guess = isCvoSequential? lastRef_2_slast.inverse() * transform : transform;
        //double new_res = std::numeric_limits<double>::max();
        double new_res = 0;
        Vec3 new_flowVec;
        float new_cvo_inner_prod = 0;
        isTrackingSuccessful = cvoTracker->trackNewestCvo(fh,
                                                          img_left,
                                                          ptsWithStaticDepth,
                                                          isCvoSequential,
                                                          lastRef_2_slast,
                                                          curr_init_guess, // ref to currs
                                                          new_res,
                                                          new_flowVec,
                                                          new_cvo_inner_prod
                                                          ); //
      
        if (!isnan(new_cvo_inner_prod) && new_cvo_inner_prod > cvo_align_inner_product) {
        //if (!isnan(new_res)  && new_res < min_residual ){
          min_residual = new_res;
          lastRef_2_fh = curr_init_guess;
          std::cout<<"Use motion guess! residual is "<<min_residual<<", cvo inner product is "<<new_cvo_inner_prod<< "\n";
          achievedRes = new_res;
          flowVec = new_flowVec;
          cvo_align_inner_product = new_cvo_inner_prod;
	  if (isTrackingSuccessful)
            break;
        }
      }

    }
    // assign pose to the newly tracked frame
    // no lock required, as fh is not used anywhere yet.
    // here camToTrackingRef means T * point_cam  = point_world
    fh->shell->camToTrackingRef = lastRef_2_fh;
    fh->shell->trackingRef = lastRef->shell;
    fh->shell->aff_g2l = aff_g2l;
    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
    std::cout<<"lastF_2_fh is \n"<<lastRef_2_fh.translation()<<std::endl;

    Eigen::Matrix<double,3,1> last_T = fh->shell->camToWorld.translation().transpose();
    std::cout<<"Frame tracking location: x:"<<last_T(0,0)<<"y:"<<last_T(1,0)<<"z:"<<last_T(2,0)<<std::endl;

    fhPtsWithDepth = ptsWithStaticDepth;

    if (isCvoSequential)
      cvoTracker->setSequentialSource<Pnt>(fh, img_left, ptsWithStaticDepth);

    Vec5 results;
    results << achievedRes, flowVec[0], flowVec[1], flowVec[2],(double) cvo_align_inner_product;
    
    return results;
  }
  
  Vec4 FullSystem::trackNewCoarse(FrameHessian* fh, FrameHessian* fh_right)
  {

    assert(allFrameHistory.size() > 0);
    // set pose initialization.

    //    printf("the size of allFrameHistory is %d \n", (int)allFrameHistory.size());

    // show original images
    for(IOWrap::Output3DWrapper* ow : outputWrapper)
    {
      ow->pushStereoLiveFrame(fh,fh_right);
    }

    // latest keyframe
    FrameHessian* lastF = coarseTracker->lastRef;

    AffLight aff_last_2_l = AffLight(0,0);

    // const motion model candidates for initializing the pose of new frame
    std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;

    // for first two frames process differently
    if(allFrameHistory.size() == 2) {
      initializeFromInitializer(fh);

      lastF_2_fh_tries.push_back(SE3(Eigen::Matrix<double, 3, 3>::Identity(), Eigen::Matrix<double,3,1>::Zero() ));
 
      for(float rotDelta=0.02; rotDelta < 0.05; rotDelta = rotDelta + 0.02)
      {
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
      }

      coarseTracker->makeK(&Hcalib);
      coarseTracker->setCTRefForFirstFrame(frameHessians);

      lastF = coarseTracker->lastRef;
    }
    else
    {
      FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];
      FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];
      SE3 slast_2_sprelast;
      SE3 lastF_2_slast; // pose of previous frame
      {	// lock on global pose consistency!
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
        lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
        aff_last_2_l = slast->aff_g2l;
      }

      SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.


      // get last delta-movement.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);	// assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// assume double motion (frame skipped)
      lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast); // assume half motion.
      lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
      lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.

      /*        lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);

                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);
                lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() *  SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);

                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);
                lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() *  SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);

                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);
                lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() *  SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);*/

      // just try a TON of different initializations (all rotations). In the end,
      // if they don't work they will only be tried on the coarsest level, which is super fast anyway.
      // also, if tracking rails here we loose, so we really, really want to avoid that.
      for(float rotDelta=0.02; rotDelta < 0.02; rotDelta++)
      {
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
      }

      if(!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
      {
        lastF_2_fh_tries.clear();
        lastF_2_fh_tries.push_back(SE3());
      }
    }


    Vec3 flowVecs = Vec3(100,100,100);
    SE3 lastF_2_fh = SE3();
    AffLight aff_g2l = AffLight(0,0);


    // as long as maxResForImmediateAccept is not reached, I'll continue through the options.
    // I'll keep track of the so-far best achieved residual for each level in achievedRes.
    // If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.


    Vec5 achievedRes = Vec5::Constant(NAN);
    bool haveOneGood = false;
    int tryIterations=0;
    for(unsigned int i=0;i<lastF_2_fh_tries.size();i++)
    {
      AffLight aff_g2l_this = aff_last_2_l;
      SE3 lastF_2_fh_this = lastF_2_fh_tries[i];

      bool trackingIsGood = coarseTracker->trackNewestCoarse(
                                                             fh, lastF_2_fh_this, aff_g2l_this,
                                                             pyrLevelsUsed-1,
                                                             achievedRes);	// in each level has to be at least as good as the last try.
      tryIterations++;

      if(i != 0)
      {
        printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
               i,
               i, pyrLevelsUsed-1,
               aff_g2l_this.a,aff_g2l_this.b,
               achievedRes[0],
               achievedRes[1],
               achievedRes[2],
               achievedRes[3],
               achievedRes[4],
               coarseTracker->lastResiduals[0],
               coarseTracker->lastResiduals[1],
               coarseTracker->lastResiduals[2],
               coarseTracker->lastResiduals[3],
               coarseTracker->lastResiduals[4]);
      }


      // do we have a new winner?
      if(trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >=  achievedRes[0]))
      {
        printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
        flowVecs = coarseTracker->lastFlowIndicators;
        aff_g2l = aff_g2l_this;
        lastF_2_fh = lastF_2_fh_this;
        haveOneGood = true;
      }

      // take over achieved res (always).
      if(haveOneGood)
      {
        for(int i=0;i<5;i++)
        {
          if(!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i])	// take over if achievedRes is either bigger or NAN.
            achievedRes[i] = coarseTracker->lastResiduals[i];
        }
      }


      if(haveOneGood &&  achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
        break;

    }

    if(!haveOneGood)
    {
      printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
      flowVecs = Vec3(0,0,0);
      aff_g2l = aff_last_2_l;
      lastF_2_fh = lastF_2_fh_tries[0];
    }

    lastCoarseRMSE = achievedRes;

    // assign pose to the newly tracked frame
    // no lock required, as fh is not used anywhere yet.
    fh->shell->camToTrackingRef = lastF_2_fh.inverse();
    fh->shell->trackingRef = lastF->shell;
    fh->shell->aff_g2l = aff_g2l;
    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;


    Eigen::Matrix<double,3,1> last_T = fh->shell->camToWorld.translation().transpose();
    std::cout<<"x:"<<last_T(0,0)<<"y:"<<last_T(1,0)<<"z:"<<last_T(2,0)<<std::endl;

    if(coarseTracker->firstCoarseRMSE < 0)
      coarseTracker->firstCoarseRMSE = achievedRes[0];

    if(!setting_debugout_runquiet)
      printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);



    if(setting_logStuff)
    {
      (*coarseTrackingLog) << std::setprecision(16)
                           << fh->shell->id << " "
                           << fh->shell->timestamp << " "
                           << fh->ab_exposure << " "
                           << fh->shell->camToWorld.log().transpose() << " "
                           << aff_g2l.a << " "
                           << aff_g2l.b << " "
                           << achievedRes[0] << " "
                           << tryIterations << "\n";
    }


    return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
  }


  void FullSystem::stereoMatchReprojected(ImageAndExposure * img_left, ImageAndExposure * img_right,

                                          //outputs
                                          std::vector<CvoTrackingPoints,
                                          Eigen::aligned_allocator<CvoTrackingPoints>> & ptsStaticStereo
                                          ) {
    
    
  }

  void FullSystem::stereoMatch( ImageAndExposure* image, ImageAndExposure* image_right, int id, cv::Mat &idepthMap)
  {
    // =========================== add into allFrameHistory =========================
    FrameHessian* fh = new FrameHessian();
    FrameHessian* fh_right = new FrameHessian();
    FrameShell* shell = new FrameShell();
    shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
    shell->aff_g2l = AffLight(0,0);
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image->timestamp;
    shell->incoming_id = id; // id passed into DSO
    fh->shell = shell;
    fh_right->shell=shell;

    // =========================== make Images / derivatives etc. =========================
    fh->ab_exposure = image->exposure_time;
    fh->makeImages(image, &Hcalib);
    fh_right->ab_exposure = image_right->exposure_time;
    fh_right->makeImages(image_right,&Hcalib);

    Mat33f K = Mat33f::Identity();
    K(0,0) = Hcalib.fxl();
    K(1,1) = Hcalib.fyl();
    K(0,2) = Hcalib.cxl();
    K(1,2) = Hcalib.cyl();


    int counter = 0;

    // the first generation of immature points in fh (not fh_right)
    makeNewTraces(fh, fh_right, 0);

    unsigned  char * idepthMapPtr = idepthMap.data;

    // loop through all immature points from image, and trace it from image_right, so as to
    //obtain stereo depth
         //
    for(ImmaturePoint* ph : fh->immaturePoints)
    {
      ph->u_stereo = ph->u;
      ph->v_stereo = ph->v;
      ph->idepth_min_stereo = ph->idepth_min = 0;
      ph->idepth_max_stereo = ph->idepth_max = NAN;

      ImmaturePointStatus phTraceRightStatus = ph->traceStereo(fh_right, K, 1);

      // create the same immature points at right frame
      if(phTraceRightStatus == ImmaturePointStatus::IPS_GOOD)
      {
        ImmaturePoint* phRight = new ImmaturePoint(ph->lastTraceUV(0), ph->lastTraceUV(1), fh_right, &Hcalib );

        phRight->u_stereo = phRight->u;
        phRight->v_stereo = phRight->v;
        phRight->idepth_min_stereo = ph->idepth_min = 0;
        phRight->idepth_max_stereo = ph->idepth_max = NAN;
        
        ImmaturePointStatus  phTraceLeftStatus = phRight->traceStereo(fh, K, 0);

        float u_stereo_delta = abs(ph->u_stereo - phRight->lastTraceUV(0)); // reprojection-ed depth
        float depth = 1.0f/ph->idepth_stereo;

        if(phTraceLeftStatus == ImmaturePointStatus::IPS_GOOD && u_stereo_delta < 1 && depth > 0 && depth < 70)    //original u_stereo_delta 1 depth < 70
        {
          ph->idepth_min = ph->idepth_min_stereo;
          ph->idepth_max = ph->idepth_max_stereo;

          *((float *)(idepthMapPtr + int(ph->v) * idepthMap.step) + (int)ph->u *3) = ph->idepth_stereo;
          *((float *)(idepthMapPtr + int(ph->v) * idepthMap.step) + (int)ph->u *3 + 1) = ph->idepth_min;
          *((float *)(idepthMapPtr + int(ph->v) * idepthMap.step) + (int)ph->u *3 + 2) = ph->idepth_max;

          counter++;
        }
      }
    }

    //    std::sort(error.begin(), error.end());
    //    std::cout << 0.25 <<" "<<error[error.size()*0.25].first<<" "<<
    //              0.5 <<" "<<error[error.size()*0.5].first<<" "<<
    //              0.75 <<" "<<error[error.size()*0.75].first<<" "<<
    //              0.1 <<" "<<error.back().first << std::endl;

    //    for(int i = 0; i < error.size(); i++)
    //        std::cout << error[i].first << " " << error[i].second.first << " " << error[i].second.second << std::endl;

    std::cout<<" frameID " << id << " got good matches " << counter << std::endl;

    delete fh;
    delete fh_right;

    return;
  }

  // process nonkey frame to refine its key frame idepth
  void FullSystem::traceNewCoarseNonKey(FrameHessian *fh, FrameHessian *fh_right) {
    boost::unique_lock<boost::mutex> lock(mapMutex);

    // new idepth after refinement
    float idepth_min_update = 0;
    float idepth_max_update = 0;

    Mat33f K = Mat33f::Identity();
    K(0, 0) = Hcalib.fxl();
    K(1, 1) = Hcalib.fyl();
    K(0, 2) = Hcalib.cxl();
    K(1, 2) = Hcalib.cyl();

    Mat33f Ki = K.inverse();


    for (FrameHessian *host : frameHessians)        // go through all active frames
    {
      //		number++;
      int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

      // trans from reference keyframe to newest frame
      SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld; //  poinhts from host projectd on to the new frame
      // KRK-1
      Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
      // KRi
      Mat33f KRi = K * hostToNew.rotationMatrix().inverse().cast<float>();
      // Kt
      Vec3f Kt = K * hostToNew.translation().cast<float>();
      // t
      Vec3f t = hostToNew.translation().cast<float>();

      //aff
      Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

      for (ImmaturePoint *ph : host->immaturePoints)
      {
        // do temperol stereo match. IPS_GOOD are points with good tracing idepth interval
        ImmaturePointStatus phTrackStatus = ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);

        if (phTrackStatus == ImmaturePointStatus::IPS_GOOD)
        {
          ImmaturePoint *phNonKey = new ImmaturePoint(ph->lastTraceUV(0), ph->lastTraceUV(1), fh, &Hcalib);

          // project onto newest frame
          Vec3f ptpMin = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_min) + Kt;
          float idepth_min_project = 1.0f / ptpMin[2];
          Vec3f ptpMax = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_max) + Kt;
          float idepth_max_project = 1.0f / ptpMax[2];

          phNonKey->idepth_min = idepth_min_project;
          phNonKey->idepth_max = idepth_max_project;
          phNonKey->u_stereo = phNonKey->u;
          phNonKey->v_stereo = phNonKey->v;
          phNonKey->idepth_min_stereo = phNonKey->idepth_min;
          phNonKey->idepth_max_stereo = phNonKey->idepth_max;

          // do static stereo match from left image to right
          ImmaturePointStatus phNonKeyStereoStatus = phNonKey->traceStereo(fh_right, K, 1);

          if(phNonKeyStereoStatus == ImmaturePointStatus::IPS_GOOD)
          {
            ImmaturePoint* phNonKeyRight = new ImmaturePoint(phNonKey->lastTraceUV(0), phNonKey->lastTraceUV(1), fh_right, &Hcalib );

            phNonKeyRight->u_stereo = phNonKeyRight->u;
            phNonKeyRight->v_stereo = phNonKeyRight->v;
            phNonKeyRight->idepth_min_stereo = phNonKey->idepth_min;
            phNonKeyRight->idepth_max_stereo = phNonKey->idepth_max;

            // do static stereo match from right image to left
            ImmaturePointStatus  phNonKeyRightStereoStatus = phNonKeyRight->traceStereo(fh, K, 0);

            // change of u after two different stereo match
            float u_stereo_delta = abs(phNonKey->u_stereo - phNonKeyRight->lastTraceUV(0));
            float disparity = phNonKey->u_stereo - phNonKey->lastTraceUV[0];

            // free to debug the threshold
            if(u_stereo_delta > 1 && disparity < 10)
            {
              ph->lastTraceStatus = ImmaturePointStatus :: IPS_OUTLIER;
              continue;
            }
            else
            {
              // project back
              Vec3f pinverse_min = KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) / phNonKey->idepth_min_stereo - t);
              idepth_min_update = 1.0f / pinverse_min(2);

              Vec3f pinverse_max = KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) / phNonKey->idepth_max_stereo - t);
              idepth_max_update = 1.0f / pinverse_max(2);

              ph->idepth_min = idepth_min_update;
              ph->idepth_max = idepth_max_update;

              delete phNonKey;
              delete phNonKeyRight;
            }
          }
          else
          {
            delete phNonKey;
            continue;
          }

        }

        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) trace_good++;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) trace_oob++;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) trace_out++;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
        if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
        trace_total++;
      }
      host->numImmaturePointsCandidates = trace_good;
      printf("tracenewcoarseNonKey for %d: good %d, bad %d, oob %d, outlier %d, skipped %d, uninitialized %d \n", host->shell->incoming_id, trace_good, trace_badcondition, trace_oob, trace_out, trace_skip);

    }

  }


  //process keyframe
  void FullSystem::traceNewCoarseKey(FrameHessian* fh, FrameHessian* fh_right)
  {
    boost::unique_lock<boost::mutex> lock(mapMutex);

    int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

    Mat33f K = Mat33f::Identity();
    K(0,0) = Hcalib.fxl();
    K(1,1) = Hcalib.fyl();
    K(0,2) = Hcalib.cxl();
    K(1,2) = Hcalib.cyl();

    for(FrameHessian* host : frameHessians)		// go through all active frames
    {

      // trans from reference key frame to the newest one
      SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
      //KRK-1
      Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
      //Kt
      Vec3f Kt = K * hostToNew.translation().cast<float>();

      Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

      printf("\n\ntraceOn: host frame %d to curr frame %d", host->shell->incoming_id, fh->shell->incoming_id);
      // trace between the new keyframe andothe keyframes?
      for(ImmaturePoint* ph : host->immaturePoints)
      {
        
        ImmaturePointStatus phTrackStatus = ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );

        if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
        if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
        if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
        if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
        if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
        if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
        trace_total++;
      }
      host->numImmaturePointsCandidates = trace_good;
      printf("tracenewcoarseKey for %d: good %d, bad %d, oob %d, outlier %d, skipped %d, uninitialized %d \n\n\n", host->shell->incoming_id, trace_good, trace_badcondition, trace_oob, trace_out, trace_skip, trace_uninitialized);
    }

  }


  // convert immature points selected to activate to actual pointHessians
  void FullSystem::activatePointsMT_Reductor(
                                             std::vector<PointHessian*>* optimized,
                                             std::vector<ImmaturePoint*>* toOptimize,
                                             int min, int max, Vec10* stats, int tid)
  {
    ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
    for(int k=min;k<max;k++)
    {
      (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
    }
    delete[] tr;
  }

  void FullSystem::activatePointsMT()
  {
    
    if(ef->nPoints < setting_desiredPointDensity*0.66)   //setting_desiredPointDensity 是2000
      currentMinActDist -= 0.8;  //original 0.8
    if(ef->nPoints < setting_desiredPointDensity*0.8)
      currentMinActDist -= 0.5;  //original 0.5
    else if(ef->nPoints < setting_desiredPointDensity*0.9)
      currentMinActDist -= 0.2;  //original 0.2
    else if(ef->nPoints < setting_desiredPointDensity)
      currentMinActDist -= 0.1;  //original 0.1

    if(ef->nPoints > setting_desiredPointDensity*1.5)
      currentMinActDist += 0.8;
    if(ef->nPoints > setting_desiredPointDensity*1.3)
      currentMinActDist += 0.5;
    if(ef->nPoints > setting_desiredPointDensity*1.15)
      currentMinActDist += 0.2;
    if(ef->nPoints > setting_desiredPointDensity)
      currentMinActDist += 0.1;

    if(currentMinActDist < 0) currentMinActDist = 0;
    if(currentMinActDist > 4) currentMinActDist = 4;

    //if(!setting_debugout_runquiet)
    printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
           currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);



    FrameHessian* newestHs = frameHessians.back(); // latest new keyframe

    // make dist map.
    coarseDistanceMap->makeK(&Hcalib);
    coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

    //coarseTracker->debugPlotDistMap("distMap");

    std::vector<ImmaturePoint*> toOptimize; toOptimize.reserve(20000);


    // look at all immature points in all keyframes
    // to check if they can be converted into activa points
    for(FrameHessian* host : frameHessians)		// go through all active frames
    {
      if(host == newestHs) continue;

      SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
      Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
      Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());


      // for all immaturePoints in frameHessian
      int num_can_activate = 0, num_finite = 0;
      for(unsigned int i=0;i<host->immaturePoints.size();i+=1)
      {
        std::cout<<"Host id "<<host->shell->incoming_id;
        ImmaturePoint* ph = host->immaturePoints[i];
        ph->idxInImmaturePoints = i;

        // delete points that have never been traced successfully, or that are outlier on the last trace.
        if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
        {
          //				immature_invalid_deleted++;
          // remove point.
          delete ph;
          host->immaturePoints[i]=0;
          continue;
        }

        // can activate only if this is true.
        bool canActivate = (ph->lastTraceStatus == IPS_GOOD
                            || ph->lastTraceStatus == IPS_SKIPPED
                            || ph->lastTraceStatus == IPS_BADCONDITION
                            || ph->lastTraceStatus == IPS_OOB )
          && ph->lastTracePixelInterval < 8
                                           && ph->quality > setting_minTraceQuality
          && (ph->idepth_max+ph->idepth_min) > 0;
        

        // if I cannot activate the point, skip it. Maybe also delete it.
        if(!canActivate)
        {
          printf("host %d, Cannot activate, status is %d, lastTracePixelInterval is %f,  quality is %f , depthMax+depthMin is %f\n ", ph->host->shell->incoming_id, ph->lastTraceStatus, ph->lastTracePixelInterval, ph->quality, ph->idepth_max + ph->idepth_min);
          // if point will be out afterwards, delete it instead.
          if(ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
          {
            //					immature_notReady_deleted++;
            delete ph;
            host->immaturePoints[i]=0;
          }
          //				immature_notReady_skipped++;
          continue;
        }


        // see if we need to activate point due to distance map.
        Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
        int u = ptp[0] / ptp[2] + 0.5f;
        int v = ptp[1] / ptp[2] + 0.5f;

        if((u > 0 && v > 0 && u < wG[1] && v < hG[1]))
        {

          float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+wG[1]*v] + (ptp[0]-floorf((float)(ptp[0])));

          // push the good points int toOptimize
          if(dist>=currentMinActDist* ph->my_type)
          {
            coarseDistanceMap->addIntoDistFinal(u,v);
            toOptimize.push_back(ph);
            std::cout<<"Can activate!\n";
          } else {
            std::cout<<"should activate but currentMinActDist restrict it. Will stay in immature point this time \n";
            
          }
          num_can_activate ++;
        }
        else
        {
          delete ph;
          std::cout<<"Cannot activate! delte the immature point \n";
          host->immaturePoints[i]=0; //删除点的操作
        }
      }

      printf("Num of can_activate is %d, num finite is %d \n", num_can_activate, num_finite);
      
    }

    std::cout<<"Size of toOptimize is "<<toOptimize.size()<<std::endl;
    //printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
    //       (int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

    std::vector<PointHessian*> optimized; optimized.resize(toOptimize.size());

    if(multiThreading)
      treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);

    else
      activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);


    for(unsigned k=0;k<toOptimize.size();k++)
    {
      PointHessian* newpoint = optimized[k];
      ImmaturePoint* ph = toOptimize[k];

      // push back the point to the frame's active points, adn remove from immature points
      printf("toOptimize[%d], optimize[] is %p\n", k, (void*)newpoint);
      if(newpoint != 0 && newpoint != (PointHessian*)((long)(-1)))
      {
        printf("host %d, Insert active point, ", ph->host->shell->incoming_id);
        newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
        newpoint->host->pointHessians.push_back(newpoint);
        ef->insertPoint(newpoint);
        for(PointFrameResidual* r : newpoint->residuals)
          ef->insertResidual(r);
        assert(newpoint->efPoint != 0);
        delete ph;
        printf("active points length is %d\n", newpoint->host->pointHessians.size());
      }
      else if(newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB)
      {
        delete ph;
        ph->host->immaturePoints[ph->idxInImmaturePoints]=0;
      }
      else
      {
        assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
      }
    }


    for(FrameHessian* host : frameHessians)
    {
      for(int i=0;i<(int)host->immaturePoints.size();i++)
      {
        if(host->immaturePoints[i]==0)
        {
          host->immaturePoints[i] = host->immaturePoints.back();
          host->immaturePoints.pop_back();
          i--;
        }
      }
      printf("After activation, size of immature points on frame %d is %d\n", host->shell->incoming_id, host->immaturePoints.size());
    }


  }


  void FullSystem::activatePointsOldFirst()
  {
    assert(false);
  }

  void FullSystem::flagPointsForRemoval()
  {
    assert(EFIndicesValid);

    std::vector<FrameHessian*> fhsToKeepPoints;
    std::vector<FrameHessian*> fhsToMargPoints;

    //if(setting_margPointVisWindow>0)
    {
      for(int i=((int)frameHessians.size())-1;i>=0 && i >= ((int)frameHessians.size());i--)
        if(!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);

      for(int i=0; i< (int)frameHessians.size();i++)
        if(frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(frameHessians[i]);
    }



    //ef->setAdjointsF();
    //ef->setDeltaF(&Hcalib);
    int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;

    for (int m = 0; m < frameHessians.size() ; m++)
    //for(FrameHessian* host : frameHessians)		// go through all active frames
    {
      auto host = frameHessians[m];
      printf("host %d has active points %d\n", host->shell->incoming_id, host->pointHessians.size());
      for(unsigned int i=0;i<host->pointHessians.size();i++)
      {
        PointHessian* ph = host->pointHessians[i];
        if(ph==0) continue;

        if(ph->idepth_scaled < 0 || ph->residuals.size()==0)
        {
          host->pointHessiansOut.push_back(ph);
          ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
          host->pointHessians[i]=0;
          flag_nores++;
          printf("flagPointsForRemoval: host %d rm the active point, idpeth_scale<0 or  residual size zero\n", host->shell->incoming_id);
          host->pointHessians[i]=0;

        }
        else if(ph->isOOB(fhsToKeepPoints, fhsToMargPoints)  || host->flaggedForMarginalization)
        {
          flag_oob++;
          if(ph->isInlierNew())
          {
            flag_in++;
            int ngoodRes=0;
            for(PointFrameResidual* r : ph->residuals)
            {
              r->resetOOB();
              r->linearize(&Hcalib);
              r->efResidual->isLinearized = false;
              r->applyRes(true);
              if(r->efResidual->isActive())
              {
                r->efResidual->fixLinearizationF(ef);
                ngoodRes++;
              }
            }
            if(ph->idepth_hessian > setting_minIdepthH_marg)
            {
              flag_inin++;
              ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
              host->pointHessiansMarginalized.push_back(ph);
            }
            else
            {
              ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
              host->pointHessiansOut.push_back(ph);
            }


          }
          else
          {
            host->pointHessiansOut.push_back(ph);
            ph->efPoint->stateFlag = EFPointStatus::PS_DROP;


            //printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
          }
          printf("flagPointsForRemoval: host %d rm the active point OOB\n", host->shell->incoming_id);
          host->pointHessians[i]=0;
        }


      }


      for(int i=0;i<(int)host->pointHessians.size();i++)
      {
        if(host->pointHessians[i]==0)
        {
          host->pointHessians[i] = host->pointHessians.back();
          host->pointHessians.pop_back();
          i--;
        }
      }
    }

  }

  void FullSystem::recordPcds(ImageAndExposure * image, ImageAndExposure * image_right, int id) {


    // =========================== add into allFrameHistory =========================
    FrameHessian* fh = new FrameHessian();
    FrameHessian* fh_right = new FrameHessian();
    FrameShell* shell = new FrameShell();
    shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
    shell->aff_g2l = AffLight(0,0);
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image->timestamp;
    shell->incoming_id = id; // id passed into DSO
    fh->shell = shell;
    fh_right->shell=shell;
    allFrameHistory.push_back(shell);


    // =========================== make Images keypoints / derivatives etc. =========================
    fh->ab_exposure = image->exposure_time;
    fh->makeImages(image, &Hcalib);
    fh_right->ab_exposure = image_right->exposure_time;
    fh_right->makeImages(image_right,&Hcalib);

    coarseInitializer->setFirstStereo(&Hcalib, fh,fh_right, image, image_right);
    std::vector<Pnt> ptsWithStaticDepth(coarseInitializer->points[0],
                                        coarseInitializer->points[0] + coarseInitializer->numPoints[0]);

    std::vector<Pnt> remains;
    for (auto &&p : ptsWithStaticDepth ) {
      if (p.idepth <0.011)
        continue;
      remains.push_back(p);
    }

    std::string filename( "cvo_points/" +  std::to_string(id) + ".txt" );
    write_cvo_pointcloud_to_file<Pnt>(filename, remains );

    std::string pcl_file("cvo_points_pcd/" + std::to_string(id) + ".pcd");
    save_points_as_color_pcd<Pnt>(pcl_file, remains );

    delete fh;
    delete fh_right;
  }


  void FullSystem::addActiveFrame( ImageAndExposure* image, ImageAndExposure* image_right, int id )
  {

    if(isLost) return;
    boost::unique_lock<boost::mutex> lock(trackMutex);


    // =========================== add into allFrameHistory =========================
    FrameHessian* fh = new FrameHessian();
    FrameHessian* fh_right = new FrameHessian();
    FrameShell* shell = new FrameShell();
    shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
    shell->aff_g2l = AffLight(0,0);
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image->timestamp;
    shell->incoming_id = id; // id passed into DSO
    fh->shell = shell;
    fh_right->shell=shell;
    fh_right->w = image->w;
    fh_right->h = image->h;
    allFrameHistory.push_back(shell);


    // =========================== make Images keypoints / derivatives etc. =========================
    fh->ab_exposure = image->exposure_time;
    fh->makeImages(image, &Hcalib);
    fh_right->ab_exposure = image_right->exposure_time;
    fh_right->makeImages(image_right,&Hcalib);

    //if (false)
    if(!initialized)
    {
      // use initializer!
      if(coarseInitializer->frameID<0)	// first frame set. fh is kept by coarseInitializer.
      {
        // init CvoTracker
        cvoTracker->setIntrinsic(Hcalib);
        coarseInitializer->setFirstStereo(&Hcalib, fh,fh_right, image, image_right);
        std::vector<Pnt> ptsWithStaticDepth(coarseInitializer->points[0],
                                            coarseInitializer->points[0] + coarseInitializer->numPoints[0]);
        cvoTracker->setCurrRef(fh, image, ptsWithStaticDepth);
        if (isCvoSequential) {
          cvoTracker->setSequentialSource(fh, image, ptsWithStaticDepth);
        }
        // init actual selected pointHessians for the window optimization
        initializeFromInitializer(fh);
        initialized=true;
      }
      return;
    }
    else	// do front-end operation.
    {
      // =========================== SWAP tracking reference?. =========================
      if(!useCvo && coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
      {
        boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
        CoarseTracker* tmp = coarseTracker; 
        coarseTracker=coarseTracker_forNewKF; 
        coarseTracker_forNewKF=tmp;
      } else if (useCvo && cvoTracker_forNewKF->getCurrRefId() > cvoTracker->getCurrRefId())
      {
        //boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
        //CvoTracker* tmp = cvoTracker; 
        //cvoTracker=cvoTracker_forNewKF; 
        //cvoTracker_forNewKF=tmp;
      }

      // track the new l/r frames
      Vec5 tres = Vec5::Zero();
      std::vector<Pnt> newPtsWithStaticDepth;
      if (useCvo) 
        tres = trackNewCvo(fh, fh_right,  image, image_right, newPtsWithStaticDepth);
      else
        tres.segment(0,4) = trackNewCoarse(fh,fh_right);
        
      if(!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
      {
        printf("Initial Tracking failed: LOST!\n");
        isLost=true;
        return;
      }

      // decide if a new keyframe is needed
      bool needToMakeKF = false;
      if(setting_keyframesPerSecond > 0)
      { // min time interval for a new keyframe
        needToMakeKF = allFrameHistory.size()== 1 ||
          (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f/setting_keyframesPerSecond;
      }
      else
      {
        Vec2 refToFh;
        if (useCvo)
          refToFh  << 1.0, 0.0; // e^0, 00
        else
          refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
                                                coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);
          
        // obtain the ref to current frame illumination model
        float delta =         
          setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]) +
            setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]) +
            setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]) +
            setting_kfGlobalWeight*setting_maxAffineWeight * fabs(logf((float)refToFh[0]));
        printf(" delta is %f, t %f, r %f, rt %f, aff %f \n", delta, tres[1], tres[2], tres[3], refToFh[0]);

        if (useCvo ) {
          // if (isCvoInnerProd) {
          //  delta =tres(4);
          //  needToMakeKF = allFrameHistory.size()== 1 || delta < 0.004;
          //  std::cout<<"delta is "<<delta<<std::endl;
          //} else
          //needToMakeKF = allFrameHistory.size()== 1 || delta> 0.26;
          delta = tres[4];
          needToMakeKF = allFrameHistory.size() == 1 || delta < setting_CvoKeyframeInnerProduct;
          //needToMakeKF = allFrameHistory.size() == 1 || delta < 0.0019;
        } else {

        // the change of optical flow (from tracker::tracknewest)
        // TODO: how is change of flow computed, in CVO??
            
          needToMakeKF = allFrameHistory.size()== 1 || delta > 0.6 || 2*coarseTracker->firstCoarseRMSE < tres[0];
        }

        std::cout<<"NeedToMakekf is "<<needToMakeKF<<std::endl;
        if (needToMakeKF && useCvo ) {// && !isCvoSequential) {
          cvoTracker->setCurrRef(fh, image, newPtsWithStaticDepth);
        }
      }

      for(IOWrap::Output3DWrapper* ow : outputWrapper)
        ow->publishCamPose(fh->shell, &Hcalib);

      lock.unlock();
        
      // based on the decision if we need keyframe, deal with this tracked frame
      deliverTrackedFrame(fh, fh_right, needToMakeKF);
      std::cout<<"Just delivered tracked frame. isKF is "<<needToMakeKF<<"\n";
      return;
    }
  }

  void FullSystem::deliverTrackedFrame(FrameHessian* fh, FrameHessian* fh_right, bool needKF)
  {

    if(linearizeOperation)
    {
      std::cout<<"linearOperation\n";
      // TODO: CVO
      if(goStepByStep &&
         (useCvo && lastRefStopID != coarseTracker->refFrameID || 
          !useCvo && lastRefStopID != cvoTracker->getCurrRefId() ) )
      {  
        /*
        MinimalImageF3 img(wG[0], hG[0], fh->dI, 3);
        IOWrap::displayImage("frameToTrack", &img);
        while(true)
        {
          char k=IOWrap::waitKey(0);
          if(k==' ')
            break;
          handleKey( k );
        }
        */
        lastRefStopID = useCvo? cvoTracker->getCurrRefId() : coarseTracker->refFrameID;
      }
      else
        handleKey( IOWrap::waitKey(1) );


      if(needKF) {
        makeKeyFrame(fh, fh_right);

      }
      else makeNonKeyFrame(fh, fh_right);
    }
    else
    {
      boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
      unmappedTrackedFrames.push_back(fh);
      unmappedTrackedFrames_right.push_back(fh_right);
      if(needKF) needNewKFAfter=fh->shell->trackingRef->id;
      trackedFrameSignal.notify_all();

      // TOOD: cvo tracker
      if (useCvo) {
        while(cvoTracker_forNewKF->getCurrRefId() == -1 && cvoTracker->getCurrRefId() == -1 )
          mappedFrameSignal.wait(lock);
      } else {
        while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 )
          mappedFrameSignal.wait(lock);
      }
      lock.unlock();
    }
  }

  void FullSystem::mappingLoop()
  {
    boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

    while(runMapping)
    {
      while(unmappedTrackedFrames.size()==0)
      {
        trackedFrameSignal.wait(lock);
        if(!runMapping) return;
      }

      FrameHessian* fh = unmappedTrackedFrames.front();
      unmappedTrackedFrames.pop_front();
      FrameHessian* fh_right = unmappedTrackedFrames_right.front();
      unmappedTrackedFrames_right.pop_front();


      // guaranteed to make a KF for the very first two tracked frames.
      if(allKeyFramesHistory.size() <= 2)
      {
        lock.unlock();
        makeKeyFrame(fh, fh_right);
        lock.lock();
        mappedFrameSignal.notify_all();
        continue;
      }

      if(unmappedTrackedFrames.size() > 3)
        needToKetchupMapping=true;

      if(unmappedTrackedFrames.size() > 0) // if there are other frames to track, do that first.
      {
        lock.unlock();
        makeNonKeyFrame(fh, fh_right);
        lock.lock();

        if(needToKetchupMapping && unmappedTrackedFrames.size() > 0)
        {
          FrameHessian* fh = unmappedTrackedFrames.front();
          unmappedTrackedFrames.pop_front();
          {
            boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
            assert(fh->shell->trackingRef != 0);
            fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
            fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
          }
          delete fh;
          delete fh_right;
        }

      }
      else
      {
        if(setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)
        {
          lock.unlock();
          makeKeyFrame(fh, fh_right);
          needToKetchupMapping=false;
          lock.lock();
        }
        else
        {
          lock.unlock();
          makeNonKeyFrame(fh, fh_right);
          lock.lock();
        }
      }
      mappedFrameSignal.notify_all();
    }
    printf("MAPPING FINISHED!\n");
  }

  void FullSystem::blockUntilMappingIsFinished()
  {
    boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
    runMapping = false;
    trackedFrameSignal.notify_all();
    lock.unlock();

    mappingThread.join();

  }

  void FullSystem::makeNonKeyFrame( FrameHessian* fh, FrameHessian* fh_right)
  {
    // needs to be set by mapping thread. no lock required since we are in mapping thread.
    {
      boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
      assert(fh->shell->trackingRef != 0);
      fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
      fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
    }

    traceNewCoarseNonKey(fh, fh_right);

    if (useCvo && isCvoSequential) {
      if (lastFrame) delete lastFrame;
      lastFrame = fh;
    } else
      delete fh;
    delete fh_right;

    //if (useCvo && cvoTracker->getCurrRef()->immaturePoints.size()) {
    //  cvoTracker->setCurrRefPts(cvoTracker->getCurrRef()->immaturePoints);
    //  std::cout<<"Currref frame id "<< cvoTracker->getCurrRefId()<<", cvo->setCurrRefPts size "<<cvoTracker->getCurrRef()->immaturePoints.size()<<std::endl;
    //}
  }

  void FullSystem::makeKeyFrame( FrameHessian* fh, FrameHessian* fh_right)
  {
    // needs to be set by mapping thread
    {
      boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
      assert(fh->shell->trackingRef != 0);
      fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
      fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
    }

    // trace bewteen all keyframes and the current left frame to refine immature points
    traceNewCoarseKey(fh, fh_right);
    
    boost::unique_lock<boost::mutex> lock(mapMutex);
    //if (useCvo && cvoTracker->getCurrRef()->immaturePoints.size()) {
    //  cvoTracker->setCurrRefPts(cvoTracker->getCurrRef()->immaturePoints);
    //  std::cout<<"Currref frame id "<< cvoTracker->getCurrRefId()<<"cvo->setCurrRefPts size "<<cvoTracker->getCurrRef()->immaturePoints.size()<<std::endl;
    //}

    // =========================== Flag Frames to be Marginalized. =========================
    flagFramesForMarginalization(fh);


    // =========================== add New Frame to Hessian Struct. =========================
    fh->idx = frameHessians.size();
    frameHessians.push_back(fh);

    fh->frameID = allKeyFramesHistory.size();
    allKeyFramesHistory.push_back(fh->shell);
    ef->insertFrame(fh, &Hcalib);
    // set up illumination, 
    setPrecalcValues();


    // =========================== add new residuals for old points =========================
    // go through all active frames' all active points
    // add the residual of these active points between their host frames and the current new keyframe
    int numFwdResAdde=0;
    for(FrameHessian* fh1 : frameHessians)	
    {
      if(fh1 == fh) continue;
      for(PointHessian* ph : fh1->pointHessians)
      {
        PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
        r->setState(ResState::IN);
        ph->residuals.push_back(r);
        ef->insertResidual(r);
        ph->lastResiduals[1] = ph->lastResiduals[0];
        ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
        numFwdResAdde+=1;
      }
    }


    // =========================== Activate Points that are marked as activate but is not actiate previously  (& flag for marginalization). =========================
    // conver the immature points to active point sif the traceOn function retun a good inbound status
    // construc tthe residual between these new points and all other frames in teh sliding window
    
    activatePointsMT();
    ef->makeIDX();
    //TOOD: set curr ref point in CVO

    // =========================== OPTIMIZE ALL =========================
    fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
    float rmse = optimize(setting_maxOptIterations);
    //printf("allKeyFramesHistory size is %d \n", (int)allKeyFramesHistory.size());
    printf("[FullSystem::MakeKeyframe] rmse is %f, benchmark_initializerSlackfactor is %f \n", rmse, benchmark_initializerSlackFactor);


    for(FrameHessian* fh1 : frameHessians)	
    {
      printf("The number of active points in keyframe id %d is %d\n", fh1->shell->incoming_id, fh1->pointHessians.size());

    }


    // =========================== Figure Out if INITIALIZATION FAILED =========================
    if(allKeyFramesHistory.size() <= 4)
    {
      if(allKeyFramesHistory.size()==2 &&
        rmse > 30*benchmark_initializerSlackFactor)

        //rmse > 20*benchmark_initializerSlackFactor)
      {
        printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
        initFailed=true;
      }
      if(allKeyFramesHistory.size()==3 &&
         rmse > 20*benchmark_initializerSlackFactor)
        // rmse > 13*benchmark_initializerSlackFactor)
      {
        printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
        initFailed=true;
      }
      if(allKeyFramesHistory.size()==4 &&
         rmse > 13*benchmark_initializerSlackFactor)
        //rmse > 9*benchmark_initializerSlackFactor)
      {
        printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
        initFailed=true;
      }
    }

    if(isLost) return;
	
	
    // =========================== REMOVE OUTLIER after optimization  =========================
    removeOutliers();

    {
      // TOOD:CVO
      if (useCvo) {
        boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
        //cvoTracker_forNewKF->makeK(&Hcalib);
        //cvoTracker_forNewKF->setCoarseTrackingRef(frameHessians, fh_right, Hcalib);
        // 
        cvoTracker_forNewKF->setCurrRef<CvoTrackingPoints>(frameHessians.back(),nullptr,  cvoTracker->getRefPointsWithDepth());
        
      } else {
        boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
        coarseTracker_forNewKF->makeK(&Hcalib);
        coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians, fh_right, Hcalib);
		
        coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
        coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
      }
    }


    debugPlot("post Optimize");
    if (useCvo) {
      //cvoTracker->setCurrRefPts(fh->pointHessians);
      //cvoTracker->setCurrRefPts(immaturePoints, pointHessians);
    }



    // =========================== (Activate-)Marginalize Points =========================
    flagPointsForRemoval();
    ef->dropPointsF();
    getNullspaces(
                  ef->lastNullspaces_pose,
                  ef->lastNullspaces_scale,
                  ef->lastNullspaces_affA,
                  ef->lastNullspaces_affB);
    ef->marginalizePointsF();
    for(FrameHessian* fh1 : frameHessians)	
    {
      printf("After marginalization, The number of active points in keyframe id %d is %d\n", fh1->shell->incoming_id, fh1->pointHessians.size());

    }



    // =========================== add new Immature points & new residuals =========================
    // jsut find all high gradient pooints again and convert them to immature points.
    // for the new frame only
    makeNewTraces(fh, fh_right, 0);



    for(IOWrap::Output3DWrapper* ow : outputWrapper)
    {
      ow->publishGraph(ef->connectivityMap);
      ow->publishKeyframes(frameHessians, false, &Hcalib);
    }



    // =========================== Marginalize Frames =========================

    for(unsigned int i=0;i<frameHessians.size();i++)
    {
      if(frameHessians[i]->flaggedForMarginalization)
      {
        std::cout<<"Marginalizing frame "<<frameHessians[i]->shell->incoming_id<<std::endl;
        marginalizeFrame(frameHessians[i]); i=0;
      }
    }

    delete fh_right;


    //	printLogLine();
    //    printEigenValLine();

  }

  // insert the first Frame into FrameHessians of the sliding window optimizer
  // assume it is already been coarseInitialized
  // use stereomatch to init from static stereo
  void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
  {
    boost::unique_lock<boost::mutex> lock(mapMutex);


    Mat33f K = Mat33f::Identity();
    K(0,0) = Hcalib.fxl();
    K(1,1) = Hcalib.fyl();
    K(0,2) = Hcalib.cxl();
    K(1,2) = Hcalib.cyl();


    // add firstframe.
    FrameHessian* firstFrame = coarseInitializer->firstFrame;
    firstFrame->idx = frameHessians.size();
    frameHessians.push_back(firstFrame);
    firstFrame->frameID = allKeyFramesHistory.size();
    allKeyFramesHistory.push_back(firstFrame->shell);
    ef->insertFrame(firstFrame, &Hcalib);
    setPrecalcValues();

    FrameHessian* firstFrameRight = coarseInitializer->firstRightFrame;
    frameHessiansRight.push_back(firstFrameRight);

    firstFrame->pointHessians.reserve(wG[0]*hG[0]*0.2f);
    firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f);
    firstFrame->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f);

    float idepthStereo = 0;
    float sumID=1e-5, numID=1e-5;
    for(int i=0;i<coarseInitializer->numPoints[0];i++)
    {
      sumID += coarseInitializer->points[0][i].iR;
      numID++;
    }

    // randomly sub-select the points I need.
    float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

    if(!setting_debugout_runquiet)
      printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
             (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0] );

    // initialize first frame by idepth computed by static stereo matching
    for(int i=0;i<coarseInitializer->numPoints[0];i++)
    {
      // sample
      if(rand()/(float)RAND_MAX > keepPercentage) continue;


      Pnt* point = coarseInitializer->points[0]+i;      // zhuang bi
      ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f,point->v+0.5f,firstFrame,point->my_type, &Hcalib);

      pt->u_stereo = pt->u;
      pt->v_stereo = pt->v;
      pt->idepth_min_stereo = 0;
      pt->idepth_max_stereo = NAN;

      pt->traceStereo(firstFrameRight, K, 1);

      pt->idepth_min = pt->idepth_min_stereo;
      pt->idepth_max = pt->idepth_max_stereo;
      idepthStereo = pt->idepth_stereo;

      // filter out bad points
      if(!std::isfinite(pt->energyTH) || !std::isfinite(pt->idepth_min) || !std::isfinite(pt->idepth_max)
         || pt->idepth_min < 0 || pt->idepth_max < 0)
      {
        delete pt;
        continue;

      }

      PointHessian* ph = new PointHessian(pt, &Hcalib);
      delete pt;
      if(!std::isfinite(ph->energyTH)) {delete ph; continue;}

      ph->setIdepthScaled(idepthStereo);
      ph->setIdepthZero(idepthStereo);
      ph->hasDepthPrior=true;
      ph->setPointStatus(PointHessian::ACTIVE);


      firstFrame->pointHessians.push_back(ph);
      ef->insertPoint(ph);
    }

    SE3 firstToNew = coarseInitializer->thisToNext;

    // really no lock required, as we are initializing.
    {
      boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
      firstFrame->shell->camToWorld = SE3();
      firstFrame->shell->aff_g2l = AffLight(0,0);
      firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),firstFrame->shell->aff_g2l);
      firstFrame->shell->trackingRef=0;
      firstFrame->shell->camToTrackingRef = SE3();

      newFrame->shell->camToWorld = firstToNew.inverse();
      newFrame->shell->aff_g2l = AffLight(0,0);
      newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),newFrame->shell->aff_g2l);
      newFrame->shell->trackingRef = firstFrame->shell;
      newFrame->shell->camToTrackingRef = firstToNew.inverse();

    }

    initialized=true;
    printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
  }


  // generate immatrue points according to the heatmap from Pixe
  // call it after optimization, outlider removal, and point marginalizaiton
  void FullSystem::makeNewTraces(FrameHessian* newFrame, FrameHessian* newFrameRight, float* gtDepth)
  {
    pixelSelector->allowFast = true;
    //int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
    int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,setting_desiredImmatureDensity);

    newFrame->pointHessians.reserve(numPointsTotal*1.2f);
    //fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
    newFrame->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
    newFrame->pointHessiansOut.reserve(numPointsTotal*1.2f);

    for(int y=patternPadding+1;y<hG[0]-patternPadding-2;y++)
      for(int x=patternPadding+1;x<wG[0]-patternPadding-2;x++)
      {
        int i = x+y*wG[0];
        if(selectionMap[i]==0) continue;

        ImmaturePoint* impt = new ImmaturePoint(x,y,newFrame, selectionMap[i], &Hcalib);

        if(!std::isfinite(impt->energyTH)) delete impt;
        else newFrame->immaturePoints.push_back(impt);

      }
    printf("MADE %d IMMATURE POINTS from frame %d!\n", (int)newFrame->immaturePoints.size(), newFrame->shell->incoming_id);

  }


  void FullSystem::setPrecalcValues()
  {
    for(FrameHessian* fh : frameHessians)
    {
      fh->targetPrecalc.resize(frameHessians.size());
      for(unsigned int i=0;i<frameHessians.size();i++)
        fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
    }

    ef->setDeltaF(&Hcalib);
  }

  void FullSystem::printLogLine()
  {
    if(frameHessians.size()==0) return;

    if(!setting_debugout_runquiet)
      printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
             allKeyFramesHistory.back()->id,
             statistics_lastFineTrackRMSE,
             ef->resInA,
             ef->resInL,
             ef->resInM,
             (int)statistics_numForceDroppedResFwd,
             (int)statistics_numForceDroppedResBwd,
             allKeyFramesHistory.back()->aff_g2l.a,
             allKeyFramesHistory.back()->aff_g2l.b,
             frameHessians.back()->shell->id - frameHessians.front()->shell->id,
             (int)frameHessians.size());


    if(!setting_logStuff) return;

    if(numsLog != 0)
    {
      (*numsLog) << allKeyFramesHistory.back()->id << " "  <<
        statistics_lastFineTrackRMSE << " "  <<
        (int)statistics_numCreatedPoints << " "  <<
        (int)statistics_numActivatedPoints << " "  <<
        (int)statistics_numDroppedPoints << " "  <<
        (int)statistics_lastNumOptIts << " "  <<
        ef->resInA << " "  <<
        ef->resInL << " "  <<
        ef->resInM << " "  <<
        statistics_numMargResFwd << " "  <<
        statistics_numMargResBwd << " "  <<
        statistics_numForceDroppedResFwd << " "  <<
        statistics_numForceDroppedResBwd << " "  <<
        frameHessians.back()->aff_g2l().a << " "  <<
        frameHessians.back()->aff_g2l().b << " "  <<
        frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "  <<
        (int)frameHessians.size() << " "  << "\n";
      numsLog->flush();
    }


  }



  void FullSystem::printEigenValLine()
  {
    if(!setting_logStuff) return;
    if(ef->lastHS.rows() < 12) return;


    MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
    MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
    int n = Hp.cols()/8;
    assert(Hp.cols()%8==0);

    // sub-select
    for(int i=0;i<n;i++)
    {
      MatXX tmp6 = Hp.block(i*8,0,6,n*8);
      Hp.block(i*6,0,6,n*8) = tmp6;

      MatXX tmp2 = Ha.block(i*8+6,0,2,n*8);
      Ha.block(i*2,0,2,n*8) = tmp2;
    }
    for(int i=0;i<n;i++)
    {
      MatXX tmp6 = Hp.block(0,i*8,n*8,6);
      Hp.block(0,i*6,n*8,6) = tmp6;

      MatXX tmp2 = Ha.block(0,i*8+6,n*8,2);
      Ha.block(0,i*2,n*8,2) = tmp2;
    }

    VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
    VecX eigenP = Hp.topLeftCorner(n*6,n*6).eigenvalues().real();
    VecX eigenA = Ha.topLeftCorner(n*2,n*2).eigenvalues().real();
    VecX diagonal = ef->lastHS.diagonal();

    std::sort(eigenvaluesAll.data(), eigenvaluesAll.data()+eigenvaluesAll.size());
    std::sort(eigenP.data(), eigenP.data()+eigenP.size());
    std::sort(eigenA.data(), eigenA.data()+eigenA.size());

    int nz = std::max(100,setting_maxFrames*10);

    if(eigenAllLog != 0)
    {
      VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
      (*eigenAllLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
      eigenAllLog->flush();
    }
    if(eigenALog != 0)
    {
      VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
      (*eigenALog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
      eigenALog->flush();
    }
    if(eigenPLog != 0)
    {
      VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
      (*eigenPLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
      eigenPLog->flush();
    }

    if(DiagonalLog != 0)
    {
      VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
      (*DiagonalLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
      DiagonalLog->flush();
    }

    if(variancesLog != 0)
    {
      VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
      (*variancesLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
      variancesLog->flush();
    }

    std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
    (*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
    for(unsigned int i=0;i<nsp.size();i++)
      (*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " " ;
    (*nullspacesLog) << "\n";
    nullspacesLog->flush();

  }

  void FullSystem::printFrameLifetimes()
  {
    if(!setting_logStuff) return;


    boost::unique_lock<boost::mutex> lock(trackMutex);

    std::ofstream* lg = new std::ofstream();
    lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
    lg->precision(15);

    for(FrameShell* s : allFrameHistory)
    {
      (*lg) << s->id
            << " " << s->marginalizedAt
            << " " << s->statistics_goodResOnThis
            << " " << s->statistics_outlierResOnThis
            << " " << s->movedByOpt;



      (*lg) << "\n";
    }


    lg->close();
    delete lg;

  }


  void FullSystem::printEvalLine()
  {
    return;
  }





}
