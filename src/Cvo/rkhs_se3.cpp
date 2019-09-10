/* ----------------------------------------------------------------------------
 * Copyright 2019, Tzu-yuan Lin <tzuyuan@umich.edu>, Maani Ghaffari <maanigj@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   rkhs_se3.cpp
 *  @author Tzu-yuan Lin, Maani Ghaffari 
 *  @brief  Source file for contineuous visual odometry rkhs_se3 registration
 *  @date   August 15, 2019
 **/

#include "rkhs_se3.hpp"
#include "util/settings.h"

using namespace std;

namespace cvo{

  rkhs_se3::rkhs_se3():
    // initialize parameters
    init(false),           // initialization indicator
    ptr_fixed_fr(new frame),
    ptr_moving_fr(new frame),
    ptr_fixed_pcd(new point_cloud),
    ptr_moving_pcd(new point_cloud),
    ell(0.15*7),             // kernel characteristic length-scale
    sigma(0.1),            // kernel signal variance (set as std)      
    sp_thres(1e-3),        // kernel sparsification threshold       
    c(7.0),                // so(3) inner product scale     
    d(7.0),                // R^3 inner product scale
    color_scale(1.0e-5),   // color space inner product scale
    c_ell(200),             // kernel characteristic length-scale for color kernel
    c_sigma(1),
    s_ell(0.1),
    s_sigma(1),
    MAX_ITER(2000),        // maximum number of iteration
    min_step(2*1.0e-1),    // minimum integration step
    eps(5*1.0e-5),         // threshold for stopping the function
    eps_2(1.0e-5),         // threshold for se3 distance
    R(Eigen::Matrix3f::Identity(3,3)), // initialize rotation matrix to I
    T(Eigen::Vector3f::Zero()),        // initialize translation matrix to zeros
    transform(Eigen::Affine3f::Identity()),    // initialize transformation to I
    prev_transform(Eigen::Affine3f::Identity()),
    accum_tf(Eigen::Affine3f::Identity()),
    accum_tf_vis(Eigen::Affine3f::Identity())
  {
  }

  rkhs_se3::~rkhs_se3(){
  }

  inline Eigen::VectorXcf rkhs_se3::poly_solver(const Eigen::VectorXf& coef){
    // extract order
    int order = coef.size()-1;
    Eigen::VectorXcf roots;
    
    // create M = diag(ones(n-1,1),-1)
    Eigen::MatrixXf M = Eigen::MatrixXf::Zero(order,order);
    M.bottomLeftCorner(order-1,order-1) = Eigen::MatrixXf::Identity(order-1,order-1);
    
    // M(1,:) = -p(2:n+1)./p(1)
    M.row(0) = -(coef/coef(0)).segment(1,order).transpose();

    // eigen(M) and get the answer
    roots = M.eigenvalues();

    return roots;
  }

  inline float rkhs_se3::dist_se3(const Eigen::Matrix3f& R, const Eigen::Vector3f& T){
    // create transformation matrix
    Eigen::Matrix4f temp_transform = Eigen::Matrix4f::Identity();
    temp_transform.block<3,3>(0,0)=R;
    temp_transform.block<3,1>(0,3)=T;
    
    // distance = frobenius_norm(logm(trans))
    float d = temp_transform.log().norm();
    
    return d;
  }

  inline void rkhs_se3::update_tf(){
    // transform = [R', -R'*T; 0,0,0,1]
    transform.matrix().block<3,3>(0,0) = R.transpose();
    transform.matrix().block<3,1>(0,3) = -R.transpose()*T;
  }


  inline float rkhs_se3::color_kernel(const int i, const int j){
    Eigen::Matrix<float,3,1> RGB_x = ptr_fixed_pcd->RGB.row(i).transpose();
    Eigen::Matrix<float,3,1> RGB_y = ptr_moving_pcd->RGB.row(j).transpose();

    return((RGB_x-RGB_y).squaredNorm());
  }



  void rkhs_se3::se_kernel(const float l, const float s2){
    A_trip_concur.clear();
    // convert k threshold to d2 threshold (so that we only need to calculate k when needed)
    float d2_thres = -2.0*l*l*log(sp_thres/s2);
    float d2_c_thres = -2.0*c_ell*c_ell*log(sp_thres/c_sigma/c_sigma);

    /** 
     * kdtreeeeeeeeeeeeeeeeeeeee
     **/

    typedef KDTreeVectorOfVectorsAdaptor<cloud_t, float>  kd_tree_t;

    kd_tree_t mat_index(3 /*dim*/, (*cloud_y), 10 /* max leaf */ );
    mat_index.index->buildIndex();

    // loop through points
    tbb::parallel_for(int(0),num_fixed,[&](int i){
                                         // for(int i=0; i<num_fixed; ++i){

                                         const float search_radius = d2_thres;
                                         std::vector<std::pair<size_t,float>>  ret_matches;

                                         nanoflann::SearchParams params;
                                         //params.sorted = false;

                                         const size_t nMatches = mat_index.index->radiusSearch(&(*cloud_x)[i](0), search_radius, ret_matches, params);

                                         // Eigen::Vector3f cloud_xi = (*cloud_x)[i];
                                         Eigen::Matrix<float,3,1> RGB_x = ptr_fixed_pcd->RGB.row(i).transpose();
                                         Eigen::VectorXf labels_x;
                                         labels_x = ptr_fixed_pcd -> labels.row(i);
                                         //Eigen::Matrix<float,NUM_CLASS,1> labels_x = ptr_fixed_pcd->labels.row(i).transpose();
                                         // std::cout<<"nMatches: "<<nMatches<<std::endl;

                                         // for(int j=0; j<num_moving; j++){
                                         for(size_t j=0; j<nMatches; ++j){
                                           int idx = ret_matches[j].first;
                                           float d2 = ret_matches[j].second;
                                           // d2 = (x-y)^2
                                           float k = 0;
                                           float ck = 0;
                                           float sk = 0;
                                           float d2_color = 0;
                                           float d2_semantic = 0;
                                           float a = 0;
                                           // float d2 = 100;
                                           // d2 = (cloud_xi-(*cloud_y)[j]).squaredNorm();
                                           if(d2<d2_thres){
                                             // d2_color = color_kernel(i,j);
                                             Eigen::Matrix<float,3,1> RGB_y = ptr_moving_pcd->RGB.row(idx).transpose();
                                             Eigen::VectorXf labels_y = ptr_moving_pcd->labels.row(idx);
                                             //Eigen::Matrix<float,NUM_CLASS,1> labels_y = ptr_moving_pcd->labels.row(idx).transpose();
                                             d2_color = ((RGB_x-RGB_y).squaredNorm());
                                             d2_semantic = ((labels_x-labels_y).squaredNorm());
                                             if(d2_color<d2_c_thres){
                                               k = s2*exp(-d2/(2.0*l*l));
                                               ck = c_sigma*c_sigma*exp(-d2_color/(2.0*c_ell*c_ell));
                                               sk = s_sigma*s_sigma*exp(-d2_semantic/(2.0*s_ell*s_ell));
                                               a = ck*k*sk;
                                               if (a > sp_thres) A_trip_concur.push_back(Trip_t(i,idx,a));
                                             }
                                           }
                                         }
                                       });
    // }
    // form A
    A.setFromTriplets(A_trip_concur.begin(), A_trip_concur.end());
    A.makeCompressed();
    // std::cout<<"num of non-zeros in A: "<<A.nonZeros()<<std::endl;
  }


  void rkhs_se3::compute_flow(){
    // compute SE kernel
    se_kernel(ell, sigma*sigma);

    // some initialization of the variables
    omega = Eigen::Vector3f::Zero();
    v = Eigen::Vector3f::Zero();
    Eigen::Vector3d double_omega = Eigen::Vector3d::Zero(); // this is omega in double precision
    Eigen::Vector3d double_v = Eigen::Vector3d::Zero();

    tbb::spin_mutex omegav_lock;
    int sum = 0;
    // loop through points in cloud_x
    tbb::parallel_for(int(0),num_fixed,[&](int i){
                                         // initialize reused varaibles
                                         int num_non_zeros = A.innerVector(i).nonZeros();
                                         Eigen::MatrixXf Ai = Eigen::MatrixXf::Zero(1,num_non_zeros);
                                         Eigen::MatrixXf cross_xy = Eigen::MatrixXf::Zero(num_non_zeros,3);
                                         Eigen::MatrixXf diff_yx = Eigen::MatrixXf::Zero(num_non_zeros,3);
                                         Eigen::Matrix<double, 1, 3> partial_omega;
                                         Eigen::Matrix<double, 1, 3> partial_v;

                                         int j = 0;
                                         // loop through non-zero ids in ith row
                                         for(Eigen::SparseMatrix<float,Eigen::RowMajor>::InnerIterator it(A,i); it; ++it){
                                           int idx = it.col();
                                           Ai(0,j) = it.value();    // extract current value in A
                                           // std::cout<<"i: "<<i<<"A: "<<Ai(0,j)<<std::endl;
                                           cross_xy.row(j) = ((*cloud_x)[i].transpose().cross((*cloud_y)[idx].transpose()));
                                           diff_yx.row(j) = ((*cloud_y)[idx]-(*cloud_x)[i]).transpose();

                                           ++j;
                                         }

                                         partial_omega = (1/c*Ai*cross_xy).cast<double>();
                                         partial_v = (1/d*Ai*diff_yx).cast<double>();

                                         // sum them up
                                         omegav_lock.lock();
                                         double_omega += partial_omega.transpose();
                                         double_v += partial_v.transpose();
                                         sum += num_non_zeros;
                                         omegav_lock.unlock();
                                       });

    // std::cout<<"num of nonzeros in A: "<<sum<<std::endl;
    // update them to class-wide variables
    omega = double_omega.cast<float>();
    v = double_v.cast<float>();
  }


  void rkhs_se3::compute_step_size(){
    // compute skew matrix
    Eigen::Matrix3f omega_hat = skew(omega);
    
    // compute xi*z+v, xi^2*z+xi*v, xi^3*z+xi^2*v, xi^4*z+xi^3*v
    Eigen::MatrixXf xiz(num_moving,3);
    Eigen::MatrixXf xi2z(num_moving,3);
    Eigen::MatrixXf xi3z(num_moving,3);
    Eigen::MatrixXf xi4z(num_moving,3);
    Eigen::MatrixXf normxiz2(num_moving,1);
    Eigen::MatrixXf xiz_dot_xi2z(num_moving,1);
    Eigen::MatrixXf epsil_const(num_moving,1);

    tbb::parallel_for( int(0), num_moving, [&]( int j ){
                                             Eigen::Vector3f cloud_yi = (*cloud_y)[j];
                                             xiz.row(j) = omega.transpose().cross(cloud_yi.transpose())+v.transpose(); // (xi*z+v)
                                             xi2z.row(j) = (omega_hat*omega_hat*cloud_yi\
                                                            +(omega_hat*v)).transpose();    // (xi^2*z+xi*v)
                                             xi3z.row(j) = (omega_hat*omega_hat*omega_hat*cloud_yi\
                                                            +(omega_hat*omega_hat*v)).transpose();  // (xi^3*z+xi^2*v)
                                             xi4z.row(j) = (omega_hat*omega_hat*omega_hat*omega_hat*cloud_yi\
                                                            +(omega_hat*omega_hat*omega_hat*v)).transpose();    // (xi^4*z+xi^3*v)
                                             normxiz2(j,0) = xiz.row(j).squaredNorm();
                                             xiz_dot_xi2z(j,0) = (-xiz.row(j).dot(xi2z.row(j)));
                                             epsil_const(j,0) = xi2z.row(j).squaredNorm()+2*xiz.row(j).dot(xi3z.row(j));
                                           });

    // initialize coefficients
    float temp_coef = 1/(2.0*ell*ell);   // 1/(2*l^2)
    double B = 0;
    double C = 0;
    double D = 0;
    double E = 0;

    tbb::spin_mutex BCDE_lock;
    
    tbb::parallel_for(int(0),num_fixed,[&](int i){
                                         // loop through used index in ith row
                                         double Bi=0;
                                         double Ci=0;
                                         double Di=0;
                                         double Ei=0;

                                         for(Eigen::SparseMatrix<float,Eigen::RowMajor>::InnerIterator it(A,i); it; ++it){
                                           int idx = it.col();
            
                                           // diff_xy = x[i] - y[used_idx[j]]
                                           auto diff_xy = ((*cloud_x)[i] - (*cloud_y)[idx]);    
                                           // beta_i = -1/l^2 * dot(xiz,diff_xy)
                                           float beta_ij = (-2.0*temp_coef * xiz.row(idx)*diff_xy).value();
                                           // gamma_i = -1/(2*l^2) * (norm(xiz).^2 + 2*dot(xi2z,diff_xy))
                                           float gamma_ij = (-temp_coef * (normxiz2.row(idx)\
                                                                           + 2.0*xi2z.row(idx)*diff_xy)).value();
                                           // delta_i = 1/l^2 * (dot(-xiz,xi2z) + dot(-xi3z,diff_xy))
                                           float delta_ij = (2.0*temp_coef * (xiz_dot_xi2z.row(idx)\
                                                                              + (-xi3z.row(idx)*diff_xy))).value();
                                           // epsil_i = -1/(2*l^2) * (norm(xi2z).^2 + 2*dot(xiz,xi3z) + 2*dot(xi4z,diff_xy))
                                           float epsil_ij = (-temp_coef * (epsil_const.row(idx)\
                                                                           + 2.0*xi4z.row(idx)*diff_xy)).value();

                                           float A_ij = it.value();

                                           Bi += double(A_ij * beta_ij);
                                           Ci += double(A_ij * (gamma_ij+beta_ij*beta_ij/2.0));
                                           Di += double(A_ij * (delta_ij+beta_ij*gamma_ij + beta_ij*beta_ij*beta_ij/6.0));
                                           Ei += double(A_ij * (epsil_ij+beta_ij*delta_ij+1/2.0*beta_ij*beta_ij*gamma_ij\
                                                                + 1/2.0*gamma_ij*gamma_ij + 1/24.0*beta_ij*beta_ij*beta_ij*beta_ij));
                                         }
        
                                         // sum them up
                                         BCDE_lock.lock();
                                         B+=Bi;
                                         C+=Ci;
                                         D+=Di;
                                         E+=Ei;
                                         BCDE_lock.unlock();
                                       });

    Eigen::VectorXf p_coef(4);
    p_coef << 4.0*float(E),3.0*float(D),2.0*float(C),float(B);
    
    // solve polynomial roots
    Eigen::VectorXcf rc = poly_solver(p_coef);
    
    // find usable step size
    float temp_step = numeric_limits<float>::max();
    for(int i=0;i<rc.real().size();i++)
      if(rc(i,0).real()>0 && rc(i,0).real()<temp_step && rc(i,0).imag()==0)
        temp_step = rc(i,0).real();
    
    // if none of the roots are suitable, use min_step
    step = temp_step==numeric_limits<float>::max()? min_step:temp_step;

    // std::cout<<"step: "<<step<<std::endl;
    // if step>0.8, just use 0.8 as step
    step = step>0.8 ? 0.8:step;
  }

  void rkhs_se3::transform_pcd(){
    tbb::parallel_for(int(0), num_moving, [&]( int j ){
                                            (*cloud_y)[j] = transform.linear()*ptr_moving_pcd->positions[j]+transform.translation();
                                          });
    
  }


  /*
    void rkhs_se3::set_pcd(const int dataset_seq ,const cv::Mat& RGB_img,const cv::Mat& dep_img,MatrixXf_row semantic_labels){

    // create pcd_generator class
    pcd_generator pcd_gen(pcd_id);
    pcd_gen.dataset_seq = dataset_seq;
    pcd_id ++;
    // if it's the first image
    if(init == false){
    std::cout<<"initializing cvo..."<<std::endl;
    pcd_gen.load_image(RGB_img,dep_img,semantic_labels,ptr_fixed_fr.get());
    pcd_gen.create_pointcloud(ptr_fixed_fr.get(), ptr_fixed_pcd.get());
    std::cout<<"first pcd generated!"<<std::endl;
    init = true;
    return;
    }

    ptr_moving_fr.reset(new frame);
    ptr_moving_pcd.reset(new point_cloud);

    pcd_gen.load_image(RGB_img,dep_img,semantic_labels,ptr_moving_fr.get());
    pcd_gen.create_pointcloud(ptr_moving_fr.get(), ptr_moving_pcd.get());

    // get total number of points
    num_fixed = ptr_fixed_pcd->num_points;
    num_moving = ptr_moving_pcd->num_points;
    std::cout<<"num fixed: "<<num_fixed<<std::endl;
    std::cout<<"num moving: "<<num_moving<<std::endl;

    // extract cloud x and y
    cloud_x = &(ptr_fixed_pcd->positions);
    cloud_y = new std::vector<Eigen::Vector3f>(ptr_moving_pcd->positions);

    // transform = Eigen::Affine3f::Identity();
    // R = Eigen::Matrix3f::Identity(3,3); // initialize rotation matrix to I
    // T = Eigen::Vector3f::Zero();        // initialize translation matrix to zeros

    // initialization of parameters
    A_trip_concur.reserve(num_moving*20);
    A.resize(num_fixed,num_moving);
    A.setZero();
    }

  */

  void rkhs_se3::align(){
    int n = tbb::task_scheduler_init::default_num_threads();
    std::cout<<"num_thread: "<<n<<std::endl;

    // loop until MAX_ITER
    iter = MAX_ITER;
    for(int k=0; k<MAX_ITER; k++){
      // update transformation matrix
      update_tf();

      // apply transform to the point cloud
      transform_pcd();
        
      // compute omega and v
      compute_flow();

      // compute step size for integrating the flow
      compute_step_size();

      // stop if the step size is too small
      if(omega.cast<double>().norm()<eps && v.cast<double>().norm()<eps){
        iter = k;
        std::cout<<"norm, omega: "<<omega.norm()<<", v: "<<v.norm()<<std::endl;
        break;
      }

      // stacked omega and v for finding dtrans
      Eigen::VectorXf vec_joined(omega.size()+v.size());
      vec_joined << omega, v;

      // find the change of translation matrix dtrans
      Eigen::MatrixXf dtrans = Exp_SEK3(vec_joined, step);

      // extract dR and dT from dtrans
      Eigen::Matrix3f dR = dtrans.block<3,3>(0,0);
      Eigen::Vector3f dT = dtrans.block<3,1>(0,3);

      // calculate new R and T
      T = R * dT + T;
      R = R * dR;

      // if the se3 distance is smaller than eps2, break
      if(dist_se3(dR,dT)<eps_2){
        iter = k;
        std::cout<<"dist: "<<dist_se3(dR,dT)<<std::endl;
        break;
      }

      ell = (k>2)? 0.1*7:ell;
      ell = (k>9)? 0.06*7:ell;
      ell = (k>19)? 0.03*7:ell;
      // std::cout<<"omega: "<<omega<<std::endl;
      // std::cout<<"v: "<<v<<std::endl;
        
    }
    prev_transform = transform.matrix();
    // accum_tf.matrix() = transform.matrix().inverse() * accum_tf.matrix();
    accum_tf = accum_tf * transform.matrix();
    accum_tf_vis = accum_tf_vis * transform.matrix();   // accumilate tf for visualization
    update_tf();

    // visualize_pcd();
    ptr_fixed_pcd = std::move(ptr_moving_pcd);

    delete cloud_y;
  }

  template <class PointType>
  void rkhs_se3::set_pcd(int w, int h,
                         const dso::FrameHessian * img_source,
                         const vector<PointType> & source_points,
                         const dso::FrameHessian * img_target,
                         const vector<PointType> & target_points,
                         const Eigen::Affine3f & init_guess_transform) {

    if (source_points.size() == 0 || target_points.size() == 0) {
      return;
    }

    // function: fill in the features and pointcloud 
    auto loop_fill_pcd =
      [w, h] (const std::vector<PointType> & dso_pts,
              const dso::FrameHessian * frame,
              point_cloud & output_cvo_pcd ) {
        
        output_cvo_pcd.positions.clear();
        output_cvo_pcd.positions.resize(dso_pts.size());
        output_cvo_pcd.num_points = dso_pts.size();
        output_cvo_pcd.RGB = Eigen::MatrixXf::Zero(dso_pts.size(), 3);
        
        if (dso_pts.size() && dso_pts[0].num_semantic_classes ) {
          output_cvo_pcd.labels = Eigen::MatrixXf::Zero(dso_pts.size(), dso_pts[0].num_semantic_classes );
          output_cvo_pcd.num_classes = dso_pts[0].num_semantic_classes;
        } else
          output_cvo_pcd.num_classes = 0;
        
        for (int i = 0; i < dso_pts.size(); i++ ) {
          int semantic_class = -1;
          auto & p = dso_pts[i];
          if (dso_pts[0].num_semantic_classes) {
            p.semantics.maxCoeff(&semantic_class);
          }
          if (semantic_class && dso::classToIgnore.find(semantic_class) != dso::classToIgnore.end() ) {
            continue;
          } 

          // TODO: type of float * img???
          output_cvo_pcd.RGB(i, 2) = p.rgb(2);
          output_cvo_pcd.RGB(i, 1) = p.rgb(1);
          output_cvo_pcd.RGB(i, 0) = p.rgb(0);
          output_cvo_pcd.labels.row(i) = p.semantics; //output_cvo_pcd.semantic_labels.row(y*w+x);
          // gradient??
          //output_cvo_pcd.features(i,3) = frame->dI[(int)p.v * w + (int)p.u][1];
          //output_cvo_pcd.features(i,4) = frame->dI[(int)p.v * w + (int)p.u][2];

          // is dso::Pnt's 3d coordinates already generated??
          output_cvo_pcd.positions[i] = p.local_coarse_xyz;

        }
        
      };
    //ptr_moving_fr.reset(new frame);
    ptr_moving_pcd.reset(new point_cloud);

    loop_fill_pcd(source_points, img_source, *ptr_fixed_pcd);
    loop_fill_pcd(target_points, img_target, *ptr_moving_pcd);

    // get total number of points
    num_fixed = ptr_fixed_pcd->num_points;
    num_moving = ptr_moving_pcd->num_points;
    std::cout<<"num fixed: "<<num_fixed<<std::endl;
    std::cout<<"num moving: "<<num_moving<<std::endl;

    // extract cloud x and y
    cloud_x = &(ptr_fixed_pcd->positions);
    cloud_y = new cloud_t (ptr_moving_pcd->positions);

    // initialization of parameters
    A_trip_concur.reserve(num_moving*20);
    A.resize(num_fixed,num_moving);
    A.setZero();

    transform = init_guess_transform.inverse();
    R = transform.linear();
    T = transform.translation();
    std::cout<<"[Cvo ] the init guess for the transformation is \n"
             <<R<<std::endl<<T<<std::endl; 
  }


  template void rkhs_se3::set_pcd<dso::Pnt>(int w, int h,
                                   const dso::FrameHessian * img_source,
                                   const std::vector<dso::Pnt> & source_points,
                                   const dso::FrameHessian * img_target,
                                   const vector<dso::Pnt> & target_points,
                                   const Eigen::Affine3f & init_guess_transform);

  template void rkhs_se3::set_pcd<dso::CvoTrackingPoints>(int w, int h,
                                                          const dso::FrameHessian * img_source,
                                                 const std::vector<dso::CvoTrackingPoints> & source_points,
                                                 const dso::FrameHessian * img_target,
                                                 const vector<dso::CvoTrackingPoints> & target_points,
                                                 const Eigen::Affine3f & init_guess_transform);

  
  /*
    void rkhs_se3::visualize_pcd(){
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ptr_fixed_vis(new pcl::PointCloud<pcl::PointXYZRGBA>());
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ptr_moved_vis(new pcl::PointCloud<pcl::PointXYZRGBA>());
    pcl::PointCloud<pcl::PointXYZRGBA> fixed_vis; 
    pcl::PointCloud<pcl::PointXYZRGBA> moved_vis; 
    fixed_vis = ptr_fixed_pcd->pcl_cloud;
    moved_vis = ptr_moving_pcd->pcl_cloud;
    
    // pcl::transformPointCloud(fixed_vis, fixed_vis, prev_transform);
    pcl::transformPointCloud(moved_vis, moved_vis, accum_tf_vis);
    *ptr_fixed_vis = fixed_vis;
    *ptr_moved_vis = moved_vis;
    string cloud_name = "frame" + to_string(frame_id);
    // if it's the first pair, add target pcd into viewer
    
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> fixed_rgb(ptr_fixed_vis);
    viewer->addPointCloud<pcl::PointXYZRGBA> (ptr_fixed_vis,fixed_rgb, cloud_name);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloud_name);
    frame_id+=1;
    
    // transform moving_vis
    cloud_name = "frame" + to_string(frame_id);

    // transform moving pcd
    // pcl::transformPointCloud(ptr_moving_pcd->pcl_cloud, moved_vis, accum_tf_vis);
    // *ptr_moved_vis = moved_vis;
    
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> moved_rgb(ptr_moved_vis);
    viewer->addPointCloud<pcl::PointXYZRGBA> (ptr_moved_vis,moved_rgb, cloud_name);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloud_name);
    frame_id+=1;

    // if(frame_id == 6){
    while (!viewer->wasStopped()){
    viewer->spinOnce ();
    }
    // }
    }

    void rkhs_se3::run_cvo(const int dataset_seq,const cv::Mat& RGB_img,const cv::Mat& dep_img,MatrixXf_row semantic_labels){

    if(init == false){
    set_pcd(dataset_seq,RGB_img,dep_img,semantic_labels);
    }
    else{
    set_pcd(dataset_seq, RGB_img,dep_img,semantic_labels);
    align();
        
    std::cout<<"Total iterations: "<<iter<<std::endl;
    std::cout<<"RKHS-SE(3) Object Transformation Estimate: \n"<<transform.matrix()<<std::endl;
    }

    }
  */

}
