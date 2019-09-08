/* ----------------------------------------------------------------------------
 * Copyright 2019, Tzu-yuan Lin <tzuyuan@umich.edu>, Maani Ghaffari <maanigj@umich.edu>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   pcd_generator.cpp
 *  @author Tzu-yuan Lin, Maani Ghaffari 
 *  @brief  Source file for point cloud generator
 *  @date   August 15, 2019
 **/

#include "pcd_generator.hpp"


// using namespace cv;

namespace cvo{

    pcd_generator::pcd_generator(int img_idx):
    num_want(3000),
    dep_thres(30000),
    pcd_idx(img_idx)
    // ds_viewer(new pcl::visualization::PCLVisualizer ("CVO Pointcloud Visualization"))
    {
        
    }

    pcd_generator::~pcd_generator(){
        delete map;
    }

    void pcd_generator::make_pyramid(frame* ptr_fr){
        
        /** 
         * In this function we reference to Direct Sparse Odometry (DSO) by Engel et al.
         * https://github.com/JakobEngel/dso/blob/master/src/FullSystem/HessianBlocks.cpp#L128
         **/

        // initialize array for storing the pyramid
        int wl = ptr_fr->w;     // wl is the width of image at level l
        int hl = ptr_fr->h;     // hl is the height of image at level l
        for(int i=0; i<PYR_LEVELS; ++i){
            ptr_fr->dI_pyr[i] = new Eigen::Vector3f[wl*hl];
            ptr_fr->abs_squared_grad[i] = new float[wl*hl];
            wl /= 2;    // reduce the size of the image by 2 for each lower level
            hl /= 2;
        }

        ptr_fr->dI = ptr_fr->dI_pyr[0]; // dI = dI_pyr. Note: dI is a pointer so now dI and dI_ptr[0] point to the same place

        // extract intensity and flatten it into dI = dI_pyr[0]
        int h = ptr_fr->h;
        int w = ptr_fr->w;
        int _stride = ptr_fr->intensity.step;
        uint8_t *inten_val = ptr_fr->intensity.data;
        for(int i=0; i<h; ++i){
            for(int j=0; j<w; ++j){
                ptr_fr->dI[i*w+j][0] = inten_val[i*_stride+j];
            }
        }
        
        // initialize w and l to first level
        wl = w;
        hl = h;
        // start making pyramid, loop through different levels
        for(int lvl=0; lvl<PYR_LEVELS; ++lvl){
            
            // create a pointer point to dI at current level
            Eigen::Vector3f* dI_l = ptr_fr->dI_pyr[lvl];
        

            // create a pointer point to abs_squared_grad at current level
            float* abs_l = ptr_fr->abs_squared_grad[lvl];

            // if it's not the finest level, downsample 
            if(lvl>0){
                // create pointer to previous level
                int prev_lvl = lvl-1;
                int prev_wl = wl*2;
                Eigen::Vector3f* prev_dI_l = ptr_fr->dI_pyr[prev_lvl];

                // downsampling
                for(int y=0; y<hl; ++y){
                    for(int x=0; x<wl; ++x){
                        dI_l[x+y*wl][0] = 0.25f*(prev_dI_l[2*x   + 2*y*prev_wl][0] + \
												 prev_dI_l[2*x+1 + 2*y*prev_wl][0] + \
												 prev_dI_l[2*x   + 2*y*prev_wl+prev_wl][0] + \
												 prev_dI_l[2*x+1 + 2*y*prev_wl+prev_wl][0]);
                    }
                }
            }

            // calculate gradient
            // we skip the first row&col and the last row&col
            for(int idx=wl; idx<wl*(hl-1); ++idx){
                
                float dx = 0.5f*(dI_l[idx+1][0] - dI_l[idx-1][0]);
			    float dy = 0.5f*(dI_l[idx+wl][0] - dI_l[idx-wl][0]);

                // if it's not finite, set to 0
                if(!std::isfinite(dx)) dx=0;
			    if(!std::isfinite(dy)) dy=0;
                
                dI_l[idx][1] = dx;
			    dI_l[idx][2] = dy;

                // save current absolute gradient value (dx^2+dy^2) into ptr_fr->abs_squared_grad[lvl]
                abs_l[idx] = dx*dx+dy*dy;
                // abs_l[idx] = sqrt(dx*dx+dy*dy);    
            

            }

            // update level 
            wl/=2;
            hl/=2;
        }
    }

    void pcd_generator::select_point(frame* ptr_fr){
        
        int w = ptr_fr->w;
        int h = ptr_fr->h;
        
        make_pyramid(ptr_fr);   // create image pyramid

        map = new float[w*h];   // initialize the map for point selection

        PixelSelector pixel_selector(w, h);    // create point selection class
        num_selected = pixel_selector.makeMaps(ptr_fr, map, num_want);
        std::cout<<"num_selected from selector: "<<num_selected<<std::endl;

    }

    void pcd_generator::visualize_selected_pixels(frame* ptr_fr){
        
        int h = ptr_fr->h;
        int w = ptr_fr->w;

        // visualize the selected pixels in image
        cv::Mat img_selected;
        ptr_fr->image.copyTo(img_selected);
        for(int y=0; y<h; ++y){
            for(int x=0; x<w; ++x){
                uint16_t dep = ptr_fr->depth.at<uint16_t>(cv::Point(x, y));
                if(map[y*w+x]==0 || dep==0 ||  isnan(dep) || dep>dep_thres){
                    img_selected.at<cv::Vec3b>(cv::Point(x, y)).val[0] = 0;
                    img_selected.at<cv::Vec3b>(cv::Point(x, y)).val[1] = 0;
                    img_selected.at<cv::Vec3b>(cv::Point(x, y)).val[2] = 0;
                }
            }
        }
        cv::imshow("original image", ptr_fr->image);     // visualize original image
        cv::imshow("depth image", ptr_fr->depth);
        cv::imshow("selected image", img_selected);      // visualize selected pixels
        cv::imshow("semantic", ptr_fr->semantic_img);
        cv::waitKey(0);

    }

    void pcd_generator::get_points_from_pixels(frame* ptr_fr, point_cloud* ptr_pcd){
        
        float scaling_factor = 5000;    // scaling factor for depth data
        float fx;  // focal length x
        float fy;  // focal length y
        float cx;  // optical center x
        float cy;  // optic0.0484359

        // set camera parameters
        switch (dataset_seq){
        // 0: real sense camera
        case 0:
            scaling_factor = 1000.0;
            fx = 616.368;  // focal length x
            fy = 616.745;  // focal length y
            cx = 319.935;  // optical center x
            cy = 243.639;  // optical center y
            break;
        case 1: // tum fr1
            scaling_factor = 5000.0;    // scaling factor for depth data
            fx = 517.3;  // focal length x
            fy = 516.5;  // focal length y
            cx = 318.6;  // optical center x
            cy = 255.3;  // optical center y
            break;
        case 2: // tum fr2
            scaling_factor = 5000.0;
            fx = 520.9;  // focal length x
            fy = 521.0;  // focal length y
            cx = 325.1;  // optical center x
            cy = 249.7;  // optical center y
            break;
        case 3: // tum fr3
            scaling_factor = 5000.0;
            fx = 535.4;  // focal length x
            fy = 539.2;  // focal length y
            cx = 320.1;  // optical center x
            cy = 247.6;  // optical center y
            break;
        case 4: // kitti 15
            scaling_factor = 2000.0;
            fx = 718.856;
            fy = 718.856;
            cx = 607.1928;
            cy = 185.2157;
            break;

        case 5: // kitti 05
            scaling_factor = 2000.0;
            fx = 707.0912;
            fy = 707.0912;
            cx = 601.8873;
            cy = 183.1104;
            break;
        
        default:
            // default set to real sense
            scaling_factor = 1000.0;
            fx = 616.368;  // focal length x
            fy = 616.745;  // focal length y
            cx = 319.935;  // optical center x
            cy = 243.639;  // optical center y
            break;
        }

        int h = ptr_fr->h;
        int w = ptr_fr->w;
        
        int idx = 0;
        Eigen::Vector3f temp_position;
        cv::Mat temp_cv_position;
        for(int y=0; y<h; ++y){
            for(int x=0; x<w; ++x){
                uint16_t dep = ptr_fr->depth.at<uint16_t>(cv::Point(x, y));
                // if the point is selected
                int semantic_class;
                ptr_fr->semantic_labels.row(y*w+x).maxCoeff(&semantic_class);    // remove sky
                if(map[y*w+x]!=0 && dep!=0  && !isnan(dep) && dep<dep_thres && semantic_class!=1){
                    // construct depth
                    temp_position(2) = dep/scaling_factor;
                    // construct x and y
                    temp_position(0) = (x-cx) * temp_position(2) / fx;
                    temp_position(1) = (y-cy) * temp_position(2) / fy;
                    
                    // add point to pcd
                    //ptr_pcd->positions.emplace_back(temp_position);
                    ptr_pcd->positions.push_back(temp_position);
                    // std::cout<<"("<<temp_position(0)<<", "<<temp_position(1)<<", "<<temp_position(2)<<")"<<std::endl;

                    ++idx;
                }
            }
        }
        num_selected = idx;
        

        /**
         * below are just for visualization/ writing full pcd
        **/
        // ptr_pcd->pcl_cloud.width = h*w;
        // ptr_pcd->pcl_cloud.height = 1;
        // ptr_pcd->pcl_cloud.is_dense = false;
        // ptr_pcd->pcl_cloud.points.resize (ptr_pcd->pcl_cloud.width * ptr_pcd->pcl_cloud.height);
        // ptr_fr->semantic_img = cv::Mat(h,w,CV_8UC3);
        // int i = 0;
        // for(int y=0; y<h; ++y){
        //     for(int x=0; x<w; ++x){
                
        //         cv::Vec3b color;
        //         uint16_t dep = ptr_fr->depth.at<uint16_t>(cv::Point(x, y));
        //         if(true){
        //             ptr_pcd->pcl_cloud.points[i].x = (x-cx) * dep/scaling_factor / fx;
        //             ptr_pcd->pcl_cloud.points[i].y = (y-cy) * dep/scaling_factor / fy;
        //             ptr_pcd->pcl_cloud.points[i].z = dep/scaling_factor;

        //             ptr_pcd->pcl_cloud.points[i].r = ptr_fr->image.at<cv::Vec3b>(cv::Point(x, y)).val[2]; // r
        //             ptr_pcd->pcl_cloud.points[i].g = ptr_fr->image.at<cv::Vec3b>(cv::Point(x, y)).val[1]; // g  
        //             ptr_pcd->pcl_cloud.points[i].b = ptr_fr->image.at<cv::Vec3b>(cv::Point(x, y)).val[0]; // b 

        //             // create semantic image visualization
        //             float max_prob = 0;
        //             int max_idx = 0;
        //             for(int l=0; l<NUM_CLASS; ++l){
        //                 // std::cout<<"prob: "<<ptr_fr->semantic_labels(y*w+x,l)<<std::endl;
        //                 if(ptr_fr->semantic_labels(y*w+x,l)>max_prob){
        //                     max_prob = ptr_fr->semantic_labels(y*w+x,l);
        //                     max_idx = l;
        //                 }
        //             }
        //             // 0. building  // our 11 class
        //             // 1. sky
        //             // 2. road
        //             // 3. vegetation
        //             // 4. sidewalk
        //             // 5. car
        //             // 6. pedestrian
        //             // 7. cyclist
        //             // 8. signate
        //             // 9. fence
        //             // 10. pole
        //             switch(max_idx){
        //                 case 0:
        //                     color(0) = 128;
        //                     color(1) = 0;
        //                     color(2) = 0;
        //                     break;
        //                 case 1:
        //                     color(0) = 128;
        //                     color(1) = 128;
        //                     color(2) = 128;
        //                     break;
        //                 case 2:
        //                     color(0) = 128;
        //                     color(1) = 64;
        //                     color(2) = 128;
        //                     break;
        //                 case 3:
        //                     color(0) = 128;
        //                     color(1) = 128;
        //                     color(2) = 0;
        //                     break;
        //                 case 4:
        //                     color(0) = 0;
        //                     color(1) = 0;
        //                     color(2) = 192;
        //                     break;
        //                 case 5:
        //                     color(0) = 64;
        //                     color(1) = 0;
        //                     color(2) = 128;
        //                     break;
        //                 case 6:
        //                     color(0) = 64;
        //                     color(1) = 64;
        //                     color(2) = 0;
        //                     break;
        //                 case 7:
        //                     color(0) = 0;
        //                     color(1) = 128;
        //                     color(2) = 192;
        //                     break;
        //                 case 8:
        //                     color(0) = 192;
        //                     color(1) = 128;
        //                     color(2) = 128;
        //                     break;
        //                 case 9:
        //                     color(0) = 64;
        //                     color(1) = 64;
        //                     color(2) = 128;
        //                     break;
        //                 case 10:
        //                     color(0) = 192;
        //                     color(1) = 192;
        //                     color(2) = 128;
        //                     break;
        //             }
        //             // std::cout<<"color: "<<color<<std::endl;
        //             // ptr_fr->semantic_img.at<cv::Vec3b>(cv::Point(x,y)).val[0] = color[0];
        //             // ptr_fr->semantic_img.at<cv::Vec3b>(cv::Point(x,y)).val[1] = color[1];
        //             // ptr_fr->semantic_img.at<cv::Vec3b>(cv::Point(x,y)).val[2] = color[2];
        //             ptr_fr->semantic_img.at<cv::Vec3b>(y,x) = color;

        //             ++i;
        //         }
        //     // std::cout<<"("<<pcl_cloud.points[i].x<<", "<<pcl_cloud.points[i].y<<", "<<pcl_cloud.points[i].z<<")"<<std::endl;
        //     }
        // }
        // ptr_pcd->pcl_cloud.width = i;
        // ptr_pcd->pcl_cloud.points.resize (ptr_pcd->pcl_cloud.width * ptr_pcd->pcl_cloud.height);
    
        // string img_idx_str = std::to_string(pcd_idx);
        // string path = "/media/justin/LaCie/data/data_kitti_15/"+img_idx_str+"_full.pcd";
        
        // pcl::io::savePCDFileASCII(path, ptr_pcd->pcl_cloud);
    }

    void pcd_generator::get_features(frame* ptr_fr, point_cloud* ptr_pcd){

        int h = ptr_fr->h;
        int w = ptr_fr->w;

        ptr_pcd->RGB = Eigen::MatrixXf::Zero(num_selected,3);
        ptr_pcd->labels = Eigen::MatrixXf::Zero(num_selected,ptr_pcd->num_classes);
        int idx = 0;

        for(int y=0; y<h; ++y){
            for(int x=0; x<w; ++x){
                // if the point is selected
                uint16_t dep = ptr_fr->depth.at<uint16_t>(cv::Point(x, y));
                int semantic_class;
                ptr_fr->semantic_labels.row(y*w+x).maxCoeff(&semantic_class);    // remove sky
                if(map[y*w+x]!=0 && dep!=0 && dep<dep_thres && !isnan(dep) && semantic_class!=1){
                    
                    // extract bgr value
                    ptr_pcd->RGB(idx,2) = ptr_fr->image.at<cv::Vec3b>(cv::Point(x, y)).val[0]; // b 
                    ptr_pcd->RGB(idx,1) = ptr_fr->image.at<cv::Vec3b>(cv::Point(x, y)).val[1]; // g   
                    ptr_pcd->RGB(idx,0) = ptr_fr->image.at<cv::Vec3b>(cv::Point(x, y)).val[2]; // r
                    
                    // extract gradient
                    // ptr_pcd->features(idx,3) = ptr_fr->dI[y*w+x][1];
                    // ptr_pcd->features(idx,4) = ptr_fr->dI[y*w+x][2];
                    
                    // extract semantic labels
                    ptr_pcd->labels.row(idx) = ptr_fr->semantic_labels.row(y*w+x);


                    ++idx;
                }
            }
        }
    }

    void pcd_generator::load_image(const cv::Mat& RGB_img,const cv::Mat& dep_img,MatrixXf_row semantic_labels,frame* ptr_fr){

        // load images                            
        ptr_fr->image = RGB_img;
        ptr_fr->depth = dep_img;

        cv::cvtColor(ptr_fr->image, ptr_fr->intensity, cv::COLOR_RGB2GRAY);
    
        ptr_fr->h = ptr_fr->image.rows;
        ptr_fr->w = ptr_fr->image.cols;

        ptr_fr->semantic_labels = semantic_labels;
    }

    void pcd_generator::create_pointcloud(frame* ptr_fr, point_cloud* ptr_pcd){
        
        int h = ptr_fr->h;
        int w = ptr_fr->w;

        select_point(ptr_fr);

        get_points_from_pixels(ptr_fr, ptr_pcd);

        get_features(ptr_fr, ptr_pcd);

        // visualize_selected_pixels(ptr_fr);

        // string save_path = "/media/justin/LaCie/data/data_kitti_15/";

        // create_pcl_pointcloud(ptr_pcd, save_path);

        ptr_pcd->num_points = num_selected;

        for(int i=0;i<PYR_LEVELS;i++)
		{
			delete[] ptr_fr->dI_pyr[i];
			delete[] ptr_fr->abs_squared_grad[i];
		}
    }
  /*
    void pcd_generator::create_pcl_pointcloud(point_cloud* ptr_pcd, const string& folder){

        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        
        cloud.width = num_selected;
        cloud.height = 1;
        cloud.is_dense = false;
        cloud.points.resize (cloud.width * cloud.height);


        for(int i=0; i<num_selected; ++i){

            cloud.points[i].x = ptr_pcd->positions[i](0);
            cloud.points[i].y = ptr_pcd->positions[i](1);
            cloud.points[i].z = ptr_pcd->positions[i](2);

            cloud.points[i].r = ptr_pcd->RGB(i,0);
            cloud.points[i].g = ptr_pcd->RGB(i,1);
            cloud.points[i].b = ptr_pcd->RGB(i,2);

            // std::cout<<"("<<pcl_cloud.points[i].x<<", "<<pcl_cloud.points[i].y<<", "<<pcl_cloud.points[i].z<<")"<<std::endl;
        }

        // pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptr_cloud_vis(new pcl::PointCloud<pcl::PointXYZRGB>());
        // *ptr_cloud_vis = cloud;
        
        // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> fixed_rgb(ptr_cloud_vis);
        
        // ds_viewer->addPointCloud<pcl::PointXYZRGB> (ptr_cloud_vis,fixed_rgb, "downsample");
        // ds_viewer->addCoordinateSystem(1.0,"downsample");
        // ds_viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "downsample");

        // // while (!viewer->wasStopped()){
        //     ds_viewer->spinOnce ();
        // // }
        string img_idx_str = std::to_string(pcd_idx);
        string path = folder+img_idx_str+".pcd";
        // delete ptr_cloud_vis;
        pcl::io::savePCDFileASCII (path, cloud);

    }
  */
}
