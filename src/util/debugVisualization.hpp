#include "Pnt.h"
#include "NumType.h"

#include <vector>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace dso {

  inline void save_points_as_color_pcd(std::string  filename, const std::vector<Pnt> & pts) {
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    //pcl::PointCloud<pcl::PointXYZ> cloud;

    cloud.width = pts.size();
    cloud.height = 1;
    cloud.is_dense = false;
    cloud.points.resize(pts.size());

    for (size_t i = 0; i < cloud.points.size(); i++) {
      cloud.points[i].x = pts[i].local_coarse_xyz(0);
      cloud.points[i].y = pts[i].local_coarse_xyz(1);
      cloud.points[i].z = pts[i].local_coarse_xyz(2);

      uint8_t r = pts[i].rgb(2);
      uint8_t g = pts[i].rgb(1);
      uint8_t b = pts[i].rgb(0);
      uint32_t rgb = ((uint32_t) r << 16 |(uint32_t) g << 16  | (uint32_t) b ) ;
      cloud.points[i].rgb = *reinterpret_cast<float*>(&rgb);
    }
    pcl::io::savePCDFileASCII(filename.c_str(), cloud );
  }


  inline void save_points_as_gray_pcd(std::string  filename, const std::vector<Pnt> & pts) {
    pcl::PointCloud<pcl::PointXYZ> cloud;

    cloud.width = pts.size();
    cloud.height = 1;
    cloud.is_dense = false;
    cloud.points.resize(pts.size());

    for (size_t i = 0; i < cloud.points.size(); i++) {
      cloud.points[i].x = pts[i].local_coarse_xyz(0);
      cloud.points[i].y = pts[i].local_coarse_xyz(1);
      cloud.points[i].z = pts[i].local_coarse_xyz(2);
    }
    pcl::io::savePCDFileASCII(filename.c_str(), cloud );
  }

  inline void save_img_with_projected_points(std::string filename,
                                             float * img_gray,
                                             int w, int h,
                                             const Mat33f & intrinsic, 
                                             const std::vector<Pnt> & pts,
                                             // true: write.
                                             // false: imshow
                                             bool write_or_imshow) {
    cv::Mat paint (h, w, CV_32F, img_gray);
    paint.convertTo(paint, CV_8U);
    cv::cvtColor(paint, paint, cv::COLOR_GRAY2BGR);
    std::cout<<intrinsic<<std::endl;
    for (auto && p: pts) {
      auto xyz = p.local_coarse_xyz;
      Vec3f uv = intrinsic * xyz;
      uv(0) = uv(0) / uv(2);
      uv(1) = uv(1) / uv(2);
      //std::cout<<uv(0)<<","<<uv(1)<<","<<p.local_coarse_xyz<<"\n";
      cv::circle(paint,cv::Point2f(uv(0), uv(1)), 5.0,
                 cv::Scalar(0, 255, 0));
      
    }
    if (write_or_imshow) {
      cv::imwrite(filename, paint);      
    } else {
      cv::imshow("projected", paint);
      cv::waitKey(300);
    }
      
  }

}
