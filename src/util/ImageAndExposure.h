/*
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


#pragma once
#include <cstring>
#include <iostream>
#include <opencv2/core.hpp>
#include "NumType.h"


namespace dso
{


  class ImageAndExposure
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int w,h;				// width and height;
    float* image;			// irradiance. between 0 and 256
    float* image_rgb;
    float * image_semantics;
    int num_classes;

    double timestamp;
    float exposure_time;	// eposure time in ms.
    
    inline ImageAndExposure(int w_, int h_, int num_semantic_classes=0, double timestamp_=0) : w(w_), h(h_), num_classes(num_semantic_classes), timestamp(timestamp_)
    {
      image = new float[w*h];
      image_rgb = new float [ 3 * w * h];

      if (num_classes > 0)
        image_semantics = new float [w * h * num_classes];
      else
        image_semantics = NULL;
      exposure_time=1;
    }
    inline ~ImageAndExposure()
    {
      delete[] image;
      delete [] image_rgb;
      if (num_classes > 0)
        delete [] image_semantics;
    }

    inline void copyMetaTo(ImageAndExposure &other)
    {
      other.exposure_time = exposure_time;
    }
  
    inline ImageAndExposure* getDeepCopy()
    {
      ImageAndExposure* img = new ImageAndExposure(w,h,timestamp);
      img->exposure_time = exposure_time;
      memcpy(img->image, image, w*h*sizeof(float));
      return img;
    }
  };


}
