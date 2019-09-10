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


#pragma once

#include "util/NumType.h"
#include "algorithm"
#include "util/EigenAlignment.h"
namespace dso
{

  template<typename T>
  class MinimalImage
  {
  public:
    int w;
    int h;
    T* data;
    int numChannels;
    /*
     * creates minimal image with own memory, one dimension
     */
    inline MinimalImage(int w_, int h_) : w(w_), h(h_)
    {
      data = new T[w*h];
      numChannels = 1;
      ownData=true;
    }

    /*
     * creates minimal image with own memory, multi-dimensions
     */
    inline MinimalImage(int w_, int h_, int numChannels_) : w(w_), h(h_), numChannels (numChannels_)
    {
      data = new T[w*h*numChannels];
      ownData=true;
    }

    /*
     * creates minimal image wrapping around existing memory
     */
    inline MinimalImage(int w_, int h_, T* data_, int numChannels_) : w(w_), h(h_), numChannels(numChannels_)
    {
      data = data_;
      ownData=false;
    }

    inline ~MinimalImage()
    {
      if(ownData) delete [] data;
    }

    inline MinimalImage* getClone()
    {
      MinimalImage* clone = new MinimalImage(w,h, numChannels);
      memcpy(clone->data, data, sizeof(T)*w*h * numChannels);
      
      return clone;
    }


    inline T & at(int x, int y) {return data[((int)x+((int)y)*w) * numChannels];}
    inline T & at(int i) {return data[i * numChannels];}
    inline T * at_ptr(int x, int y) {return &data[((int)x+((int)y)*w) * numChannels];}
    inline T * at_ptr(int i) {return &data[i * numChannels];}


    inline void setBlack()
    {
      memset(data, 0, sizeof(T)*w*h*numChannels);
    }

    inline void setConst(T val)
    {
      //for(int i=0;i<w*h*numChannels;i++) {
      //  data[i] = val;
      memset(data, val, sizeof(T) * w * h * numChannels);
      //}
    }

    inline void setPixel(const float &u, const float &v, T  val)
    {
      T * pos = at_ptr(u, v);
      memset(pos, val, sizeof(T) * numChannels);
      //for (int i = 0; i < numChannels; i++)
      //  pos[i] = val;
    }

    inline void setPixel(const float &u, const float &v, T * val)
    {
      T * pos = at_ptr(u, v);
      
      memcpy(pos, val, sizeof(T) * numChannels);
      //for (int i = 0; i < numChannels; i++)
      //  pos[i] = val;
    }

    inline void setPixel(int i, T * val)
    {
      T * pos = at_ptr(i);
      memcpy(pos, val, sizeof(T) * numChannels);
      //for (int i = 0; i < numChannels; i++)
      //  pos[i] = val;
    }
    inline void setPixel(int i, T  val)
    {
      T * pos = at_ptr(i);
      memset(pos, val, sizeof(T) * numChannels);
      //for (int i = 0; i < numChannels; i++)
      //  pos[i] = val;
    }


    inline void setPixel1(const float &u, const float &v, T val)
    {
      T * pos = at_ptr(u+0.5f, v+0.5f);
      for (int i = 0; i < numChannels; i++)
        pos[i] = val;

    }

    inline void setPixel4(const float &u, const float &v, T val)
    {
      setPixel(u+1.0f,v+1.0f, val);
      setPixel(u+1.0f,v, val);
      setPixel(u,v+1.0f, val);
      setPixel(u,v , val);

    }

    inline void setPixel9(const int &u, const int &v, T val)
    {
 
      setPixel(u+1,v-1, val);
      setPixel(u+1,v, val);
      setPixel(u+1,v+1, val);
      setPixel(u,v-1, val);
      setPixel(u,v, val);
      setPixel(u,v+1, val);
      setPixel(u-1,v-1, val);
      setPixel(u-1,v, val);
      setPixel(u-1,v+1, val);
    }

    inline void setPixelCirc(const int &u, const int &v, T val)
    {
      for(int i=-3;i<=3;i++)
      {
        setPixel(u+3,v+i, val);
        setPixel(u-3,v+i, val);
        setPixel(u+2,v+i, val);
        setPixel(u-2,v+i, val);

        setPixel(u+i,v-3, val);
        setPixel(u+i,v+3, val);
        setPixel(u+i,v-2, val);
        setPixel(u+i,v+2, val);
      }
    }

    inline void setPixel1(const float &u, const float &v, T *val)
    {
      setPixel(u+0.5, v+0.5, val);
    }

    inline void setPixel4(const float &u, const float &v, T *val)
    {
      setPixel(u+1.0f,v+1.0f, val);
      setPixel(u+1.0f,v, val);
      setPixel(u,v+1.0f, val);
      setPixel(u,v , val);

    }

    inline void setPixel9(const int &u, const int &v, T *val)
    {

      setPixel(u+1,v-1, val);
      setPixel(u+1,v, val);
      setPixel(u+1,v+1, val);
      setPixel(u,v-1, val);
      setPixel(u,v, val);
      setPixel(u,v+1, val);
      setPixel(u-1,v-1, val);
      setPixel(u-1,v, val);
      setPixel(u-1,v+1, val);
    }

    inline void setPixelCirc(const int &u, const int &v, T* val)
    {
      for(int i=-3;i<=3;i++)
      {
        setPixel(u+3,v+i, val);
        setPixel(u-3,v+i, val);
        setPixel(u+2,v+i, val);
        setPixel(u-2,v+i, val);

        setPixel(u+i,v-3, val);
        setPixel(u+i,v+3, val);
        setPixel(u+i,v-2, val);
        setPixel(u+i,v+2, val);
      }
    }
















  private:
    bool ownData;
  };

  typedef Eigen::Matrix<unsigned char,3,1> Vec3b;
  typedef Eigen::VectorXf VecXf;
  typedef MinimalImage<float> MinimalImageF;
  typedef MinimalImage<float> MinimalImageF3;
  typedef MinimalImage<float> MinimalImageFX;
  typedef MinimalImage<unsigned char> MinimalImageB;
  typedef MinimalImage<unsigned char> MinimalImageB3;
  typedef MinimalImage<unsigned short> MinimalImageB16;

}

