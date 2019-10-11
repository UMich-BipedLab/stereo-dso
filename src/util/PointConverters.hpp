#pragma once
#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/CvoTrackingPoints.h"
#include <cmath>
#include <cstdlib>

namespace dso {
  
  template <typename Pnt>
  void Pnt_to_CvoTrackingPoints(const Pnt & in, CvoTrackingPoints & out) {
    out.rgb = in.rgb;
    out.local_coarse_xyz = in.local_coarse_xyz;
    out.num_semantic_classes = in.num_semantic_classes;
    out.u = in.u;
    out.v = in.v;
    out.dI_xy[0] = in.dI_xy[0];
    out.dI_xy[1] = in.dI_xy[1];
    out.idepth = in.idepth;
    if (in.num_semantic_classes) 
      out.semantics = in.semantics;
  }

  inline void HSVtoRGB(float H, float S, float V, float output[3]) {
    float C = S * V;
    float X = C * (1 - std::abs(std::fmod(H / 60.0, 2) - 1));
    float m = V - C;
    float Rs, Gs, Bs;

    if(H >= 0 && H < 60) {
      Rs = C;
      Gs = X;
      Bs = 0;	
    }
    else if(H >= 60 && H < 120) {	
      Rs = X;
      Gs = C;
      Bs = 0;	
    }
    else if(H >= 120 && H < 180) {
      Rs = 0;
      Gs = C;
      Bs = X;	
    }
    else if(H >= 180 && H < 240) {
      Rs = 0;
      Gs = X;
      Bs = C;	
    }
    else if(H >= 240 && H < 300) {
      Rs = X;
      Gs = 0;
      Bs = C;	
    }
    else {
      Rs = C;
      Gs = 0;
      Bs = X;	
    }
	
    output[0] = (Rs + m) * 255;
    output[1] = (Gs + m) * 255;
    output[2] = (Bs + m) * 255;
  }
  
  inline void RGBtoHSV(float bgr[3], // not bgr! 
                float hsv[3] ) {
    float delta, min_;
    float h = 0, s, v;

    float b = bgr[0], g = bgr[1], r = bgr[2]; 

    min_ = std::min(std::min(bgr[0], bgr[1]), bgr[2]);
    v = std::max(std::max(r, g), b);
    delta = v - min_;

    if (v == 0.0)
      s = 0;
    else
      s = delta / v;

    if (s == 0)
      h = 0.0;

    else
    {
      if (r == v)
        h = (g - b) / delta;
      else if (g == v)
        h = 2 + (b - r) / delta;
      else if (b == v)
        h = 4 + (r - g) / delta;

      h *= 60;

      if (h < 0.0)
        h = h + 360;
    }

    hsv[0] = h;
    hsv[1] = s;
    hsv[2] = v / 255;

  }

  

  

}


