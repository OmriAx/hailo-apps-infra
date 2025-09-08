#pragma once

// ---- Portable include for Clipper2, with fallback to legacy clipper.hpp ----
#if __has_include(<clipper2/clipper.h>)
  #include <clipper2/clipper.h>
  #define OCRPP_USE_CLIPPER2 1
#elif __has_include("clipper2/clipper.h")
  #include "clipper2/clipper.h"
  #define OCRPP_USE_CLIPPER2 1
#elif __has_include(<clipper.hpp>)
  #include <clipper.hpp>
  #define OCRPP_USE_CLIPPER_LEGACY 1
#else
  #error "Clipper2 not found. Ensure include path: ../third_party/clipper2/include or system-wide install."
#endif

#include <vector>
#include <cmath>
#include <opencv2/core.hpp>

// ===================== Type + API Normalization =====================
#if defined(OCRPP_USE_CLIPPER2)
  namespace ocrpp {
    // Bring Clipper2 types in a stable alias namespace
    using namespace Clipper2Lib;
    using PointT = Point64;
    using PathT  = Path64;
    using PathsT = Paths64;

    inline double area(const PathT &p) { return Clipper2Lib::Area(p); }
    
    // Fix: Clipper2 doesn't have a built-in Perimeter function, so implement it
    inline double perimeter(const PathT &p) {
      if (p.size() < 2) return 0.0;
      double sum = 0.0;
      for (size_t i = 0; i < p.size(); ++i) {
        const auto &a = p[i];
        const auto &b = p[(i + 1) % p.size()];
        double dx = static_cast<double>(b.x) - static_cast<double>(a.x);
        double dy = static_cast<double>(b.y) - static_cast<double>(a.y);
        sum += std::sqrt(dx*dx + dy*dy);
      }
      return sum;
    }

    // Offset wrapper (delta first in Clipper2)
    inline void offset_execute(ClipperOffset &co, double delta, PathsT &out) {
      co.Execute(delta, out);
    }

    // AddPath wrapper (enum names differ)
    inline void offset_add_path(ClipperOffset &co, const PathT &p) {
      co.AddPath(p, JoinType::Round, EndType::Polygon);
    }
  } // namespace ocrpp

#elif defined(OCRPP_USE_CLIPPER_LEGACY)
  namespace ocrpp {
    using namespace ClipperLib;
    using PointT = IntPoint;     // legacy is 64-bit ints too
    using PathT  = Path;
    using PathsT = Paths;

    inline double area(const PathT &p) { return ClipperLib::Area(p); }

    // Legacy Clipper v6 doesn't expose Perimeter; implement it
    inline double perimeter(const PathT &p) {
      if (p.size() < 2) return 0.0;
      double sum = 0.0;
      for (size_t i = 0; i < p.size(); ++i) {
        const auto &a = p[i];
        const auto &b = p[(i + 1) % p.size()];
        double dx = static_cast<double>(b.X) - static_cast<double>(a.X);
        double dy = static_cast<double>(b.Y) - static_cast<double>(a.Y);
        sum += std::sqrt(dx*dx + dy*dy);
      }
      return sum;
    }

    // Offset wrapper (solution first in legacy)
    inline void offset_execute(ClipperOffset &co, double delta, PathsT &out) {
      co.Execute(out, delta);
    }

    // AddPath wrapper (legacy enum names)
    inline void offset_add_path(ClipperOffset &co, const PathT &p) {
      co.AddPath(p, jtRound, etClosedPolygon);
    }
  } // namespace ocrpp
#endif

// ===================== Public Helper: polygon unclip =====================
// Scale by 10 for numeric stability (matches pyclipper approach).
inline std::vector<cv::Point> ocrpp_unclip_polygon(const std::vector<cv::Point> &poly,
                                                    float unclip_ratio,
                                                    int scale = 10)
{
  using namespace ocrpp;

  if (poly.size() < 3) return poly;

  // Build integer path with scaling
  PathT subj;
  subj.reserve(poly.size());
#if defined(OCRPP_USE_CLIPPER2)
  for (const auto &pt : poly) subj.emplace_back(static_cast<long long>(pt.x) * scale,
                                                static_cast<long long>(pt.y) * scale);
#else
  for (const auto &pt : poly) subj << PointT(static_cast<cInt>(pt.x) * scale,
                                             static_cast<cInt>(pt.y) * scale);
#endif

  const double A = std::fabs(area(subj));
  const double P = perimeter(subj);
  if (P <= 0.0) return poly;

  const double delta = (A * static_cast<double>(unclip_ratio)) / P;

  ClipperOffset co;
  offset_add_path(co, subj);
  PathsT solution;
  offset_execute(co, delta, solution);

  std::vector<cv::Point> out;
  if (!solution.empty()) {
    const auto &first = solution.front();
    out.reserve(first.size());
#if defined(OCRPP_USE_CLIPPER2)
    for (const auto &pt : first)
      out.emplace_back(static_cast<int>(pt.x / scale), static_cast<int>(pt.y / scale));
#else
    for (const auto &pt : first)
      out.emplace_back(static_cast<int>(pt.X / scale), static_cast<int>(pt.Y / scale));
#endif
  } else {
    // If offset failed, fall back to original polygon
    out = poly;
  }

  return out;
}