#include "test.hpp"
#include <opencv2/opencv.hpp>
#include "clipper2/clipper.h"
#include <numeric>
#include <cmath>

using namespace Clipper2Lib;


namespace dbpost {

DBPostProcessor::DBPostProcessor(float thresh,
                                 float box_thresh,
                                 int max_candidates,
                                 float unclip_ratio,
                                 bool use_dilation,
                                 const std::string &score_mode,
                                 const std::string &box_type)
    : thresh_(thresh),
      box_thresh_(box_thresh),
      max_candidates_(max_candidates),
      unclip_ratio_(unclip_ratio),
      min_size_(3),
      score_mode_(score_mode),
      box_type_(box_type),
      use_dilation_(use_dilation) {
    if (use_dilation_) {
        dilation_kernel_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    }
}

float DBPostProcessor::box_score_fast(const cv::Mat &bitmap, const std::vector<cv::Point> &box) {
    cv::Rect rect = cv::boundingRect(box);
    std::vector<cv::Point> shifted_box;
    for (const auto &pt : box) {
        shifted_box.emplace_back(pt.x - rect.x, pt.y - rect.y);
    }

    cv::Mat mask = cv::Mat::zeros(rect.height, rect.width, CV_8UC1);
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{shifted_box}, cv::Scalar(1));
    cv::Mat roi = bitmap(rect);
    return static_cast<float>(cv::mean(roi, mask)[0]);
}

std::vector<cv::Point2f> DBPostProcessor::unclip(const std::vector<cv::Point2f> &box, float unclip_ratio) {
    // Convert input points to Clipper2 PathsD format
    PathD path;
    for (const auto &pt : box) {
        path.push_back(PointD(pt.x, pt.y));
    }

    // Compute offset (aka "unclip") distance
    double area = std::fabs(Area(path));
    double perimeter = Length(path);
    double offset = area * unclip_ratio / (perimeter + 1e-6); // Avoid division by 0

    PathsD expanded = InflatePaths({path}, offset, JoinType::Round, EndType::Polygon);

    std::vector<cv::Point2f> out;
    if (!expanded.empty()) {
        for (const auto &pt : expanded[0]) {
            out.emplace_back(cv::Point2f(pt.x, pt.y));
        }
    }
    return out;
}

std::vector<cv::Point> DBPostProcessor::get_mini_box(const std::vector<cv::Point2f> &box, float &side_len) {
    cv::RotatedRect rect = cv::minAreaRect(box);
    cv::Point2f points[4];
    rect.points(points);

    std::vector<cv::Point> box_int(4);
    for (int i = 0; i < 4; ++i) {
        box_int[i] = points[i];
    }
    side_len = std::min(rect.size.width, rect.size.height);
    return box_int;
}

std::vector<std::vector<cv::Point2f>> DBPostProcessor::boxes_from_bitmap(const cv::Mat &pred_map,
                                                                         const cv::Mat &bitmap,
                                                                         int dest_width,
                                                                         int dest_height) {
    std::vector<std::vector<cv::Point2f>> boxes;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    int num_contours = std::min(static_cast<int>(contours.size()), max_candidates_);

    for (int i = 0; i < num_contours; ++i) {
        auto &contour = contours[i];
        if (cv::contourArea(contour) < min_size_) continue;

        float score = box_score_fast(pred_map, contour);
        if (score < box_thresh_) continue;

        std::vector<cv::Point2f> contour_f;
        for (const auto &pt : contour) {
            contour_f.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
        }

        auto expanded = unclip(contour_f, unclip_ratio_);
        if (expanded.size() < 4) continue;

        float sside;
        auto mini_box = get_mini_box(expanded, sside);
        if (sside < min_size_ + 2) continue;

        std::vector<cv::Point2f> scaled_box;
        for (const auto &pt : mini_box) {
            float x = std::clamp(static_cast<float>(std::round(pt.x / bitmap.cols * dest_width)), 0.f, static_cast<float>(dest_width));
            float y = std::clamp(static_cast<float>(std::round(pt.y / bitmap.rows * dest_height)), 0.f, static_cast<float>(dest_height));
            scaled_box.emplace_back(x, y);
        }
        boxes.emplace_back(scaled_box);
    }
    return boxes;
}

std::vector<std::vector<cv::Point2f>> DBPostProcessor::run(const cv::Mat &pred_map,
                                                           int orig_height,
                                                           int orig_width) {
    cv::Mat bin_bitmap;
    cv::threshold(pred_map, bin_bitmap, thresh_, 1, cv::THRESH_BINARY);
    bin_bitmap.convertTo(bin_bitmap, CV_8UC1, 255);

    if (use_dilation_) {
        cv::dilate(bin_bitmap, bin_bitmap, dilation_kernel_);
    }

    return boxes_from_bitmap(pred_map, bin_bitmap, orig_width, orig_height);
}

extern "C" {

int db_postprocess(float* pred_data, int height, int width,
                   int orig_height, int orig_width,
                   float* results_buffer, int max_boxes)
{
    // Wrap raw data in cv::Mat
    cv::Mat pred_map(height, width, CV_32FC1, pred_data);

    dbpost::DBPostProcessor processor(
        0.3f,  // thresh
        0.5f,  // box_thresh
        max_boxes,
        2.0f,  // unclip_ratio
        false, // use_dilation
        "fast",
        "poly"
    );

    auto boxes = processor.run(pred_map, orig_height, orig_width);

    int count = std::min(static_cast<int>(boxes.size()), max_boxes);
    for (int i = 0; i < count; ++i) {
        const auto& box = boxes[i];
        for (int j = 0; j < 4; ++j) {
            results_buffer[i * 8 + j * 2 + 0] = box[j].x;
            results_buffer[i * 8 + j * 2 + 1] = box[j].y;
        }
    }

    return count;
}

}

} // namespace dbpost
