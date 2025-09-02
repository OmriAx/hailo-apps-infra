// db_postprocess.hpp
#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace dbpost {

class DBPostProcessor {
public:
    DBPostProcessor(float thresh = 0.3,
                    float box_thresh = 0.6,
                    int max_candidates = 1000,
                    float unclip_ratio = 2.0,
                    bool use_dilation = false,
                    const std::string &score_mode = "fast",
                    const std::string &box_type = "quad");

    std::vector<std::vector<cv::Point2f>> run(const cv::Mat &pred_map,
                                              int orig_height,
                                              int orig_width);

private:
    float thresh_;
    float box_thresh_;
    int max_candidates_;
    float unclip_ratio_;
    int min_size_;
    std::string score_mode_;
    std::string box_type_;
    bool use_dilation_;
    cv::Mat dilation_kernel_;

    float box_score_fast(const cv::Mat &bitmap, const std::vector<cv::Point> &box);
    std::vector<cv::Point2f> unclip(const std::vector<cv::Point2f> &box, float unclip_ratio);
    std::vector<cv::Point> get_mini_box(const std::vector<cv::Point2f> &box, float &side_len);
    std::vector<std::vector<cv::Point2f>> boxes_from_bitmap(const cv::Mat &pred_map,
                                                             const cv::Mat &bitmap,
                                                             int dest_width,
                                                             int dest_height);
};

extern "C" {

/**
 * Perform full DB postprocessing on a prediction map.
 *
 * @param pred_data Raw float* prediction map (1 channel, HxW).
 * @param height Height of the prediction map.
 * @param width Width of the prediction map.
 * @param orig_height Original image height.
 * @param orig_width Original image width.
 * @param results_buffer Preallocated output buffer for [num_boxes][4][2] float coordinates (x, y).
 * @param max_boxes Max number of boxes the results_buffer can hold.
 * @return Number of boxes returned.
 */
__attribute__((visibility("default")))
int db_postprocess(float* pred_data, int height, int width,
                   int orig_height, int orig_width,
                   float* results_buffer, int max_boxes);

}

} // namespace dbpost
