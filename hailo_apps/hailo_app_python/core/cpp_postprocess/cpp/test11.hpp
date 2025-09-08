#pragma once  
#include "hailo_objects.hpp"  
#include "hailo_common.hpp"  
#include "vms_croppers.hpp" 

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>  
#include <string>  
  
__BEGIN_DECLS
void filter_db(HailoROIPtr roi);  
void filter_ocr(HailoROIPtr roi);
void ocr_postprocess(HailoROIPtr roi, void *params);
void db_postprocess(HailoROIPtr roi, void *params);
std::vector<HailoROIPtr> crop_text_regions(std::shared_ptr<HailoMat> image, HailoROIPtr roi) ;  

  
// Character set for OCR recognition  
extern const std::vector<std::string> CHARACTERS;  
  
struct OCRResult {  
    std::string text;  
    float confidence;  
};  
  
OCRResult decode_ocr_output(const float* logits, int sequence_length, int num_classes);
  
struct DBParams {  
    float thresh = 0.3f;  
    float box_thresh = 0.7f;  
    int max_candidates = 1000;  
    float unclip_ratio = 2.0f;  
    int min_size = 3;  
    bool use_dilation = false;  
    std::string score_mode = "fast";  
    std::string box_type = "quad";  
};  
  
// Helper functions  
std::vector<cv::Point2f> get_mini_boxes(const std::vector<cv::Point>& contour, float& min_side);  
float box_score_fast(const cv::Mat& bitmap, const std::vector<cv::Point2f>& box);  
std::vector<cv::Point2f> unclip_polygon(const std::vector<cv::Point2f>& box, float unclip_ratio);
HailoBBox adjust_text_bbox(const HailoBBox &bbox, float padding_ratio = 0.1f);
HailoDetectionPtr clone_detection_for_ocr(HailoDetectionPtr detection);
__END_DECLS  
