#pragma once  
#include "hailo_objects.hpp"  
#include "hailo_common.hpp"  
#include <opencv2/opencv.hpp>  
#include <vector>  
#include <string>  
  
// Character vocabulary for CTC decoding  
extern const std::vector<std::string> CHARACTERS;  
  
struct OcrParams {  
    float det_bin_thresh = 0.3;  
    float det_box_thresh = 0.6;  
    float det_unclip_ratio = 1.5;  
    int det_max_candidates = 1000;  
    std::string det_output_name = "output";  
    int det_map_h = 640;  
    int det_map_w = 640;  
};  
 
__BEGIN_DECLS  
void paddleocr_det(HailoROIPtr roi, void *params_void_ptr);  
void paddleocr_recognize(HailoROIPtr roi, void *params_void_ptr);  
void crop_text_regions_filter(HailoROIPtr roi, void *params_void_ptr);  // Add this line  
std::vector<HailoDetection> db_postprocess(cv::Mat heatmap, cv::Mat orig_img, OcrParams *params);  
std::string ctc_decode(const std::vector<std::vector<float>>& logits);  
__END_DECLS