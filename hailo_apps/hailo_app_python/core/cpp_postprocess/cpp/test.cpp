#include "ocr_postprocess.hpp"  
#include <opencv2/imgproc.hpp>  
  
const std::vector<std::string> CHARACTERS = {  
    "blank", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@",  
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",  
    "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",  
    "{", "|", "}", "~", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", " "  
};  
  
cv::Mat resize_heatmap_to_original(const cv::Mat& heatmap, cv::Size original_size, int model_w, int model_h) {  
    int orig_h = original_size.height;  
    int orig_w = original_size.width;  
      
    float scale = std::min(static_cast<float>(model_w) / orig_w, static_cast<float>(model_h) / orig_h);  
    int new_w = static_cast<int>(orig_w * scale);  
    int new_h = static_cast<int>(orig_h * scale);  
      
    int x_offset = (model_w - new_w) / 2;  
    int y_offset = (model_h - new_h) / 2;  
      
    // Crop the heatmap to remove padding  
    cv::Rect crop_rect(x_offset, y_offset, new_w, new_h);  
    cv::Mat cropped_heatmap = heatmap(crop_rect);  
      
    // Resize back to original image size  
    cv::Mat resized_heatmap;  
    cv::resize(cropped_heatmap, resized_heatmap, original_size, 0, 0, cv::INTER_CUBIC);  
      
    return resized_heatmap;  
}  
std::vector<cv::Point> unclip_polygon(const std::vector<cv::Point>& polygon, float unclip_ratio) {  
    if (polygon.empty()) return polygon;  
      
    // Get bounding rect to determine appropriate mask size  
    cv::Rect bounds = cv::boundingRect(polygon);  
    int margin = static_cast<int>(unclip_ratio * 10);  // Add margin for dilation  
    cv::Size mask_size(bounds.width + 2*margin, bounds.height + 2*margin);  
      
    // Use dynamic sizing instead of fixed 1000x1000  
    cv::Mat mask = cv::Mat::zeros(mask_size, CV_8UC1);  
      
    // Adjust polygon coordinates to mask coordinate system  
    std::vector<cv::Point> adjusted_polygon;  
    for (const auto& pt : polygon) {  
        adjusted_polygon.push_back(cv::Point(pt.x - bounds.x + margin, pt.y - bounds.y + margin));  
    }  
      
    std::vector<std::vector<cv::Point>> contours = {adjusted_polygon};  
    cv::fillPoly(mask, contours, cv::Scalar(255));  
      
    int kernel_size = static_cast<int>(unclip_ratio * 2);  
    if (kernel_size > 0) {  
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));  
        cv::Mat dilated;  
        cv::dilate(mask, dilated, kernel);  
          
        std::vector<std::vector<cv::Point>> new_contours;  
        cv::findContours(dilated, new_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);  
          
        if (!new_contours.empty()) {  
            // Convert back to original coordinate system  
            std::vector<cv::Point> result;  
            for (const auto& pt : new_contours[0]) {  
                result.push_back(cv::Point(pt.x + bounds.x - margin, pt.y + bounds.y - margin));  
            }  
            return result;  
        }  
    }  
      
    return polygon;  
}
  
std::vector<HailoDetection> db_postprocess(cv::Mat heatmap, cv::Mat orig_img, OcrParams *params) {  
    std::vector<HailoDetection> detections;  
      
    // Resize heatmap to original image size  
    cv::Mat resized_heatmap = resize_heatmap_to_original(heatmap, orig_img.size(), params->det_map_w, params->det_map_h);  
      
    // Binarize the heatmap  
    cv::Mat binary;  
    cv::threshold(resized_heatmap, binary, params->det_bin_thresh * 255, 255, cv::THRESH_BINARY);  
    binary.convertTo(binary, CV_8U);  
      
    // Find contours  
    std::vector<std::vector<cv::Point>> contours;  
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);  
      
    for (const auto& contour : contours) {  
        if (contour.size() < 4) continue;  
          
        // Calculate contour score (mean intensity in original heatmap)  
        cv::Mat mask = cv::Mat::zeros(resized_heatmap.size(), CV_8UC1);  
        cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{contour}, cv::Scalar(255));  
        cv::Scalar mean_val = cv::mean(resized_heatmap, mask);  
        float score = mean_val[0] / 255.0f;  
          
        if (score < params->det_box_thresh) continue;  
          
        // Unclip the polygon  
        std::vector<cv::Point> unclipped = unclip_polygon(contour, params->det_unclip_ratio);  
          
        // Get bounding rectangle  
        cv::Rect bbox = cv::boundingRect(unclipped);  
          
        // Convert to normalized coordinates  
        float xmin = static_cast<float>(bbox.x) / orig_img.cols;  
        float ymin = static_cast<float>(bbox.y) / orig_img.rows;  
        float width = static_cast<float>(bbox.width) / orig_img.cols;  
        float height = static_cast<float>(bbox.height) / orig_img.rows;  
          
        HailoBBox hailo_bbox(xmin, ymin, width, height);  
        detections.emplace_back(hailo_bbox, "text_region", score);  
          
        if (detections.size() >= static_cast<size_t>(params->det_max_candidates)) break;  
    }  
      
    return detections;  
}  
  
extern "C"  
void paddleocr_det(HailoROIPtr roi, void *params_void_ptr) {  
    if (!roi->has_tensors()) {  
        return;  
    }  
      
    auto *params = reinterpret_cast<OcrParams *>(params_void_ptr);  
      
    // Get the detection tensor  
    HailoTensorPtr tensor = roi->get_tensor(params->det_output_name);  
    if (!tensor) {  
        std::cout << "DEBUG: No tensor found with name: " << params->det_output_name << std::endl;  
        return;  
    }  
      
    // Convert tensor to OpenCV Mat  
    auto tensor_data = tensor->data();  
    int height = tensor->height();  
    int width = tensor->width();  
    int channels = tensor->features();  
      
    cv::Mat heatmap(height, width, CV_8UC1, tensor_data);  
    heatmap.convertTo(heatmap, CV_32F, 1.0/255.0);  
      
    // Create dummy original image size (you'll need to get this from somewhere)  
    cv::Mat orig_img = cv::Mat::zeros(params->det_map_h, params->det_map_w, CV_8UC3);  
      
    // Apply DB post-processing  
    std::vector<HailoDetection> detections = db_postprocess(heatmap, orig_img, params);  
      
    std::cout << "DEBUG: Created " << detections.size() << " text region detections" << std::endl;  
      
    // Add detections to ROI  
    hailo_common::add_detections(roi, detections);  
}

std::string ctc_decode(const std::vector<std::vector<float>>& logits) {  
    std::string result;  
    int prev_idx = -1;  
      
    for (const auto& timestep : logits) {  
        // Find the character with maximum probability  
        int max_idx = 0;  
        float max_prob = timestep[0];  
        for (size_t i = 1; i < timestep.size(); ++i) {  
            if (timestep[i] > max_prob) {  
                max_prob = timestep[i];  
                max_idx = static_cast<int>(i);  
            }  
        }  
          
        // CTC decoding: skip blanks (index 0) and consecutive duplicates  
        if (max_idx != 0 && max_idx != prev_idx) {  
            if (max_idx < static_cast<int>(CHARACTERS.size())) {  
                result += CHARACTERS[max_idx];  
            }  
        }  
        prev_idx = max_idx;  
    }  
      
    return result;  
}  
  
extern "C"  
void paddleocr_recognize(HailoROIPtr roi, void *params_void_ptr) {  
    if (!roi->has_tensors()) {  
        return;  
    }  
      
    // Get recognition tensor  
    std::vector<HailoTensorPtr> tensors = roi->get_tensors();  
    if (tensors.empty()) {  
        return;  
    }  
      
    HailoTensorPtr tensor = tensors[0];  
      
    // Convert tensor to logits format  
    auto tensor_data = reinterpret_cast<float*>(tensor->data());  
    int seq_len = tensor->width();  
    int num_classes = tensor->features();  
      
    std::vector<std::vector<float>> logits(seq_len, std::vector<float>(num_classes));  
    for (int t = 0; t < seq_len; ++t) {  
        for (int c = 0; c < num_classes; ++c) {  
            logits[t][c] = tensor_data[t * num_classes + c];  
        }  
    }  
      
    // Decode text using CTC  
    std::string decoded_text = ctc_decode(logits);  
      
    std::cout << "DEBUG: Decoded text: '" << decoded_text << "'" << std::endl;  
      
    // Add as classification to detections  
    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);  
    for (auto detection : detections) {  
        HailoClassification classification("license_plate", decoded_text, 1.0f);  
        detection->add_object(std::make_shared<HailoClassification>(classification));  
    }  
}

extern "C"  
void crop_text_regions_filter(HailoROIPtr roi, void *params_void_ptr) {  
    std::cout << "DEBUG: crop_text_regions_filter called as hailofilter" << std::endl;  
      
    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);  
    std::cout << "DEBUG: Found " << detections.size() << " detections in ROI" << std::endl;  
      
    std::vector<HailoDetectionPtr> text_detections;  
      
    for (auto detection : detections) {  
        std::string label = detection->get_label();  
        std::cout << "DEBUG: Processing detection with label: '" << label << "'" << std::endl;  
          
        // Only keep text_region detections, filter out others  
        if (label == "text_region") {  
            std::cout << "DEBUG: Found text_region detection - keeping for OCR" << std::endl;  
            text_detections.push_back(detection);  
        } else {  
            std::cout << "DEBUG: Skipping detection with label: '" << label << "'" << std::endl;  
        }  
    }  
      
    // Remove all detections first  
    roi->remove_objects_typed(HAILO_DETECTION);  
      
    // Add back only text region detections  
    for (auto text_detection : text_detections) {  
        roi->add_object(text_detection);  
    }  
      
    std::cout << "DEBUG: Filtered to " << text_detections.size() << " text region detections" << std::endl;  
}
