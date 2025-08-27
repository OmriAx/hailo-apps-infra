#include "test.hpp"  
#include "vms_croppers.hpp" 

#include <opencv2/imgproc.hpp>  
#include <algorithm>  
#include <cmath>  
#include <opencv2/core.hpp>
#include <limits>
#include <numeric> 

  
void filter_db(HailoROIPtr roi) {  
    db_postprocess(roi, nullptr);  
} 

// Character set matching your Python implementation  
const std::vector<std::string> CHARACTERS = {  
    "blank", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@",   
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",   
    "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",   
    "{", "|", "}", "~", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", " ", " "  
};  
  
void filter_ocr(HailoROIPtr roi) {  
    ocr_postprocess(roi, nullptr);  
}  

// void filter_text(HailoROIPtr roi) {
//     crop_text_regions(roi, nullptr);
// }
  
void db_postprocess(HailoROIPtr roi, void *params) {
    std::cout << "DB postprocess called" << std::endl;   
    if (!roi->has_tensors()) {  
        return;  
    }  
  
    // Get default parameters or use provided ones  
    DBParams db_params;  
    if (params != nullptr) {  
        db_params = *static_cast<DBParams*>(params);  
    }  
  
    // Get the output tensor (heatmap)  
    std::vector<HailoTensorPtr> tensors = roi->get_tensors();  
    if (tensors.empty()) {  
        return;  
    }  
  
    HailoTensorPtr tensor = tensors[0]; // Assuming first tensor is the detection map  
      
    // Get tensor dimensions and data  
    auto shape = tensor->shape();  
    int height = shape[0];  
    int width = shape[1];  
      
    // Dequantize the tensor data  
    float qp_scale = tensor->quant_info().qp_scale;  
    float qp_zp = tensor->quant_info().qp_zp;  
      
    // Convert tensor data to OpenCV Mat  
    cv::Mat pred(height, width, CV_32F);  
    uint8_t* tensor_data = reinterpret_cast<uint8_t*>(tensor->data());  
      
    for (int i = 0; i < height * width; i++) {  
        pred.at<float>(i / width, i % width) = (tensor_data[i] - qp_zp) * qp_scale;  
    }  
  
    // Apply threshold to create binary mask  
    cv::Mat bitmap;  
    cv::threshold(pred, bitmap, db_params.thresh, 1.0, cv::THRESH_BINARY);  
    bitmap.convertTo(bitmap, CV_8U, 255.0);  
  
    // Apply dilation if specified  
    if (db_params.use_dilation) {  
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));  
        cv::dilate(bitmap, bitmap, kernel);  
    }  
  
    // Find contours  
    std::vector<std::vector<cv::Point>> contours;  
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);  
  
    std::vector<HailoDetection> detections;  
    int num_candidates = std::min(static_cast<int>(contours.size()), db_params.max_candidates);  
  
    for (int i = 0; i < num_candidates; i++) {  
        const auto& contour = contours[i];  
          
        // Get minimum area rectangle  
        float min_side;  
        std::vector<cv::Point2f> points = get_mini_boxes(contour, min_side);  
          
        if (min_side < db_params.min_size) {  
            continue;  
        }  
  
        // Calculate score  
        float score = box_score_fast(pred, points);  
        if (score < db_params.box_thresh) {  
            continue;  
        }  
  
        // Unclip the polygon  
        std::vector<cv::Point2f> unclipped_box = unclip_polygon(points, db_params.unclip_ratio);  
        if (unclipped_box.empty()) {  
            continue;  
        }  
  
        // Get final bounding box  
        float final_min_side;  
        std::vector<cv::Point2f> final_box = get_mini_boxes(  
            std::vector<cv::Point>(unclipped_box.begin(), unclipped_box.end()),   
            final_min_side  
        );  
          
        if (final_min_side < db_params.min_size + 2) {  
            continue;  
        }  
  
        // Convert to normalized coordinates (0-1 range)  
        std::vector<cv::Point2f> normalized_box;  
        for (const auto& point : final_box) {  
            normalized_box.push_back(cv::Point2f(  
                std::max(0.0f, std::min(1.0f, point.x / width)),  
                std::max(0.0f, std::min(1.0f, point.y / height))  
            ));  
        }  
  
        // Create bounding box from the four points  
        float xmin = std::min({normalized_box[0].x, normalized_box[1].x, normalized_box[2].x, normalized_box[3].x});  
        float ymin = std::min({normalized_box[0].y, normalized_box[1].y, normalized_box[2].y, normalized_box[3].y});  
        float xmax = std::max({normalized_box[0].x, normalized_box[1].x, normalized_box[2].x, normalized_box[3].x});  
        float ymax = std::max({normalized_box[0].y, normalized_box[1].y, normalized_box[2].y, normalized_box[3].y});  
  
        HailoBBox bbox(xmin, ymin, xmax - xmin, ymax - ymin);  
        HailoDetection detection(bbox, "text", score);  
        detections.push_back(detection);  
    }  
  
    hailo_common::add_detections(roi, detections);  
}  
  
std::vector<cv::Point2f> get_mini_boxes(const std::vector<cv::Point>& contour, float& min_side) {  
    cv::RotatedRect bounding_box = cv::minAreaRect(contour);  
    cv::Point2f vertices[4];  
    bounding_box.points(vertices);  
      
    // Sort points by x coordinate  
    std::vector<cv::Point2f> points(vertices, vertices + 4);  
    std::sort(points.begin(), points.end(), [](const cv::Point2f& a, const cv::Point2f& b) {  
        return a.x < b.x;  
    });  
  
    // Reorder points to match Python implementation  
    std::vector<cv::Point2f> ordered_points(4);  
    if (points[1].y > points[0].y) {  
        ordered_points[0] = points[0];  
        ordered_points[3] = points[1];  
    } else {  
        ordered_points[0] = points[1];  
        ordered_points[3] = points[0];  
    }  
      
    if (points[3].y > points[2].y) {  
        ordered_points[1] = points[2];  
        ordered_points[2] = points[3];  
    } else {  
        ordered_points[1] = points[3];  
        ordered_points[2] = points[2];  
    }  
  
    min_side = std::min(bounding_box.size.width, bounding_box.size.height);  
    return ordered_points;  
}  
  
float box_score_fast(const cv::Mat& bitmap, const std::vector<cv::Point2f>& box) {  
    int h = bitmap.rows;  
    int w = bitmap.cols;  
      
    // Find bounding rectangle  
    float xmin = std::numeric_limits<float>::max();  
    float xmax = std::numeric_limits<float>::lowest();  
    float ymin = std::numeric_limits<float>::max();  
    float ymax = std::numeric_limits<float>::lowest();  
      
    for (const auto& point : box) {  
        xmin = std::min(xmin, point.x);  
        xmax = std::max(xmax, point.x);  
        ymin = std::min(ymin, point.y);  
        ymax = std::max(ymax, point.y);  
    }  
      
    int x1 = std::max(0, std::min(w - 1, static_cast<int>(std::floor(xmin))));  
    int x2 = std::max(0, std::min(w - 1, static_cast<int>(std::ceil(xmax))));  
    int y1 = std::max(0, std::min(h - 1, static_cast<int>(std::floor(ymin))));  
    int y2 = std::max(0, std::min(h - 1, static_cast<int>(std::ceil(ymax))));  
  
    // Create mask  
    cv::Mat mask = cv::Mat::zeros(y2 - y1 + 1, x2 - x1 + 1, CV_8U);  
      
    // Adjust box coordinates relative to the cropped region  
    std::vector<cv::Point> mask_box;  
    for (const auto& point : box) {  
        mask_box.push_back(cv::Point(  
            static_cast<int>(point.x) - x1,  
            static_cast<int>(point.y) - y1  
        ));  
    }  
      
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{mask_box}, cv::Scalar(255));  
      
    // Calculate mean score in the masked region  
    cv::Scalar mean_val = cv::mean(bitmap(cv::Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1)), mask);  
    return static_cast<float>(mean_val[0]);  
}  
  
std::vector<cv::Point2f> unclip_polygon(const std::vector<cv::Point2f>& box, float unclip_ratio) {  
    // Simplified polygon expansion - in production you might want to use a proper clipping library  
    // This is a basic implementation that expands the bounding box  
      
    // Calculate centroid  
    cv::Point2f centroid(0, 0);  
    for (const auto& point : box) {  
        centroid += point;  
    }  
    centroid *= (1.0f / box.size());  
      
    // Expand each point away from centroid  
    std::vector<cv::Point2f> expanded_box;  
    for (const auto& point : box) {  
        cv::Point2f direction = point - centroid;  
        float length = cv::norm(direction);  
        if (length > 0) {  
            direction /= length;  
            cv::Point2f expanded_point = point + direction * (length * unclip_ratio * 0.1f);  
            expanded_box.push_back(expanded_point);  
        } else {  
            expanded_box.push_back(point);  
        }  
    }  
      
    return expanded_box;  
}

void ocr_postprocess(HailoROIPtr roi, void *params) {  
    std::cout << "OCR postprocess called" << std::endl;  
      
    if (!roi->has_tensors()) {  
        std::cout << "No tensors found in ROI" << std::endl;  
        return;  
    }  
  
    std::vector<HailoTensorPtr> tensors = roi->get_tensors();  
    if (tensors.empty()) {  
        std::cout << "No tensors available" << std::endl;  
        return;  
    }  
  
    HailoTensorPtr tensor = tensors[0]; // OCR output tensor  
    auto shape = tensor->shape();  
      
    std::cout << "Tensor shape: ";  
    for (size_t i = 0; i < shape.size(); i++) {  
        std::cout << shape[i] << " ";  
    }  
    std::cout << std::endl;  
      
    // Determine tensor dimensions  
    int sequence_length, num_classes;  
    if (shape.size() == 3) {  
        // Batch dimension present: [batch, sequence_length, num_classes]  
        sequence_length = shape[1];  
        num_classes = shape[2];  
    } else if (shape.size() == 2) {  
        // No batch dimension: [sequence_length, num_classes]  
        sequence_length = shape[0];  
        num_classes = shape[1];  
    } else {  
        std::cout << "Unexpected tensor shape size: " << shape.size() << std::endl;  
        return;  
    }  
  
    std::cout << "Sequence length: " << sequence_length << ", Num classes: " << num_classes << std::endl;  
  
    // Dequantize tensor data  
    float qp_scale = tensor->quant_info().qp_scale;  
    float qp_zp = tensor->quant_info().qp_zp;  
      
    std::cout << "Quantization - Scale: " << qp_scale << ", Zero point: " << qp_zp << std::endl;  
      
    std::vector<float> dequantized_data(sequence_length * num_classes);  
    uint8_t* tensor_data = reinterpret_cast<uint8_t*>(tensor->data());  
      
    for (int i = 0; i < sequence_length * num_classes; i++) {  
        dequantized_data[i] = (tensor_data[i] - qp_zp) * qp_scale;  
    }  
  
    // Decode the OCR output  
    OCRResult result = decode_ocr_output(dequantized_data.data(), sequence_length, num_classes);  
      
    std::cout << "Decoded text: '" << result.text << "' with confidence: " << result.confidence << std::endl;  
  
    // Get detection objects from the detection stage (following cascaded pattern)  
    auto detections = roi->get_objects_typed(HAILO_DETECTION);  
    std::cout << "Found " << detections.size() << " detections" << std::endl;  
      
    if (!detections.empty()) {  
        // Attach classification to the detection object (cascaded approach)  
        auto detection = std::dynamic_pointer_cast<HailoDetection>(detections[0]);  
        if (detection) {  
            // Check if there's already a classification and replace if confidence is higher  
            auto existing_classifications = detection->get_objects_typed(HAILO_CLASSIFICATION);  
            bool should_add = true;  
              
            for (auto& existing_class : existing_classifications) {  
                auto classification_ptr = std::dynamic_pointer_cast<HailoClassification>(existing_class);  
                if (classification_ptr && classification_ptr->get_type() == HAILO_CLASSIFICATION) {
                    if (classification_ptr->get_confidence() < result.confidence) {  
                        // Remove existing classification with lower confidence  
                        detection->remove_object(existing_class);  
                        std::cout << "Removed existing classification with lower confidence" << std::endl;  
                    } else {  
                        should_add = false;  
                        std::cout << "Keeping existing classification with higher confidence" << std::endl;  
                    }  
                    break;  
                }  
            }  
              
            if (should_add && !result.text.empty()) {  
                auto classification = std::make_shared<HailoClassification>("text", result.text, result.confidence);  
                detection->add_object(classification);  
                std::cout << "Added classification '" << result.text << "' to detection with confidence " << result.confidence << std::endl;  
            }  
        }  
    } else {  
        std::cout << "No detections found to attach classification to" << std::endl;  
    }  
} 
  
OCRResult decode_ocr_output(const float* logits, int sequence_length, int num_classes) {  
    std::vector<int> text_indices(sequence_length);  
    std::vector<float> text_probs(sequence_length);  
      
    // Find argmax and max probability for each position in sequence  
    for (int i = 0; i < sequence_length; i++) {  
        int max_idx = 0;  
        float max_val = logits[i * num_classes];  
          
        for (int j = 1; j < num_classes; j++) {  
            float current_val = logits[i * num_classes + j];  
            if (current_val > max_val) {  
                max_val = current_val;  
                max_idx = j;  
            }  
        }  
          
        text_indices[i] = max_idx;  
        text_probs[i] = max_val;  
    }  
      
    // Apply CTC-like decoding - remove consecutive duplicates and blank tokens  
    std::vector<bool> selection(sequence_length, true);  
      
    // Remove consecutive duplicates  
    for (int i = 1; i < sequence_length; i++) {  
        if (text_indices[i] == text_indices[i-1]) {  
            selection[i] = false;  
        }  
    }  
      
    // Remove blank tokens (index 0)  
    for (int i = 0; i < sequence_length; i++) {  
        if (text_indices[i] == 0) {  
            selection[i] = false;  
        }  
    }  
      
    // Build character list and confidence list  
    std::vector<std::string> char_list;  
    std::vector<float> conf_list;  
      
    for (int i = 0; i < sequence_length; i++) {  
        if (selection[i] && text_indices[i] < static_cast<int>(CHARACTERS.size())) {  
            char_list.push_back(CHARACTERS[text_indices[i]]);  
            conf_list.push_back(text_probs[i]);  
        }  
    }  
      
    // Calculate mean confidence  
    float mean_confidence = 0.0f;  
    if (!conf_list.empty()) {  
        float sum = 0.0f;  
        for (float conf : conf_list) {  
            sum += conf;  
        }  
        mean_confidence = sum / conf_list.size();  
    }  
      
    // Join characters to form text  
    std::string decoded_text;  
    for (const auto& ch : char_list) {  
        decoded_text += ch;  
    }  
      
    std::cout << "CTC decoding: " << char_list.size() << " characters, confidence: " << mean_confidence << std::endl;  
      
    return {decoded_text, mean_confidence};  
}

std::vector<HailoROIPtr> crop_text_regions(std::shared_ptr<HailoMat> image, HailoROIPtr roi)  
{  
        std::cout << "Crop Text Regoins postprocess called" << std::endl;  
    std::vector<HailoROIPtr> crop_rois;  
      
    // Get all text detections from the detection model  
    std::vector<HailoDetectionPtr> detections_ptrs = hailo_common::get_hailo_detections(roi);  
      
    for (HailoDetectionPtr &detection : detections_ptrs)  
    {  
        if (std::string("text") == detection->get_label())  
        {  
            HailoBBox bbox = detection->get_bbox();  
            HailoBBox adjusted_bbox = adjust_text_bbox(bbox);  
              
            // Use the existing clone_detection_for_ocr function  
            HailoDetectionPtr new_roi = clone_detection_for_ocr(detection);  
            new_roi->set_bbox(adjusted_bbox);  
              
            crop_rois.emplace_back(new_roi);  
        }  
    }  
      
    return crop_rois;  
}

HailoBBox adjust_text_bbox(const HailoBBox &bbox, float padding_ratio)  
{  
    // Add padding around the text region for better recognition  
    float padding_w = bbox.width() * padding_ratio;  
    float padding_h = bbox.height() * padding_ratio;  
      
    float xmin = std::max(0.0f, bbox.xmin() - padding_w);  
    float ymin = std::max(0.0f, bbox.ymin() - padding_h);  
    float width = std::min(1.0f - xmin, bbox.width() + 2 * padding_w);  
    float height = std::min(1.0f - ymin, bbox.height() + 2 * padding_h);  
      
    return HailoBBox(xmin, ymin, width, height);  
}

HailoDetectionPtr clone_detection_for_ocr(HailoDetectionPtr detection)  
{  
    return std::make_shared<HailoDetection>(  
        detection->get_bbox(),   
        detection->get_label(),   
        detection->get_confidence()  
    );  
}