#include "ocr_postprocess.hpp"  
  
#include <cstdio>  
#include <cstring>  
#include <cmath>  
#include <fstream>  
#include <algorithm>  
#include <stdexcept>  
#include <iostream>  
  
#include <opencv2/imgproc.hpp>  
#include <opencv2/imgcodecs.hpp>  
  
// RapidJSON for optional config  
#include "rapidjson/document.h"  
#include "rapidjson/error/en.h"  
#include "rapidjson/filereadstream.h"  
#include "rapidjson/schema.h"  
  
#if __GNUC__ > 8  
  #include <filesystem>  
  namespace fs = std::filesystem;  
#else  
  #include <experimental/filesystem>  
  namespace fs = std::experimental::filesystem;  
#endif  
  
// ---------------------------  
// JSON helpers  
// ---------------------------  
static bool validate_json_with_schema(FILE *fp, const char *schema) {  
    char buffer[1 << 12];  
    rapidjson::FileReadStream is(fp, buffer, sizeof(buffer));  
  
    rapidjson::Document sd; sd.Parse(schema);  
    if (sd.HasParseError()) return false;  
  
    rapidjson::SchemaDocument sdoc(sd);  
    rapidjson::SchemaValidator validator(sdoc);  
  
    fseek(fp, 0, SEEK_SET);  
    rapidjson::FileReadStream is2(fp, buffer, sizeof(buffer));  
    rapidjson::Document d; d.ParseStream(is2);  
    if (d.HasParseError()) return false;  
  
    return d.Accept(validator);  
}  
  
static void load_default_charset(OcrParams &p) {  
    // Based on your Python CHARACTERS array - add blank first, then 0-9, symbols, A-Z, a-z  
    p.charset.emplace_back("blank");  // blank token at index 0  
    for (char c='0'; c<='9'; ++c) p.charset.emplace_back(1, c);  
    // Add common symbols found in license plates  
    p.charset.emplace_back(":");  
    p.charset.emplace_back(";");  
    p.charset.emplace_back("<");  
    p.charset.emplace_back("=");  
    p.charset.emplace_back(">");  
    p.charset.emplace_back("?");  
    p.charset.emplace_back("@");  
    for (char c='A'; c<='Z'; ++c) p.charset.emplace_back(1, c);  
    p.charset.emplace_back("[");  
    p.charset.emplace_back("\\");  
    p.charset.emplace_back("]");  
    p.charset.emplace_back("^");  
    p.charset.emplace_back("_");  
    p.charset.emplace_back("`");  
    for (char c='a'; c<='z'; ++c) p.charset.emplace_back(1, c);  
    p.charset.emplace_back("{");  
    p.charset.emplace_back("|");  
    p.charset.emplace_back("}");  
    p.charset.emplace_back("~");  
    p.charset.emplace_back("!");  
    p.charset.emplace_back("\"");  
    p.charset.emplace_back("#");  
    p.charset.emplace_back("$");  
    p.charset.emplace_back("%");  
    p.charset.emplace_back("&");  
    p.charset.emplace_back("'");  
    p.charset.emplace_back("(");  
    p.charset.emplace_back(")");  
    p.charset.emplace_back("*");  
    p.charset.emplace_back("+");  
    p.charset.emplace_back(",");  
    p.charset.emplace_back("-");  
    p.charset.emplace_back(".");  
    p.charset.emplace_back("/");  
    p.charset.emplace_back(" ");  
}  
  
static void load_charset_from_file(OcrParams &p) {  
    if (p.charset_path.empty()) { load_default_charset(p); return; }  
    std::ifstream in(p.charset_path);  
    if (!in.is_open()) throw std::runtime_error("Failed to open charset file: " + p.charset_path);  
    std::string line;  
    while (std::getline(in, line)) p.charset.push_back(line);  
    if (p.charset.empty()) load_default_charset(p);  
}  
  
// ---------------------------  
// init / free_resources  
// ---------------------------  
OcrParams *init(const std::string config_path, const std::string /*function_name*/) {  
    auto *params = new OcrParams();  
  
    if (fs::exists(config_path)) {  
        const char *schema = R""""({  
          "$schema": "http://json-schema.org/draft-04/schema#",  
          "type": "object",  
          "properties": {  
            "det_bin_thresh":     { "type": "number" },  
            "det_box_thresh":     { "type": "number" },  
            "det_unclip_ratio":   { "type": "number" },  
            "det_max_candidates": { "type": "integer" },  
            "det_min_box_size":   { "type": "number" },  
            "det_output_name":    { "type": "string" },  
            "det_map_h":          { "type": "integer" },  
            "det_map_w":          { "type": "integer" },  
            "letterbox_fix":      { "type": "boolean" },  
  
            "rec_output_name":    { "type": "string" },  
            "charset_path":       { "type": "string" },  
            "blank_index":        { "type": "integer" },  
            "logits_are_softmax": { "type": "boolean" },  
            "time_major":         { "type": "boolean" },  
            "text_conf_smooth":   { "type": "number" },  
            "attach_caption_box": { "type": "boolean" }  
          }  
        })"""";  
  
        FILE *fp = fopen(config_path.c_str(), "r");  
        if (!fp) throw std::runtime_error("JSON config file cannot be opened");  
        bool ok = validate_json_with_schema(fp, schema);  
        if (!ok) { fclose(fp); throw std::runtime_error("JSON config doesn't match schema"); }  
        fseek(fp, 0, SEEK_SET);  
  
        char buffer[1 << 14];  
        rapidjson::FileReadStream frs(fp, buffer, sizeof(buffer));  
        rapidjson::Document d; d.ParseStream(frs);  
        fclose(fp);  
  
        auto getf=[&](const char* k, float &dst){ if (d.HasMember(k)) dst = d[k].GetFloat(); };  
        auto geti=[&](const char* k, int &dst){ if (d.HasMember(k)) dst = d[k].GetInt(); };  
        auto getb=[&](const char* k, bool &dst){ if (d.HasMember(k)) dst = d[k].GetBool(); };  
        auto gets=[&](const char* k, std::string &dst){ if (d.HasMember(k)) dst = d[k].GetString(); };  
  
        getf("det_bin_thresh", params->det_bin_thresh);  
        getf("det_box_thresh", params->det_box_thresh);  
        getf("det_unclip_ratio", params->det_unclip_ratio);  
        geti("det_max_candidates", params->det_max_candidates);  
        getf("det_min_box_size", params->det_min_box_size);  
        gets("det_output_name", params->det_output_name);  
        geti("det_map_h", params->det_map_h);  
        geti("det_map_w", params->det_map_w);  
        getb("letterbox_fix", params->letterbox_fix);  
  
        gets("rec_output_name", params->rec_output_name);  
        gets("charset_path", params->charset_path);  
        geti("blank_index", params->blank_index);  
        getb("logits_are_softmax", params->logits_are_softmax);  
        getb("time_major", params->time_major);  
        getf("text_conf_smooth", params->text_conf_smooth);  
        getb("attach_caption_box", params->attach_caption_box);  
    }  
  
    load_charset_from_file(*params);  
    return params;  
}  
  
void free_resources(void *params_void_ptr) {  
    auto *p = reinterpret_cast<OcrParams *>(params_void_ptr);  
    delete p;  
}  
  
// ---------------------------  
// Tensor helpers (typed access)  
// ---------------------------  
static cv::Mat tensor_to_probmap_u8_as_float(HailoTensorPtr t, int H, int W) {  
    const uint8_t *u8 = reinterpret_cast<const uint8_t*>(t->data());  
    if (!u8) throw std::runtime_error("Detector tensor has null data()");  
    cv::Mat prob(H, W, CV_8UC1);  
    std::memcpy(prob.data, u8, (size_t)H * (size_t)W * sizeof(uint8_t));  
    cv::Mat out; prob.convertTo(out, CV_32F, 1.0/255.0);  
    return out;  
}  
  
static HailoTensorPtr get_tensor_by_name_or_fallback(const HailoROIPtr &roi, const std::string &desired) {  
    HailoTensorPtr chosen;  
    for (auto &t : roi->get_tensors()) { if (t->name() == desired) { chosen = t; break; } }  
    if (!chosen) {  
        auto tensors = roi->get_tensors();  
        if (tensors.empty()) throw std::runtime_error("ROI has no tensors");  
        chosen = tensors.front();  
    }  
    return chosen;  
}  
  
// ---------------------------  
// Detector (DB-like) postprocess - Enhanced based on Python DBPostProcess  
// ---------------------------  
static float region_score(const cv::Mat &prob, const std::vector<cv::Point> &poly) {  
    cv::Rect bbox = cv::boundingRect(poly) & cv::Rect(0,0,prob.cols,prob.rows);  
    if (bbox.empty()) return 0.f;  
    cv::Mat mask = cv::Mat::zeros(bbox.size(), CV_8UC1);  
    std::vector<std::vector<cv::Point>> polys(1);  
    polys[0].reserve(poly.size());  
    for (auto &p : poly) polys[0].push_back(cv::Point(p.x - bbox.x, p.y - bbox.y));  
    cv::fillPoly(mask, polys, cv::Scalar(255));  
    cv::Scalar s = cv::mean(prob(bbox), mask);  
    return (float)s[0];  
}  
  
// Unclip polygon using offset (similar to Python's pyclipper)  
static std::vector<cv::Point> unclip_polygon(const std::vector<cv::Point>& polygon, float unclip_ratio) {  
    // Simple approximation of polygon expansion  
    cv::Rect bbox = cv::boundingRect(polygon);  
    float area = cv::contourArea(polygon);  
    float length = cv::arcLength(polygon, true);  
    float distance = area * unclip_ratio / length;  
      
    // Expand bounding box by distance  
    int expand = (int)distance;  
    std::vector<cv::Point> expanded;  
    for (const auto& pt : polygon) {  
        cv::Point new_pt;  
        new_pt.x = pt.x + (pt.x > bbox.x + bbox.width/2 ? expand : -expand);  
        new_pt.y = pt.y + (pt.y > bbox.y + bbox.height/2 ? expand : -expand);  
        expanded.push_back(new_pt);  
    }  
    return expanded;  
}  

extern "C"  
void paddleocr_det(HailoROIPtr roi, void *params_void_ptr) {  
    if (!roi->has_tensors()) {  
        std::cout << "DEBUG: paddleocr_det - No tensors in ROI" << std::endl;  
        return;  
    }  
    auto *p = reinterpret_cast<OcrParams *>(params_void_ptr);  
    std::cout << "DEBUG: paddleocr_det called" << std::endl;  
      
    std::cout << "DEBUG: Using thresholds - bin:" << p->det_bin_thresh   
              << " box:" << p->det_box_thresh   
              << " min_size:" << p->det_min_box_size << std::endl;  
  
    // 1) fetch prob-map (UINT8 -> float [0..1])  
    HailoTensorPtr t = get_tensor_by_name_or_fallback(roi, p->det_output_name);  
    const int H = p->det_map_h;  
    const int W = p->det_map_w;  
    cv::Mat prob = tensor_to_probmap_u8_as_float(t, H, W);  
  
    // 2) threshold to binary  
    cv::Mat bin;   
    cv::threshold(prob, bin, p->det_bin_thresh, 1.0, cv::THRESH_BINARY);  
    bin.convertTo(bin, CV_8U, 255.0);  
  
    // 3) contours  
    std::vector<std::vector<cv::Point>> contours;  
    cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);  
    std::cout << "DEBUG: Found " << contours.size() << " contours" << std::endl;  
  
    // 4) build detections (bbox only) - BYPASS MODE  
    std::vector<HailoDetection> outs;  
    outs.reserve(std::min((int)contours.size(), p->det_max_candidates));  
  
    // ROI letterbox region in original frame  
    HailoBBox roi_box = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());  
    const float sx = roi_box.width()  / (float)W;  
    const float sy = roi_box.height() / (float)H;  
  
    int pushed = 0;  
    int contour_count = 0;  
    for (auto &c : contours) {  
        contour_count++;  
        std::cout << "DEBUG: Processing contour " << contour_count << " with " << c.size() << " points" << std::endl;  
          
        // More lenient point count filter - allow even single points for now  
        if ((int)c.size() < 1) {  
            std::cout << "DEBUG: Contour " << contour_count << " rejected - no points" << std::endl;  
            continue;  
        }  
  
        // For single points or lines, use bounding rect instead of minAreaRect  
        cv::Rect contour_bb = cv::boundingRect(c);  // Renamed from 'bb' to 'contour_bb'  
        float short_side = std::min(contour_bb.width, contour_bb.height);  
        std::cout << "DEBUG: Contour " << contour_count << " short_side: " << short_side << std::endl;  
          
        // Very permissive size filter - even allow zero-size for testing  
        if (short_side < 0) {  // Only reject negative sizes  
            std::cout << "DEBUG: Contour " << contour_count << " rejected - invalid size" << std::endl;  
            continue;  
        }  
  
        float score = region_score(prob, c);  
        std::cout << "DEBUG: Contour " << contour_count << " score: " << score << std::endl;  
          
        // Keep your permissive score threshold  
        if (score < p->det_box_thresh) {  
            std::cout << "DEBUG: Contour " << contour_count << " rejected - low score" << std::endl;  
            continue;  
        }  
  
        std::cout << "DEBUG: Contour " << contour_count << " ACCEPTED - creating detection" << std::endl;  
  
        // Unclip the polygon (expand it)  
        std::vector<cv::Point> unclipped = unclip_polygon(c, p->det_unclip_ratio);  
          
        // Use bounding rect of unclipped polygon  
        cv::Rect final_bb = cv::boundingRect(unclipped);  // Renamed from 'bb' to 'final_bb'  
  
        // Map to frame coords (letterbox fix)  
        float xmin = final_bb.x * sx + roi_box.xmin();  
        float ymin = final_bb.y * sy + roi_box.ymin();  
        float w    = final_bb.width  * sx;  
        float h    = final_bb.height * sy;  
  
        outs.emplace_back(HailoBBox(xmin, ymin, w, h), std::string("text_region"), score);  
        if (++pushed >= p->det_max_candidates) break;  
    }  
  
    std::cout << "DEBUG: Created " << outs.size() << " text region detections from " << contours.size() << " contours" << std::endl;  
  
    if (!outs.empty()) {  
        hailo_common::add_detections(roi, outs);  
        if (p->letterbox_fix) roi->clear_scaling_bbox();  
    }  
}

// ---------------------------  
// Recognizer (CTC greedy) - Enhanced based on Python implementation  
// ---------------------------  
static void softmax1d(std::vector<float> &v) {  
    float m = *std::max_element(v.begin(), v.end());  
    double sum = 0.0;  
    for (float &x : v) sum += std::exp(double(x - m));  
    for (float &x : v) x = float(std::exp(double(x - m)) / sum);  
}  
  
extern "C"  
void paddleocr_recognize(HailoROIPtr roi, void *params_void_ptr) {  
    if (!roi->has_tensors()) {  
        std::cout << "DEBUG: paddleocr_recognize - No tensors in ROI" << std::endl;  
        return;  
    }  
    auto *p = reinterpret_cast<OcrParams *>(params_void_ptr);  
    std::cout << "DEBUG: paddleocr_recognize called" << std::endl;  
  
    HailoTensorPtr t = get_tensor_by_name_or_fallback(roi, p->rec_output_name);  
    const auto &shape = t->shape(); // FCR(1x40x97) => size()==3  
    if (shape.size() != 3) throw std::runtime_error("Unexpected recognizer rank (expected 3)");  
  
    // Pull UINT8 -> float probs in [0..1]  
    const uint8_t *u8 = reinterpret_cast<const uint8_t*>(t->data());  
    if (!u8) throw std::runtime_error("Recognizer tensor not UINT8");  
    const size_t N = shape[0];                  // 1  
    const size_t D1 = shape[1];                 // 40 or 97  
    const size_t D2 = shape[2];                 // 97 or 40  
    if (N != 1) throw std::runtime_error("Recognizer expects N=1");  
  
    // Heuristic: treat larger of (D1,D2) as T (timesteps), smaller as C (classes)  
    size_t C = std::min(D1, D2);  
    size_t T = std::max(D1, D2);  
    bool layout_is_NCT = (D1 == C && D2 == T);  // [N, C, T]  
      
    std::cout << "DEBUG: Tensor shape - N:" << N << " D1:" << D1 << " D2:" << D2 << " C:" << C << " T:" << T << std::endl;  
      
    // Build probs[T][C]  
    std::vector<std::vector<float>> probs(T, std::vector<float>(C));  
  
    // copy & normalize  
    const uint8_t *base = u8; // N=1, contiguous  
    if (layout_is_NCT) {  
        // [1, C, T]  
        for (size_t c=0; c<C; ++c) {  
            for (size_t t0=0; t0<T; ++t0) {  
                float v = base[c*T + t0] * (1.0f/255.0f);  
                probs[t0][c] = p->logits_are_softmax ? v : v; // if not softmax, will softmax below  
            }  
        }  
    } else {  
        // [1, T, C]  
        for (size_t t0=0; t0<T; ++t0) {  
            for (size_t c=0; c<C; ++c) {  
                float v = base[t0*C + c] * (1.0f/255.0f);  
                probs[t0][c] = p->logits_are_softmax ? v : v;  
            }  
        }  
    }  
  
    if (!p->logits_are_softmax) {  
        for (size_t t0=0; t0<T; ++t0) softmax1d(probs[t0]);  
    }  
  
    // Greedy CTC decode - similar to Python ocr_eval_postprocess  
    std::string out_text;  
    out_text.reserve(T);  
    float conf_sum = 0.f;   
    int conf_count = 0;  
    int prev = -1;  
      
    for (size_t t0=0; t0<T; ++t0) {  
        auto &row = probs[t0];  
        auto it = std::max_element(row.begin(), row.end());  
        int idx = int(std::distance(row.begin(), it));  
        float pmax = *it;  
          
        // CTC decoding: skip blank and repeated characters  
        if (idx != p->blank_index && idx != prev) {  
            if (idx >= 0 && idx < (int)p->charset.size()) {  
                out_text += p->charset[idx];  
            } else {  
                out_text += "?";  
            }  
            conf_sum += pmax;  
            conf_count++;  
        }  
        prev = idx;  
    }  
      
    float conf = (conf_count > 0) ? (conf_sum / (float)conf_count) : 0.f;  
      
    std::cout << "DEBUG: Decoded text: '" << out_text << "' confidence: " << conf << std::endl;  
  
    // Create classification objects and attach to existing detections  
    if (!out_text.empty() && out_text != " ") {  
        // Get existing detections from the ROI  
        auto detections = hailo_common::get_hailo_detections(roi);  
        std::cout << "DEBUG: Found " << detections.size() << " detections to attach classification to" << std::endl;  
          
        if (!detections.empty()) {  
            // Add classification to the first detection  
            auto classification = std::make_shared<HailoClassification>("license_plate", out_text, conf);  
            detections[0]->add_object(classification);  
            std::cout << "DEBUG: Added classification '" << out_text << "' to detection" << std::endl;  
        } else {  
            std::cout << "DEBUG: No detections found to attach classification to" << std::endl;  
        }  
    } else {  
        std::cout << "DEBUG: Empty or whitespace-only text, not creating classification" << std::endl;  
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
          
        // // Only keep text_region detections, filter out others  
        // if (label == "text_region") {  
        //     std::cout << "DEBUG: Found text_region detection - keeping for OCR" << std::endl;  
        //     text_detections.push_back(detection);  
        // } else {  
        //     std::cout << "DEBUG: Skipping detection with label: '" << label << "'" << std::endl;  
        // }  
        text_detections.push_back(detection); 
    }  
      
    // Remove all detections first  
    roi->remove_objects_typed(HAILO_DETECTION);  
      
    // Add back only text region detections  
    for (auto text_detection : text_detections) {  
        roi->add_object(text_detection);  
    }  
      
    std::cout << "DEBUG: Filtered to " << text_detections.size() << " text region detections" << std::endl;  
}

extern "C"
std::vector<HailoROIPtr> crop_text_regions(std::shared_ptr<HailoMat> image,
                                           HailoROIPtr roi,
                                           bool use_letterbox,
                                           bool no_scaling_bbox,
                                           bool internal_offset,
                                           const std::string &resize_method)
{
    std::cout << "DEBUG: crop_text_regions called as cropper function" << std::endl;

    // 1) Gather detections
    std::vector<HailoROIPtr> crop_rois;
    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);

    // 2) Image size for pixel-based thresholds
    const int img_w = image->width();
    const int img_h = image->height();

    // 3) Tunables (pixels)
    constexpr int MAX_TEXT_REGIONS   = 8;   // match your batch size
    constexpr float MIN_W_PX         = 8.0f;
    constexpr float MIN_H_PX         = 4.0f;     // allow very thin lines
    constexpr float TARGET_MIN_H_PX  = 12.0f;    // inflate to this if too thin
    constexpr float PAD_X_PX         = 4.0f;     // small horizontal padding
    constexpr float PAD_Y_PX         = 2.0f;     // small vertical padding

    auto clamp01 = [](float v){ return std::max(0.0f, std::min(1.0f, v)); };

    int count = 0;
    for (auto &detection : detections) {
        if (count >= MAX_TEXT_REGIONS) break;

        const std::string label = detection->get_label();
        if (label != "text_region") continue;

        auto nb = detection->get_bbox(); // normalized box in [0,1] (assumption)
        float nx = nb.xmin();
        float ny = nb.ymin();
        float nw = nb.width();
        float nh = nb.height();

        std::cout << "DEBUG: Processing detection with label: '" << label << "'\n";
        std::cout << "DEBUG: Text region (normalized): w=" << nw << " h=" << nh
                  << " x=" << nx << " y=" << ny << std::endl;

        // 4) If frames were letterboxed, undo letterbox *before* pixel conversion
        //    (Replace this block with your project’s actual de-letterbox helper if you have one)
        if (use_letterbox) {
            // Example heuristic de-letterbox for 16:9 input letterboxed into a square model, etc.
            // If you have model-input W,H and padding meta in ROI, use that instead of guessing.
            // Here we assume nb is in the letterboxed domain and map it back to the raw image.
            // Remove this if your pipeline already writes non-letterboxed boxes.
            float img_aspect = static_cast<float>(img_w) / img_h;

            // Assume model input is square; letterbox on vertical for wide images:
            // scale so that the *short* side fits, then compute paddings.
            float scale = 1.0f;
            float pad_x = 0.0f;
            float pad_y = 0.0f;

            if (img_aspect >= 1.0f) {
                // wide image: height matched, horizontal pad
                scale = 1.0f / img_aspect;
                pad_x = (1.0f - scale) * 0.5f;
                // pad_y = 0
            } else {
                // tall image: width matched, vertical pad
                scale = img_aspect;
                pad_y = (1.0f - scale) * 0.5f;
                // pad_x = 0
            }

            // Remove padding + rescale back to full image norm
            float x0 = clamp01((nx - pad_x) / scale);
            float y0 = clamp01((ny - pad_y) / scale);
            float x1 = clamp01((nx + nw - pad_x) / scale);
            float y1 = clamp01((ny + nh - pad_y) / scale);

            nx = x0; ny = y0; nw = std::max(0.0f, x1 - x0); nh = std::max(0.0f, y1 - y0);
            std::cout << "DEBUG: De-letterboxed (normalized): w=" << nw << " h=" << nh
                      << " x=" << nx << " y=" << ny << std::endl;
        }

        // 5) Convert to pixels
        float w_px = nw * img_w;
        float h_px = nh * img_h;

        std::cout << "DEBUG: Text region (pixels): w=" << w_px << " h=" << h_px
                  << " x=" << nx * img_w << " y=" << ny * img_h << std::endl;

        // 6) Size check in pixels
        if (w_px < MIN_W_PX || h_px < MIN_H_PX) {
            std::cout << "DEBUG: Skipping text_region (too small in px) "
                         "[min_w=" << MIN_W_PX << ", min_h=" << MIN_H_PX << "]" << std::endl;
            continue;
        }

        // 7) Inflate very thin text lines to a minimum pixel height (helps OCR crops)
        if (h_px < TARGET_MIN_H_PX) {
            float center_y = ny + nh * 0.5f;
            float new_h_n  = TARGET_MIN_H_PX / img_h;
            float new_y    = center_y - new_h_n * 0.5f;

            ny = clamp01(new_y);
            nh = std::min(1.0f - ny, new_h_n); // clamp if we shifted near the bottom

            // Recompute pixel height (for logs)
            h_px = nh * img_h;
            std::cout << "DEBUG: Inflated thin text box to h_px=" << h_px << " (target "
                      << TARGET_MIN_H_PX << " px)" << std::endl;
        }

        // 8) Add a little padding
        float pad_x_n = PAD_X_PX / img_w;
        float pad_y_n = PAD_Y_PX / img_h;

        float x0 = clamp01(nx - pad_x_n);
        float y0 = clamp01(ny - pad_y_n);
        float x1 = clamp01(nx + nw + pad_x_n);
        float y1 = clamp01(ny + nh + pad_y_n);

        // 9) Write back to detection bbox (normalized), or clone to a new ROI if you prefer
        //    If your cropper expects ROIs, you can create child-ROIs from `roi` using these coords.
        detection->set_bbox(HailoBBox(x0, y0, std::max(0.0f, x1 - x0), std::max(0.0f, y1 - y0)));

        std::cout << "DEBUG: Accepted text_region crop [x0=" << x0 << ", y0=" << y0
                  << ", x1=" << x1 << ", y1=" << y1 << "]" << std::endl;

        crop_rois.push_back(detection);
        ++count;
    }

    std::cout << "DEBUG: Returning " << crop_rois.size() << " text regions for cropping" << std::endl;
    return crop_rois;
}
