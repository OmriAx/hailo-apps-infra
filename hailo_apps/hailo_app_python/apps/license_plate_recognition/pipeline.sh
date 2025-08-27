# /bin/bash

gst-launch-1.0 \
  filesrc location="/home/omri/dev/hailo-apps-infra/test3.mp4" name=source ! \
  queue name=source_queue_decode max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  decodebin name=source_decodebin \
  \
  source_decodebin. ! queue name=source_scale_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  videoscale name=source_videoscale n-threads=2 ! \
  queue name=source_convert_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  videoconvert n-threads=3 name=source_convert qos=false ! \
  video/x-raw, pixel-aspect-ratio=1/1, width=640, height=640 ! \
  videorate name=source_videorate ! "video/x-raw, framerate=1/1" ! \
  queue name=inference_wrapper_input_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  hailocropper name=inference_wrapper_crop \
    so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so \
    function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true \
  hailoaggregator name=inference_wrapper_agg \
  \
  # BYPASS branch
  inference_wrapper_crop. ! queue name=inference_wrapper_bypass_q max-size-buffers=4 max-size-bytes=0 max-size-time=0 ! inference_wrapper_agg.sink_0 \
  \
  # DETECTION branch
  inference_wrapper_crop. ! queue name=inference_scale_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  videoscale name=inference_videoscale n-threads=2 qos=false ! \
  queue name=inference_convert_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  videoconvert name=inference_videoconvert n-threads=2 ! \
  video/x-raw, pixel-aspect-ratio=1/1, format=RGB ! \
  queue name=inference_hailonet_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  hailonet name=inference_hailonet \
    hef-path=/home/omri/dev/hailo-apps-infra/resources/models/hailo8/yolov5m_vehicles.hef \
    batch-size=1 vdevice-group-id=1 nms-score-threshold=0.3 nms-iou-threshold=0.45 ! \
  queue name=inference_hailofilter_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  hailofilter name=inference_hailofilter \
    so-path=/home/omri/dev/hailo-apps-infra/resources/so/libyolo_hailortpp_postprocess.so function-name=yolov5m_vehicles qos=false ! \
  queue name=inference_output_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  inference_wrapper_agg.sink_1 \
  \
  # TRACK
  inference_wrapper_agg. ! queue name=inference_wrapper_output_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  hailotracker name=vehicle_tracker class-id=-1 kalman-dist-thr=0.8 iou-thr=0.9 init-iou-thr=0.7 \
    keep-new-frames=2 keep-tracked-frames=15 keep-lost-frames=2 keep-past-metadata=false qos=false ! \
  queue name=vehicle_tracker_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  \
  # VEHICLE CROPPER + OCR DET
  queue name=vehicle_cropper_input_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  hailocropper name=vehicle_cropper_cropper \
    so-path=/home/omri/dev/hailo-apps-infra/resources/so/libvms_croppers.so function-name=crop_vehicles \
    use-letterbox=true no-scaling-bbox=true internal-offset=true resize-method=bilinear \
  hailoaggregator name=vehicle_cropper_agg \
  \
  vehicle_cropper_cropper. ! queue name=vehicle_cropper_bypass_q max-size-buffers=4 max-size-bytes=0 max-size-time=0 ! vehicle_cropper_agg.sink_0 \
  \
  vehicle_cropper_cropper. ! queue name=text_detection_scale_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  videoscale name=text_detection_videoscale n-threads=2 qos=false ! \
  queue name=text_detection_convert_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  videoconvert name=text_detection_videoconvert n-threads=2 ! \
  video/x-raw, pixel-aspect-ratio=1/1, format=RGB ! \
  queue name=text_detection_hailonet_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  hailonet name=text_detection_hailonet \
    hef-path=/home/omri/dev/hailo-apps-infra/ocr_det.hef batch-size=1 vdevice-group-id=1 ! \
  queue name=text_detection_hailofilter_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  hailofilter name=text_detection_hailofilter \
    so-path=/home/omri/dev/hailo-apps-infra/resources/so/libtest_postprocess.so function-name=paddleocr_det qos=false ! \
  queue name=text_detection_output_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  hailofilter so-path=/home/omri/dev/hailo-apps-infra/resources/so/libtest_postprocess.so function-name=crop_text_regions_filter name=text_crop_filter ! \
  \
  # OCR REC
  queue name=ocr_recognition_scale_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  videoscale name=ocr_recognition_videoscale n-threads=2 qos=false ! \
  queue name=ocr_recognition_convert_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  videoconvert name=ocr_recognition_videoconvert n-threads=2 ! \
  video/x-raw, pixel-aspect-ratio=1/1, format=RGB ! \
  queue name=ocr_recognition_hailonet_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  hailonet name=ocr_recognition_hailonet \
    hef-path=/home/omri/dev/hailo-apps-infra/ocr.hef batch-size=1 vdevice-group-id=1 ! \
  queue name=ocr_recognition_hailofilter_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  hailofilter name=ocr_recognition_hailofilter \
    so-path=/home/omri/dev/hailo-apps-infra/resources/so/libtest_postprocess.so function-name=paddleocr_recognize qos=false ! \
  queue name=ocr_recognition_output_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  vehicle_cropper_agg.sink_1 \
  \
  # DISPLAY (leaky to avoid back-pressure)
  vehicle_cropper_agg. ! queue name=vehicle_cropper_output_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  queue name=identity_callback_q max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! identity name=identity_callback ! \
  queue name=hailo_display_overlay_q leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  hailooverlay name=hailo_display_overlay ! \
  queue name=hailo_display_videoconvert_q leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  videoconvert name=hailo_display_videoconvert n-threads=2 qos=false ! \
  queue name=hailo_display_q leaky=downstream max-size-buffers=2 max-size-bytes=0 max-size-time=0 ! \
  fpsdisplaysink name=hailo_display video-sink=autovideosink sync=true text-overlay=false signal-fps-measurements=true
