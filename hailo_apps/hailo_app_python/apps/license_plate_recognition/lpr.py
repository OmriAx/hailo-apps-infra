import setproctitle
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
from hailo_apps.hailo_app_python.core.common.core import get_default_parser
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import GStreamerApp , app_callback_class
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    CROPPER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
    TRACKER_PIPELINE,
)
from hailo_apps.hailo_app_python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
import hailo

class GStreamerOCRApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        # use default CLI parser if none is supplied
        if parser is None:
            parser = get_default_parser()
        parser.add_argument(
            "--print-database",
            default=None,
            help="Path to costume labels JSON file",
        )
        # initialise the base class; it parses CLI args and sets up video_source, etc.
        super().__init__(parser, user_data)

        # store the callback so GStreamerApp can hook it up at run time
        self.app_callback = app_callback

        # model and post–processing paths
       # self.vehicle_hef_path = "/home/omri/dev/hailo-apps-infra/yolov5m_vehicles.hef"
        self.vehicle_hef_path = "/home/omri/dev/hailo-apps-infra/resources/models/hailo8/yolov5m_vehicles.hef"
        self.det_hef_path = "/home/omri/dev/hailo-apps-infra/ocr_det.hef"
        self.rec_hef_path = "/home/omri/dev/hailo-apps-infra/ocr.hef"


        self.det_post_process_so = "/home/omri/dev/hailo-apps-infra/resources/so/libocr_postprocess.so"
        self.det_post_function = "paddleocr_det"
        self.rec_post_process_so = "/home/omri/dev/hailo-apps-infra/resources/so/libocr_postprocess.so"
        self.rec_post_function = "paddleocr_recognize"
        self.vehicle_post_process_so = "/home/omri/dev/hailo-apps-infra/resources/so/libyolo_hailortpp_postprocess.so"
        self.vehicle_post_process_function = "yolov5m_vehicles"
        # cropper library and function
        self.cropper_so_path = "/home/omri/dev/hailo-apps-infra/resources/so/libvms_croppers.so"
        self.cropper_function = "crop_vehicles"

        # override the video source and dimensions to match the detector input (960×544)
       # self.video_source = "/home/omri/dev/hailo-apps-infra/istockphoto-1188451252-640_adpp_is.mp4"
        self.video_width = 640
        self.video_height = 640
        self.batch_size = 2

        self.video_source = "/home/omri/dev/hailo-apps-infra/test3.mp4"
        self.thresholds_str = "nms-score-threshold=0.3 nms-iou-threshold=0.45"  
        self.vehicle_labels_json = "/home/hailo/omria/hailo-apps-infra/resources/json/yolov5m_vehicles.json"



        setproctitle.setproctitle("Hailo OCR App")
        self.create_pipeline()

    def get_pipeline_string(self):  
        # Source pipeline  
        source_pipeline = SOURCE_PIPELINE(  
            video_source=self.video_source,  
            video_width=self.video_width,   
            video_height=self.video_height,  
            frame_rate=self.frame_rate,   
            sync=self.sync)  
        
        # 1. Vehicle detection (yolov5m_vehicles)  
        vehicle_detection_pipeline = INFERENCE_PIPELINE(  
            hef_path=self.vehicle_hef_path,  
            post_process_so=self.vehicle_post_process_so,  
            post_function_name=self.vehicle_post_process_function, 
            batch_size=self.batch_size,  
            additional_params=self.thresholds_str,
            )  
        
        vehicle_detection_wrapper = INFERENCE_PIPELINE_WRAPPER(vehicle_detection_pipeline)  
        
        # 2. Vehicle tracker  
        vehicle_tracker_pipeline = TRACKER_PIPELINE(  
            class_id=-1,  # Track all vehicle classes  
            kalman_dist_thr=0.8,  
            iou_thr=0.9,  
            init_iou_thr=0.7,  
            keep_tracked_frames=15,  
            name='vehicle_tracker')  
        
        # 3. Text detection pipeline (runs on vehicle crops)  
        text_detection_pipeline = INFERENCE_PIPELINE(  
            hef_path=self.det_hef_path,  
            post_process_so=self.det_post_process_so,  
            post_function_name=self.det_post_function,  # Your detection function  
            batch_size=self.batch_size,  
            name='text_detection')  
        
        # 4. OCR recognition pipeline (runs on text crops)  
        ocr_recognition_pipeline = INFERENCE_PIPELINE(  
            hef_path=self.rec_hef_path,  
            post_process_so=self.rec_post_process_so,  
            post_function_name=self.rec_post_function,  
            batch_size=self.batch_size,  
            name='ocr_recognition')  
        
        # First cropper: crop vehicles and run text detection  
        vehicle_cropper = CROPPER_PIPELINE(  
            inner_pipeline=text_detection_pipeline,  
            so_path=self.cropper_so_path,  # Use the correct variable  
            function_name=self.cropper_function,
            name='vehicle_cropper')  # Use the correct variable  
        
        # Second cropper: crop text regions and run OCR  
        text_cropper = CROPPER_PIPELINE(  
            inner_pipeline=ocr_recognition_pipeline,  
            so_path=self.det_post_process_so,  # Your OCR post-process SO  
            function_name="crop_text_regions",
            name='text_cropper')  # Your existing function  
        
        user_callback_pipeline = USER_CALLBACK_PIPELINE()  
        display_pipeline = DISPLAY_PIPELINE(  
            video_sink=self.video_sink,   
            sync=self.sync,   
            show_fps=self.show_fps)  
        
        pipeline_string = (  
            f'{source_pipeline} ! '  
            f'{vehicle_detection_wrapper} ! '  
            f'{vehicle_tracker_pipeline} ! '  
            f'{vehicle_cropper} ! '  
            f'{text_cropper} ! '  
            f'{user_callback_pipeline} ! '  
            f'{display_pipeline}'  
        )  
        
        print("Generated pipeline string:")  
        print(pipeline_string)  # Add this debug line  
        return pipeline_string


class user_app_callback_class(app_callback_class):  
    def __init__(self):  
        super().__init__()  

def app_callback(pad, info, user_data):  
    print("CALLBACK CALLED!")  
    buffer = info.get_buffer()  
    if buffer is None:  
        return Gst.PadProbeReturn.OK  
      
    user_data.increment()  
    roi = hailo.get_roi_from_buffer(buffer)  
    if roi is None:  
        print(f"Frame {user_data.get_count()}: No ROI data")  
        return Gst.PadProbeReturn.OK  
      
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)  
    print(f"Frame {user_data.get_count()}: Found {len(detections)} detections")  
      
    # Debug each detection in detail  
    for i, detection in enumerate(detections):  
        label = detection.get_label()  
        confidence = detection.get_confidence()  
        print(f"  Detection {i}: Label='{label}', Confidence={confidence:.2f}")  
          
        # Check tracking  
        track_ids = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)  
        track_id = track_ids[0].get_id() if track_ids else "No Track ID"  
        print(f"    Track ID: {track_id}")  
          
        # Check classifications  
        classifications = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)  
        print(f"    Classifications: {len(classifications)}")  
          
        for j, classification in enumerate(classifications):  
            text = classification.get_label()  
            conf = classification.get_confidence()  
            print(f"      Classification {j}: '{text}' (confidence: {conf:.2f})")  
      
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    ud = user_app_callback_class()
    app = GStreamerOCRApp(app_callback, ud)
    app.run()
