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
import time

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

        self.video_source = "/home/omri/dev/hailo-apps-infra/test1.mp4"
        self.thresholds_str = "nms-score-threshold=0.3 nms-iou-threshold=0.45"  
        self.vehicle_labels_json = "/home/hailo/omria/hailo-apps-infra/resources/json/yolov5m_vehicles.json"



        setproctitle.setproctitle("Hailo OCR App")
        self.create_pipeline()

    def get_pipeline_string(self):  
        source_pipeline = SOURCE_PIPELINE(  
            video_source=self.video_source,  
            video_width=self.video_width,  
            video_height=self.video_height,  
            frame_rate=self.frame_rate,  
            sync=self.sync  
        )  
        
        vehicle_detection_pipeline = INFERENCE_PIPELINE(  
            hef_path=self.vehicle_hef_path,  
            post_process_so=self.vehicle_post_process_so,  
            post_function_name=self.vehicle_post_process_function,  
            batch_size=8,  
            additional_params=self.thresholds_str  
        )  
        
        vehicle_detection_wrapper = INFERENCE_PIPELINE_WRAPPER(vehicle_detection_pipeline)  
        
        vehicle_tracker_pipeline = TRACKER_PIPELINE(  
            class_id=-1,  
            kalman_dist_thr=0.8,  
            iou_thr=0.9,  
            init_iou_thr=0.7,  
            keep_tracked_frames=3,
            keep_lost_frames=1,  
            name='vehicle_tracker'  
        )  
        
        # OCR processing branch (runs in background)  
        det_pipe = INFERENCE_PIPELINE(  
            hef_path=self.det_hef_path,  
            post_process_so=self.det_post_process_so,  
            post_function_name=self.det_post_function,  
            batch_size=4,  
            name="text_detection"  
        )  
        
        rec_pipe = INFERENCE_PIPELINE(  
            hef_path=self.rec_hef_path,  
            post_process_so=self.rec_post_process_so,  
            post_function_name=self.rec_post_function,  
            batch_size=4,  
            name="ocr_recognition"  
        )  
        
        cropper = CROPPER_PIPELINE(  
            inner_pipeline=rec_pipe,  
            so_path=self.det_post_process_so,  
            function_name="crop_text_regions",  
            name="text_cropper"  
        )  
        
        combined_ocr_pipeline = det_pipe + " ! " + cropper  
        
        optimized_cropper = CROPPER_PIPELINE(  
            inner_pipeline=combined_ocr_pipeline,  
            so_path=self.cropper_so_path,  
            function_name="crop_top_8_vehicles",  
            name='vehicle_cropper'  
        )  
        
        # Split pipeline: one branch for OCR, one for display  
        pipeline_string = (  
            f'{source_pipeline} ! '  
            f'{vehicle_detection_wrapper} ! '  
            f'{vehicle_tracker_pipeline} ! '  
            f'tee name=detection_tee '  
            f'detection_tee. ! queue ! {optimized_cropper} ! fakesink '  # OCR branch  
            f'detection_tee. ! queue ! {USER_CALLBACK_PIPELINE()} ! {DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)}'  # Display branch  
        )  
        print(pipeline_string)
        return pipeline_string

class LicensePlateTracker:  
    def __init__(self):  
        self.plates = {}  # track_id -> license_plate_text  
          
    def update_plate(self, track_id, plate_text, confidence):  
        if track_id not in self.plates or confidence > self.plates[track_id]['confidence']:  
            self.plates[track_id] = {  
                'text': plate_text,  
                'confidence': confidence,  
                'last_seen': time.time()  
            }  
      
    def get_plate(self, track_id):  
        return self.plates.get(track_id, {}).get('text', '')  
  
# Add to your user_app_callback_class  
class user_app_callback_class(app_callback_class):  
    def __init__(self):  
        super().__init__()  
        self.license_tracker = LicensePlateTracker()
  

def app_callback(pad, info, user_data):  
    buffer = info.get_buffer()  
    if buffer is None:  
        return Gst.PadProbeReturn.OK  
      
    user_data.increment()  
    roi = hailo.get_roi_from_buffer(buffer)  
    if roi is None:  
        return Gst.PadProbeReturn.OK  
      
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)  
      
    for detection in detections:  
        label = detection.get_label()  
        confidence = detection.get_confidence()  
          
        if label == "car":  
            # Get track ID  
            track_ids = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)  
            track_id = track_ids[0].get_id() if track_ids else None  
              
            if track_id:  
                # Check for license plate classifications  
                classifications = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)  
                for classification in classifications:  
                    class_label = classification.get_label()  
                    if class_label.startswith("track_"):  
                        # Parse track ID and license plate text  
                        parts = class_label.split(":", 1)  
                        if len(parts) == 2:  
                            extracted_track_id = int(parts[0].replace("track_", ""))  
                            plate_text = parts[1]  
                              
                            # Store in tracker  
                            user_data.license_tracker.update_plate(  
                                extracted_track_id,   
                                plate_text,   
                                classification.get_confidence()  
                            )  
                  
                # Get stored license plate for this vehicle  
                plate_text = user_data.license_tracker.get_plate(track_id)  
                  
                print(f"Vehicle ID: {track_id}, Confidence: {confidence:.2f}, License: {plate_text}")  
      
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    ud = user_app_callback_class()
    app = GStreamerOCRApp(app_callback, ud)
    app.run()
