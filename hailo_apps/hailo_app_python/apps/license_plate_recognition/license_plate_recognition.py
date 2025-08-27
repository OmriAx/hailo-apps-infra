# region imports
# Standard library imports
import setproctitle

# Local application-specific imports
from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.common.core import get_default_parser, get_resource_path
from hailo_apps.hailo_app_python.core.common.defines import LPR_DB_DIR_NAME,LPR_DIR_NAME,LPR_CAPTURED_VEHICLES_DIR_NAME,LPR_VIDEO_NAME,RESOURCES_VIDEOS_DIR_NAME,LPR_OCR_POSTPROCESS_FUNCTION,LPT_TEXT_DET_POSTPROCESS_FUNCTION,LPR_APP_TITLE,LPR_PIPELINE,LPR_POSTPROCESS_SO_FILENAME,LPR_OCR_MODEL_NAME,LPR_TEXT_DET_MODEL_NAME,LPR_VEHICLE_POSTPROCESS_FUNCTION,LPR_VEHICLE_HEF_NAME, DETECTION_APP_TITLE, DETECTION_PIPELINE, RESOURCES_MODELS_DIR_NAME, RESOURCES_SO_DIR_NAME, DETECTION_POSTPROCESS_SO_FILENAME, DETECTION_POSTPROCESS_FUNCTION
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import SOURCE_PIPELINE, INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER, TRACKER_PIPELINE, USER_CALLBACK_PIPELINE, DISPLAY_PIPELINE , CROPPER_PIPELINE
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import GStreamerApp, app_callback_class, dummy_callback
from hailo_apps.hailo_app_python.core.common.db_handler import DatabaseHandler, Record
# endregion imports

# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

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
    app = GstreamerLPRApp(app_callback, ud)
    app.run()