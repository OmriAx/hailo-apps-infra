import setproctitle
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
from hailo_apps.hailo_app_python.core.common.core import get_default_parser
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import GStreamerApp
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    CROPPER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
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
        # initialise the base class; it parses CLI args and sets up video_source, etc.
        super().__init__(parser, user_data)

        # store the callback so GStreamerApp can hook it up at run time
        self.app_callback = app_callback

        # model and post–processing paths
        self.det_hef_path = "/home/omri/dev/hailo-apps-infra/ocr_det.hef"
        self.rec_hef_path = "/home/omri/dev/hailo-apps-infra/ocr.hef"
        self.det_post_process_so = "/home/omri/dev/hailo-apps-infra/resources/so/libtest_postprocess.so"
        self.det_post_function = "filter_db"
        self.rec_post_process_so = "/home/omri/dev/hailo-apps-infra/resources/so/libtest_postprocess.so"
        self.rec_post_function = "filter_ocr"

        # cropper library and function
        self.cropper_so_path = "/home/omri/dev/hailo-apps-infra/resources/so/libtest_postprocess.so"
        self.cropper_function = "crop_text_regions"

        # override the video source and dimensions to match the detector input (960×544)
        self.video_source = "/home/omri/dev/hailo-apps-infra/test3.mp4"
        self.video_width = 960
        self.video_height = 544

        setproctitle.setproctitle("Hailo OCR App")
        self.create_pipeline()

    def get_pipeline_string(self):  
        # Source pipeline  
        source = SOURCE_PIPELINE(  
            video_source=self.video_source,  
            video_width=self.video_width,  
            video_height=self.video_height,  
            frame_rate=self.frame_rate,  
            sync=self.sync,  
        )  
        
        # Text detection pipeline  
        detection_pipeline = INFERENCE_PIPELINE(  
            hef_path=self.det_hef_path,  
            post_process_so=self.det_post_process_so,  
            post_function_name="filter_db",  # Use your actual function name  
        )  
        detection_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)  
        
        # Text recognition pipeline  
        recognition_pipeline = INFERENCE_PIPELINE(  
            hef_path=self.rec_hef_path,  
            post_process_so=self.rec_post_process_so,  
            post_function_name="filter_ocr",  # Use your actual function name  
            name='recognition'  
        )  
        
        # Cropper pipeline to crop detected text regions and feed to recognition  
        cropper = CROPPER_PIPELINE(  
            inner_pipeline=recognition_pipeline,  
            so_path=self.cropper_so_path,  
            function_name="crop_text_regions",  # You need to implement this  
        )  
        
        # User callback and display  
        callback = USER_CALLBACK_PIPELINE()  
        display = DISPLAY_PIPELINE(  
            video_sink=self.video_sink,  
            sync=self.sync,  
            show_fps=self.show_fps,  
        )  
        
        return f"{source} ! {detection_wrapper} ! {cropper} ! {callback} ! {display}"


class user_app_callback_class:
    """Simple container for frame counting and OCR results."""
    def __init__(self):
        self.frame_count = 0
        self.use_frame = False
        self.results = []          # store recognised strings here

    def increment(self):
        self.frame_count += 1

    def get_count(self):
        return self.frame_count

    def set_frame(self, frame):
        pass  # implement if you need to display frames

def app_callback(pad, info, user_data):  
    print("In app callback")
    buffer = info.get_buffer()  
    if buffer is None:  
        return Gst.PadProbeReturn.OK  
  
    user_data.increment()  
      
    # Get the ROI from buffer  
    roi = hailo.get_roi_from_buffer(buffer)  
      
    # Get detections (text regions)  
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)  
      
    for detection in detections:  
        # Each detection may have classification results (OCR text)  
        classifications = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)  
        for classification in classifications:  
            text = classification.get_label()  # The recognized text  
            confidence = classification.get_confidence()  
            print(f"Recognized text: {text} (confidence: {confidence:.2f})")  
              
            # Write to file as in your original code  
            with open("ocr_results.txt", "a") as f:  
                f.write(text + "\n")  

    print(f"Processed frame count: {user_data.get_count()}") 
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    ud = user_app_callback_class()
    app = GStreamerOCRApp(app_callback, ud)
    app.run()