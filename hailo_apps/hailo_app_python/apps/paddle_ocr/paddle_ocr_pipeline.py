import setproctitle
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
from hailo_apps.hailo_app_python.core.common.core import get_default_parser, get_resource_path
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import GStreamerApp , app_callback_class
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
from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger
from hailo_apps.hailo_app_python.core.common.defines import (
    OCR_PIPELINE,
    RESOURCES_MODELS_DIR_NAME,
    RESOURCES_SO_DIR_NAME,
    RESOURCES_VIDEOS_DIR_NAME,
    OCR_DET_MODEL_NAME,
    OCR_REC_MODEL_NAME,
    OCR_POSTPROCESS_SO_FILENAME,
    OCR_DET_POSTPROCESS_FUNCTION,
    OCR_REC_POSTPROCESS_FUNCTION,
    OCR_CROPPER_POSTPROCESS_FUNCTION,
    OCR_VIDEO_NAME,
)

import hailo
import cv2

hailo_logger = get_logger(__name__)

class GStreamerOCRApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        # use default CLI parser if none is supplied
        if parser is None:
            parser = get_default_parser()
        # initialise the base class; it parses CLI args and sets up video_source, etc.
        super().__init__(parser, user_data)

        self.batch_size = 2
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45

        # Determine the architecture if not specified
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            hailo_logger.debug("Auto-detected Hailo arch: %s", detected_arch)
            if detected_arch is None:
                hailo_logger.error("Could not auto-detect Hailo architecture.")
                raise ValueError(
                    "Could not auto-detect Hailo architecture. Please specify --arch manually."
                )
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch
            hailo_logger.debug("Using user-specified arch: %s", self.arch)

        # model and post–processing paths
        self.det_hef_path = get_resource_path(OCR_PIPELINE,RESOURCES_MODELS_DIR_NAME,OCR_DET_MODEL_NAME)
        self.rec_hef_path = get_resource_path(OCR_PIPELINE,RESOURCES_MODELS_DIR_NAME,OCR_REC_MODEL_NAME)

        self.post_process_so = get_resource_path(OCR_PIPELINE,RESOURCES_SO_DIR_NAME,OCR_POSTPROCESS_SO_FILENAME)

        self.det_post_function = OCR_DET_POSTPROCESS_FUNCTION
        self.rec_post_function = OCR_REC_POSTPROCESS_FUNCTION

        # cropper library and function
        self.cropper_function = OCR_CROPPER_POSTPROCESS_FUNCTION

        # override the video source and dimensions to match the detector input (960×544)
        self.video_source = get_resource_path(OCR_PIPELINE,RESOURCES_VIDEOS_DIR_NAME,OCR_VIDEO_NAME)
        self.video_width = 960
        self.video_height = 544

        self.app_callback = app_callback  

        setproctitle.setproctitle("Hailo OCR App")
        self.create_pipeline()

    def get_pipeline_string(self):
        # source stage
        source = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
        )
        # text detection stage; wrap it to preserve original frame size
        det = INFERENCE_PIPELINE(
            hef_path=self.det_hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.det_post_function,
        )
        det_wrapper = INFERENCE_PIPELINE_WRAPPER(det)

        # text recognition stage
        rec = INFERENCE_PIPELINE(
            hef_path=self.rec_hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.rec_post_function,
            name='recognition'   
        )

        # cropper stage: crops detections and feeds them into the recognition network
        cropper = CROPPER_PIPELINE(
            inner_pipeline=rec,
            so_path=self.post_process_so,
            function_name=self.cropper_function,
        )

        # user callback and display
        callback = USER_CALLBACK_PIPELINE()
        display = DISPLAY_PIPELINE(
            video_sink=self.video_sink,
            sync=self.sync,
            show_fps=self.show_fps,
        )

        return f"{source} ! {det_wrapper} ! {cropper} ! {callback} ! {display}"


class user_app_callback_class(app_callback_class):  
    """Simple container for frame counting and OCR results."""  
    def __init__(self):  
        super().__init__()  # CRITICAL FIX: Call parent constructor  
        self.results = []   # store recognised strings here

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()

    roi = hailo.get_roi_from_buffer(buffer)
    if not roi:
        return Gst.PadProbeReturn.OK

    # Extract OpenCV image from buffer
    frame = get_numpy_from_buffer(buffer, "RGB", 960, 544)

    if frame is None:
        return Gst.PadProbeReturn.OK

    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    for det in detections:
        bbox = det.get_bbox()
        x = int(bbox.xmin() * frame.shape[1])
        y = int(bbox.ymin() * frame.shape[0])
        w = int(bbox.width()  * frame.shape[1])
        h = int(bbox.height() * frame.shape[0])

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract classification result (if any)
        classifications = det.get_objects_typed(hailo.HAILO_CLASSIFICATION)
        for cls in classifications:
            try:
                text = cls.get_label()
                conf = cls.get_score()
                label_text = f"{text} ({conf:.2f})"

                # Overlay OCR result
                cv2.putText(frame, label_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Print to console
                print(f"Recognised text: {text} (confidence={conf:.2f})")
                user_data.results.append(text)

                # Save to file
                with open("ocr_results.txt", "a") as f:
                    f.write(text + "\n")

            except Exception as e:
                print(f"Classification label extraction error: {e}")
                continue

    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    ud = user_app_callback_class()
    app = GStreamerOCRApp(app_callback, ud)
    app.run()