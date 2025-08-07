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

class GstreamerLPRApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser is None:
            parser = get_default_parser()
        super().__init__(parser, user_data)

        self.batch_size = 2
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45

                # Initialize the database and table
        self.db_handler = DatabaseHandler(db_name='vehicles.db', 
                                          table_name='vehicles', 
                                          schema=Record, 
                                          threshold=nms_iou_threshold,
                                          database_dir=get_resource_path(pipeline_name=None, resource_type=LPR_DIR_NAME, model=LPR_DB_DIR_NAME),
                                          samples_dir = get_resource_path(pipeline_name=None, resource_type=LPR_DIR_NAME, model=LPR_CAPTURED_VEHICLES_DIR_NAME))

        if self.options_menu.input is None:  # Setting up a new application-specific default video (overrides the default video set in the GStreamerApp constructor)
            self.video_source = get_resource_path(
                pipeline_name=LPR_PIPELINE,
                resource_type=RESOURCES_VIDEOS_DIR_NAME,
                model=LPR_VIDEO_NAME
            )
        # Determine the architecture if not specified
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch

        self.vehicle_hef_path = get_resource_path(
            LPR_VEHICLE_HEF_NAME, RESOURCES_MODELS_DIR_NAME
        )
        self.vehicle_post_process_function = LPR_VEHICLE_POSTPROCESS_FUNCTION

        self.text_detector_hef_path = get_resource_path(
            LPR_TEXT_DET_MODEL_NAME, RESOURCES_MODELS_DIR_NAME
        )
        self.text_detector_post_process_function = LPT_TEXT_DET_POSTPROCESS_FUNCTION

        self.ocr_hef_path = get_resource_path(
            LPR_OCR_MODEL_NAME, RESOURCES_MODELS_DIR_NAME
        )
        self.ocr_post_process_function = LPR_OCR_POSTPROCESS_FUNCTION

        self.post_process_so = get_resource_path(
            LPR_PIPELINE, RESOURCES_SO_DIR_NAME, LPR_POSTPROCESS_SO_FILENAME
        )

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        setproctitle.setproctitle(LPR_APP_TITLE)

        self.create_pipeline()
    
    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(video_source=self.video_source,
                                          video_width=self.video_width, video_height=self.video_height,
                                          frame_rate=self.frame_rate, sync=self.sync)
        vehicle_detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.vehicle_hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.vehicle_post_process_function,
            batch_size=self.batch_size,
            thresholds_str=self.thresholds_str
        )

        vehicle_tracker_pipeline = TRACKER_PIPELINE(
            class_id=2  # Assuming class_id 2 is for vehicles
        )

        ocr_detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.text_detector_hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.text_detector_post_process_function,
            batch_size=self.batch_size
        )

        ocr_pipeline = INFERENCE_PIPELINE(
            hef_path=self.ocr_hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.ocr_post_process_function,
            batch_size=self.batch_size
        )

        combined_ocr_pipeline = f"{ocr_detection_pipeline} ! {ocr_pipeline}"  

        cropping_pipeline = CROPPER_PIPELINE(
            inner_pipeline=combined_ocr_pipeline,
            so_path=self.post_process_so,
            function_name="conditional_vehicle_crop" ,
        )

        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)

        pipeline_string = (
            f'{source_pipeline} ! '
            f'{vehicle_detection_pipeline} ! '
            f'{vehicle_tracker_pipeline} ! '
            f'{cropping_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )

        print(pipeline_string)
        return pipeline_string