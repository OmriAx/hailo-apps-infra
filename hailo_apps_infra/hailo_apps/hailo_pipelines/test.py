
import numpy as np
import setproctitle
from pathlib import Path
import sys
import cairo

# Ensure hailo_core is importable from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # ~/dev/hailo-apps-infra/hailo_apps_infra
sys.path.insert(0, str(PROJECT_ROOT))


# ─── Common Hailo helpers ────────────────────────────────────────────────────────
try:
    from hailo_core.hailo_common.installation_utils import detect_hailo_arch
except ImportError:
    from hailo_apps_infra.hailo_core.hailo_common.installation_utils import detect_hailo_arch
try:
    from hailo_core.hailo_common.core import (
    get_default_parser,
    get_resource_path,
)
except ImportError:
    from hailo_apps_infra.hailo_core.hailo_common.core import (
    get_default_parser,
    get_resource_path,
)
try:
    from hailo_core.hailo_common.defines import (
    DETECTION_APP_TITLE,
    DETECTION_PIPELINE,
    RESOURCES_MODELS_DIR_NAME,
    RESOURCES_SO_DIR_NAME,
    DETECTION_POSTPROCESS_SO_FILENAME,
    DETECTION_POSTPROCESS_FUNCTION,
)
except ImportError:
    from hailo_apps_infra.hailo_core.hailo_common.defines import (
    DETECTION_APP_TITLE,
    DETECTION_PIPELINE,
    RESOURCES_MODELS_DIR_NAME,
    RESOURCES_SO_DIR_NAME,
    DETECTION_POSTPROCESS_SO_FILENAME,
    DETECTION_POSTPROCESS_FUNCTION,
    )

# ─── GStreamer routines (from your hailo_gstreamer package) ────────────────────
try:
    from hailo_apps.hailo_gstreamer.gstreamer_helper_pipelines import (
        QUEUE,
        SOURCE_PIPELINE,
        INFERENCE_PIPELINE,
        INFERENCE_PIPELINE_WRAPPER,
        TRACKER_PIPELINE,
        USER_CALLBACK_PIPELINE,
        DISPLAY_PIPELINE_DRAW,
    )
except ImportError:
    from hailo_apps_infra.hailo_apps.hailo_gstreamer.gstreamer_helper_pipelines import (
        QUEUE,
        SOURCE_PIPELINE,
        INFERENCE_PIPELINE,
        INFERENCE_PIPELINE_WRAPPER,
        TRACKER_PIPELINE,
        USER_CALLBACK_PIPELINE,
        DISPLAY_PIPELINE_DRAW,
    )
try:
    from hailo_apps.hailo_gstreamer.gstreamer_app import (
        GStreamerApp,
        app_callback_class,
        dummy_callback,
    )
except ImportError:
    from hailo_apps_infra.hailo_apps.hailo_gstreamer.gstreamer_app import (
        GStreamerApp,
        app_callback_class,
        dummy_callback,
    )

# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None):
        if parser == None:
            parser = get_default_parser()

        # Call the parent class constructor
        super().__init__(parser, user_data)
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.batch_size = 2
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45


        # Determine the architecture if not specified
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = self.options_menu.arch


        if self.options_menu.hef_path is not None:
            self.hef_path = self.options_menu.hef_path
        else:
            self.hef_path = get_resource_path(DETECTION_PIPELINE, RESOURCES_MODELS_DIR_NAME)

        # Set the post-processing shared object file
        self.post_process_so = get_resource_path(
            DETECTION_PIPELINE, RESOURCES_SO_DIR_NAME, DETECTION_POSTPROCESS_SO_FILENAME
        )
        self.post_function_name = DETECTION_POSTPROCESS_FUNCTION
        # User-defined label JSON file
        self.labels_json = self.options_menu.labels_json

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Set the process title
        setproctitle.setproctitle(DETECTION_APP_TITLE)

        self.create_pipeline()

    def get_pipeline_string(self):
        src   = SOURCE_PIPELINE(
                    video_source=self.video_source,
                    video_width=self.video_width,
                    video_height=self.video_height,
                    frame_rate=self.frame_rate,
                    sync=self.sync)
        infer = INFERENCE_PIPELINE_WRAPPER(
                    INFERENCE_PIPELINE(
                        hef_path=self.hef_path,
                        post_process_so=self.post_process_so,
                        post_function_name=self.post_function_name,
                        batch_size=self.batch_size,
                        config_json=self.labels_json,
                        additional_params=self.thresholds_str))
        track = TRACKER_PIPELINE(class_id=1)
        usercb= USER_CALLBACK_PIPELINE()

        #  ────────────── inline display + custom draw ──────────────
        display = (
            f"{src} ! {infer} ! {track} ! {usercb} ! "
            # 1) original overlay
            "queue name=hailo_display_overlay_q ! "
            "hailooverlay name=hailo_display_overlay ! "
            # 2) convert caps into raw for cairooverlay
            "queue name=hailo_display_videoconvert_q ! "
            "videoconvert name=hailo_display_videoconvert ! "
            # 3) _your_ draw hook
            "queue name=hailo_display_draw_q ! "
            "cairooverlay name=hailo_display_draw ! "
            # 4) convert back to standard video
            "videoconvert ! "
            # 5) back into the original queue + sink
            "queue name=hailo_display_q ! "
            f"fpsdisplaysink name=hailo_display "
            f"video-sink={self.video_sink} sync={self.sync} "
            f"text-overlay={self.show_fps} signal-fps-measurements=true"
        )
        return display

    def on_custom_draw(self, overlay, context, timestamp, duration, user_data):
        """
        overlay:    the cairooverlay element
        context:    a cairo.Context to draw into
        timestamp:  GstClockTime of this frame
        duration:   GstClockTime duration of this frame
        user_data:  the user_data you passed (e.g. your callback class)
        """
        # pick font + size
        context.select_font_face("Sans",
                                cairo.FONT_SLANT_NORMAL,
                                cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(24.0)
        # white, semi‐opaque
        context.set_source_rgba(1.0, 1.0, 1.0, 0.8)
        context.move_to(20, 40)
        context.show_text("Hailo Omri Simple Draw")
        context.stroke()



def main():
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
    
if __name__ == "__main__":
    print("Starting Hailo Detection App...")
    main()
 