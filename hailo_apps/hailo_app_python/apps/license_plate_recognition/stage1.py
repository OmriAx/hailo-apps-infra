# region imports  
# Standard library imports  
import cv2  
import setproctitle  
import gi    
import os  
import time  
from PIL import Image  
gi.require_version('Gst', '1.0')    
from gi.repository import Gst  
  
# Local application-specific imports  
from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch  
from hailo_apps.hailo_app_python.core.common.core import get_default_parser, get_resource_path  
from hailo_apps.hailo_app_python.core.common.defines import DETECTION_APP_TITLE, DETECTION_PIPELINE, RESOURCES_MODELS_DIR_NAME, RESOURCES_SO_DIR_NAME, DETECTION_POSTPROCESS_SO_FILENAME, DETECTION_POSTPROCESS_FUNCTION  
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import SOURCE_PIPELINE, INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER, TRACKER_PIPELINE, USER_CALLBACK_PIPELINE, DISPLAY_PIPELINE  
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import GStreamerApp, app_callback_class, dummy_callback  
  
import hailo  
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer  
# endregion imports  
  
# -----------------------------------------------------------------------------------------------  
# License Plate Tracker for OCR Results  
# -----------------------------------------------------------------------------------------------  
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
  
# -----------------------------------------------------------------------------------------------  
# User Gstreamer Application  
# -----------------------------------------------------------------------------------------------  
  
# This class inherits from the hailo_rpi_common.GStreamerApp class  
class GStreamerDetectionApp(GStreamerApp):  
    def __init__(self, app_callback, user_data, parser=None):  
        if parser == None:  
            parser = get_default_parser()  
        parser.add_argument(  
            "--labels-json",  
            default=None,  
            help="Path to costume labels JSON file",  
        )  
  
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
  
        if self.options_menu.input is None:  # Setting up a new application-specific default video (overrides the default video set in the GStreamerApp constructor)  
            self.video_source = "/home/omri/dev/hailo-apps-infra/test3.mp4"  
        if self.options_menu.hef_path is not None:  
            self.hef_path = self.options_menu.hef_path  
        else:  
            self.hef_path = "/usr/local/hailo/resources/models/hailo8/yolov5m_vehicles.hef"  
  
        # Set the post-processing shared object file  
        self.post_process_so = "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so"  
  
        # Use the new post-processing function with image saving  
        self.post_function_name = "yolov5m_vehicles_with_image_saving"  
        # User-defined label JSON file  
        self.labels_json = "/home/hailo/omria/hailo-apps-infra/resources/json/yolov5m_vehicles.json"  
  
        self.app_callback = app_callback  
          
        self.thresholds_str = "output-format-type=HAILO_FORMAT_TYPE_FLOAT32"  
  
        # Set the process title  
        setproctitle.setproctitle(DETECTION_APP_TITLE)  
  
        self.create_pipeline()  
  
    def get_pipeline_string(self):  
        source_pipeline = SOURCE_PIPELINE(video_source=self.video_source,  
                                          video_width=self.video_width, video_height=self.video_height,  
                                          frame_rate=self.frame_rate, sync=self.sync)  
        detection_pipeline = INFERENCE_PIPELINE(  
            hef_path=self.hef_path,  
            post_process_so=self.post_process_so,  
            post_function_name=self.post_function_name,  
            batch_size=self.batch_size,  
            config_json=self.labels_json,  
            additional_params=self.thresholds_str)  
        detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)  
          
        # Optimized tracker settings for responsive box removal  
        tracker_pipeline = TRACKER_PIPELINE(  
            class_id=-1,  
            kalman_dist_thr=0.8,  
            iou_thr=0.9,  
            init_iou_thr=0.7,  
            keep_tracked_frames=3,  # Reduced from default 15 to 3  
            keep_lost_frames=1,     # Quick removal of lost tracks  
            name='vehicle_tracker'  
        )  
          
        user_callback_pipeline = USER_CALLBACK_PIPELINE()  
        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)  
  
        pipeline_string = (  
            f'{source_pipeline} ! '  
            f'{detection_pipeline_wrapper} ! '  
            f'{tracker_pipeline} ! '  
            f'{user_callback_pipeline} ! '  
            f'{display_pipeline}'  
        )  
        print(pipeline_string)  
        return pipeline_string  
  
# -----------------------------------------------------------------------------------------------    
# User-defined class to be used in the callback function    
# -----------------------------------------------------------------------------------------------    
# Inheritance from the app_callback_class    
class user_app_callback_class(app_callback_class):    
    def __init__(self):    
        super().__init__()    
        self.total_detections = 0  # Track total detections    
        self.detection_history = []  # Store recent detection counts  
        self.license_tracker = LicensePlateTracker()  # For future OCR integration  
        self.saved_images_count = 0  # Track saved images  
    
    def add_detection_count(self, count):    
        self.total_detections += count    
        self.detection_history.append(count)    
        # Keep only last 10 frames    
        if len(self.detection_history) > 10:    
            self.detection_history.pop(0)  
  
# -----------------------------------------------------------------------------------------------  
# Helper Functions for Image Saving  
# -----------------------------------------------------------------------------------------------  
def crop_vehicle_frame(frame, bbox, width, height):  
    """Crop vehicle from frame using bounding box coordinates"""  
    # Convert normalized coordinates to pixel coordinates  
    x_min = int(bbox.xmin() * width)  
    y_min = int(bbox.ymin() * height)  
    x_max = int((bbox.xmin() + bbox.width()) * width)  
    y_max = int((bbox.ymin() + bbox.height()) * height)  
      
    # Add some padding around the vehicle  
    padding = 20  
    x_min = max(0, x_min - padding)  
    y_min = max(0, y_min - padding)  
    x_max = min(width, x_max + padding)  
    y_max = min(height, y_max + padding)  
      
    # Crop the frame  
    return frame[y_min:y_max, x_min:x_max]  
  
# -----------------------------------------------------------------------------------------------    
# User-defined callback function with enhanced debugging and image saving  
# -----------------------------------------------------------------------------------------------    
    
# This is the callback function that will be called when data is available from the pipeline    
def app_callback(pad, info, user_data):    
    # Get the GstBuffer from the probe info    
    buffer = info.get_buffer()    
    # Check if the buffer is valid    
    if buffer is None:    
        return Gst.PadProbeReturn.OK    
    
    # Using the user_data to count the number of frames    
    user_data.increment()    
    string_to_print = f"=== Frame {user_data.get_count()} Debug Info ===\n"    
    
    # Get the caps from the pad    
    format, width, height = get_caps_from_pad(pad)    
    string_to_print += f"Frame format: {format}, Size: {width}x{height}\n"    
    
    # If the user_data.use_frame is set to True, we can get the video frame from the buffer    
    frame = None    
    if user_data.use_frame and format is not None and width is not None and height is not None:    
        # Get video frame    
        frame = get_numpy_from_buffer(buffer, format, width, height)    
    
    # Get the detections from the buffer    
    roi = hailo.get_roi_from_buffer(buffer)    
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)    
        
    string_to_print += f"Total raw detections found: {len(detections)}\n"    
    
    # Parse the detections with enhanced debugging    
    detection_count = 0    
    car_count = 0    
    other_count = 0  
    saved_this_frame = 0  
        
    for i, detection in enumerate(detections):    
        label = detection.get_label()    
        bbox = detection.get_bbox()    
        confidence = detection.get_confidence()    
        class_id = detection.get_class_id()    
            
        # Debug info for each detection    
        string_to_print += f"  Detection {i+1}: Label='{label}', Class_ID={class_id}, Confidence={confidence:.3f}\n"    
        string_to_print += f"    BBox: x={bbox.xmin():.2f}, y={bbox.ymin():.2f}, w={bbox.width():.2f}, h={bbox.height():.2f}\n"    
            
        if label == "car":    
            car_count += 1    
            # Get track ID    
            track_id = 0    
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)    
            if len(track) == 1:    
                track_id = track[0].get_id()    
                string_to_print += f"    Track ID: {track_id}\n"  
              
            # Check if this vehicle should be saved for OCR debugging  
            classifications = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)  
            should_save = False  
              
            for classification in classifications:
                string_to_print += f"    Found classification: '{classification.get_label()}'\n"    
                if classification.get_label() == "candidate":  
                    should_save = True  
                    string_to_print += f"    Marked for OCR debugging\n"  
                    break  
              
            if should_save and frame is not None:  
                # Crop and save the vehicle image  
                cropped_frame = crop_vehicle_frame(frame, bbox, width, height)  
                  
                # Save with timestamp and confidence in current directory  
                timestamp = int(time.time() * 1000)  # milliseconds  
                filename = f"vehicle_{timestamp}_{confidence:.3f}.jpg"  
                save_path = os.path.join(os.getcwd(), filename)  
                  
                try:  
                    # Save the image  
                    image = Image.fromarray(cropped_frame)  
                    image.save(save_path, format="JPEG", quality=95)  
                      
                    user_data.saved_images_count += 1  
                    saved_this_frame += 1  
                    string_to_print += f"    Saved vehicle image: {filename}\n"  
                    print(f"Saved vehicle image: {save_path}")  
                except Exception as e:  
                    string_to_print += f"    Error saving image: {e}\n"  
                      
            detection_count += 1    
        else:    
            other_count += 1    
    
    # Update detection statistics    
    user_data.add_detection_count(detection_count)    
        
    string_to_print += f"\nSummary: {car_count} cars, {other_count} other objects\n"  
    string_to_print += f"Images saved this frame: {saved_this_frame}, Total saved: {user_data.saved_images_count}\n"  
    string_to_print += f"Total detections so far: {user_data.total_detections}\n"    
    string_to_print += f"Recent detection counts: {user_data.detection_history}\n"    
    
    # Frame processing for visualization    
    if user_data.use_frame and frame is not None:    
        # Draw detection info on frame    
        cv2.putText(frame, f"Cars: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    
        cv2.putText(frame, f"Frame: {user_data.get_count()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    
        cv2.putText(frame, f"Total: {user_data.total_detections}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  
        cv2.putText(frame, f"Saved: {user_data.saved_images_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  
            
        # Draw bounding boxes for cars    
        for detection in detections:    
            if detection.get_label() == "car":    
                bbox = detection.get_bbox()    
                x1 = int(bbox.xmin() * width)    
                y1 = int(bbox.ymin() * height)    
                x2 = int((bbox.xmin() + bbox.width()) * width)    
                y2 = int((bbox.ymin() + bbox.height()) * height)  
                  
                # Check if this is marked for OCR  
                classifications = detection.get_objects_typed(hailo.HAILO_CLASSIFICATION)  
                is_ocr_candidate = any(c.get_label() == "candidate" for c in classifications)  
                  
                # Use different colors for OCR candidates  
                color = (0, 0, 255) if is_ocr_candidate else (0, 255, 0)  # Red for OCR, Green for normal  
                thickness = 3

                    
                # Draw rectangle    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)    
                # Draw confidence    
                cv2.putText(frame, f"{detection.get_confidence():.2f}",     
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  
                  
                # Draw OCR indicator if marked for saving  
                if is_ocr_candidate:  
                    cv2.putText(frame, "OCR", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  
            
        # Convert the frame to BGR    
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)    
        user_data.set_frame(frame)    
    
    print(string_to_print)    
    return Gst.PadProbeReturn.OK    
  
# -----------------------------------------------------------------------------------------------  
# Main Function  
# -----------------------------------------------------------------------------------------------  
if __name__ == "__main__":    
    print("Starting Hailo License Plate Recognition Debug App...")    
    print("This app will:")  
    print("- Show all vehicle detections on screen")  
    print("- Mark top 4 vehicles with red boxes for OCR processing")  
    print("- Save cropped images of top 4 vehicles in current directory")  
    print("- Use optimized tracker settings for responsive box removal")  
    print()  
      
    # Create an instance of the user app callback class    
    user_data = user_app_callback_class()    
    app = GStreamerDetectionApp(app_callback, user_data)    
    app.run()                