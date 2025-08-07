# region imports
# Standard library imports
import datetime
from datetime import datetime
import threading
from pathlib import Path

# Third-party imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Local application-specific imports
import hailo
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.license_plate_recognition.license_plate_recognition_pipeline import GstreamerLPRAppfrom hailo_apps.hailo_app_python.core.common.defines import HAILO_LOGO_PHOTO_NAME
from hailo_apps.hailo_app_python.core.common.defines import HAILO_LOGO_PHOTO_NAME
