import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
import hailo
import debugpy
from hailo_apps_infra.hailo_rpi_common import (
    get_default_parser,
    detect_hailo_arch,
)
from hailo_apps_infra.gstreamer_helper_pipelines import(
    QUEUE,
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    CROPPER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
    FILE_SINK_PIPELINE,
    OVERLAY_PIPELINE
)
from hailo_apps_infra.gstreamer_app import (
    GStreamerApp,
    app_callback_class,
    dummy_callback
)



# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerTsrApp(GStreamerApp):
    def __init__(self, app_callback, user_data):
        parser = get_default_parser()
        parser.add_argument(
            "--labels-json",
            default=None,
            help="Path to costume labels JSON file",
        )
        args = parser.parse_args()
        # Call the parent class constructor
        super().__init__(args, user_data)
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.batch_size = 2
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45


        # Determine the architecture if not specified
        if args.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
            self.arch = detected_arch
            print(f"Auto-detected Hailo architecture: {self.arch}")
        else:
            self.arch = args.arch


        if args.hef_path is not None:
            self.hef_path = args.hef_path
        # Set the HEF file path based on the arch
        elif self.arch == "hailo8":
            self.detection_hef_path = os.path.join(self.current_path, '../resources/yolov8m.hef')
            self.depth_hef_path = os.path.join(self.current_path, '../resources/fast_depth.hef')
        else:  # hailo8l
            self.detection_hef_path = os.path.join(self.current_path, '../resources/yolov8s_h8l.hef')
            self.depth_hef_path = os.path.join(self.current_path, '../resources/fast_depth_h8l.hef')

        # Set the post-processing shared object file
        self.detection_post_process_so = os.path.join(self.current_path, '../resources/libyolo_hailortpp_postprocess.so')
        self.detection_post_function_name = "filter_letterbox"
        self.depth_post_process_so = os.path.join(self.current_path, '../resources/libdepth_postprocess.so')
        self.depth_post_function_name = "filter"
        self.post_process_so_cropper = os.path.join(self.current_path, '../resources/libtsr_croppers.so')
        self.cropper_post_function_name = "tsr_detections"
        # User-defined label JSON file
        self.labels_json = args.labels_json

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        self.save_video_path = user_data.save_video_path

        # Set the process title
        setproctitle.setproctitle("Hailo TSR App")

        self.create_pipeline()

    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(self.video_source, self.video_width, self.video_height)
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.detection_hef_path,
            post_process_so=self.detection_post_process_so,
            post_function_name=self.detection_post_function_name,
            batch_size=self.batch_size,
            config_json=self.labels_json,
            additional_params=self.thresholds_str,
            name='detection_inference')
        detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(class_id=12)  # for what class id across frames will be tracked. coco 1 based, so 12 is stop sign.
        depth_pipeline = INFERENCE_PIPELINE(
            hef_path=self.depth_hef_path,
            post_process_so=self.depth_post_process_so,
            post_function_name=self.depth_post_function_name,
            name='depth_inference')
        cropper_pipeline = CROPPER_PIPELINE(
            inner_pipeline=(f'{depth_pipeline}'),
            so_path=self.post_process_so_cropper,
            function_name=self.cropper_post_function_name,
            internal_offset=True
        )
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)
        file_overlay_pipeline=OVERLAY_PIPELINE(name='file_overlay')
        save_video_pipeline = FILE_SINK_PIPELINE(self.save_video_path)

        # both display and file save - slow performance
        # pipeline_string = (
        #     f'{source_pipeline} ! '
        #     f'{detection_pipeline_wrapper} ! '
        #     f'{tracker_pipeline} ! '
        #     f'{cropper_pipeline} ! '
        #     f'{user_callback_pipeline} ! '
        #     f'tee name=t t. ! '
        #     f'{display_pipeline} t. ! '
        #     f'{file_overlay_pipeline} ! '
        #     f'{save_video_pipeline}'
        # )

        # only display - faster performance
        pipeline_string = (
            f'{source_pipeline} ! '
            f'{detection_pipeline_wrapper} ! '
            f'{tracker_pipeline} ! '
            f'{cropper_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )

        # only file save - faster performance
        # pipeline_string = (
        #     f'{source_pipeline} ! '
        #     f'{detection_pipeline_wrapper} ! '
        #     f'{tracker_pipeline} ! '
        #     f'{cropper_pipeline} ! '
        #     f'{user_callback_pipeline} ! '
        #     f'{file_overlay_pipeline} ! '
        #     f'{save_video_pipeline}'
        # )
        # print(pipeline_string)
        return pipeline_string

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = app_callback_class()
    app_callback = dummy_callback
    app = GStreamerTsrApp(app_callback, user_data)
    app.run()
