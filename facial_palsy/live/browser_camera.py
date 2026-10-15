
import threading
import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer


class BrowserCamera:
    """Receive webcam frames from the user's browser."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_count = 0

    def video_frame_callback(self, frame):
        image = frame.to_ndarray(format="bgr24")

        with self.lock:
            self.latest_frame = image.copy()
            self.frame_count += 1

        return av.VideoFrame.from_ndarray(
            image, format="bgr24"
        )

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()


def show_browser_camera():
    """Render a browser webcam and return the latest frame."""
    if "neuro_browser_camera" not in st.session_state:
        st.session_state.neuro_browser_camera = BrowserCamera()

    camera = st.session_state.neuro_browser_camera

    ctx = webrtc_streamer(
        key="neuro-facial-assessment-camera",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=camera.video_frame_callback,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    if ctx.state.playing:
        time.sleep(0.1)
        return camera.get_frame()

    return None

