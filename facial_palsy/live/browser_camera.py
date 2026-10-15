
import threading
import time

import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from live.live_camera import (
    LANDMARKER_PATH,
    STEPS,
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_EYEBROW,
    RIGHT_EYEBROW,
    MOUTH,
    FaceLandmarker,
    FaceLandmarkerOptions,
    BaseOptions,
    VisionRunningMode,
    calculate_measurements,
    calculate_step_score,
    get_facial_result,
    draw_points,
)


class BrowserAssessment:
    def __init__(self):
        self.lock = threading.Lock()
        self.step_index = 0
        self.results = {}
        self.state = "READY"
        self.capture_start = None
        self.result = None
        self.error = None
        self.last_timestamp = 0

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(LANDMARKER_PATH)
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = FaceLandmarker.create_from_options(
            options
        )

    def start_step(self):
        with self.lock:
            if self.state == "READY":
                self.state = "CAPTURING"
                self.capture_start = time.monotonic()
                self.error = None

    def restart(self):
        with self.lock:
            self.step_index = 0
            self.results = {}
            self.state = "READY"
            self.capture_start = None
            self.result = None
            self.error = None

    def get_result(self):
        with self.lock:
            return self.result

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass

    def process_frame(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        height, width = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        timestamp = max(
            int(time.monotonic() * 1000),
            self.last_timestamp + 1,
        )
        self.last_timestamp = timestamp

        detection = self.landmarker.detect_for_video(
            mp_image, timestamp
        )

        landmarks = (
            detection.face_landmarks[0]
            if detection.face_landmarks
            else None
        )

        with self.lock:
            state = self.state
            step_index = self.step_index
            capture_start = self.capture_start

        if state != "COMPLETE":
            step = STEPS[step_index]

            cv2.putText(
                image,
                f"Movement {step_index + 1}/5: {step['name']}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                image,
                step["instruction"],
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            if landmarks is None:
                cv2.putText(
                    image,
                    "NO FACE DETECTED",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            else:
                for indices, label in [
                    (LEFT_EYE, "L EYE"),
                    (RIGHT_EYE, "R EYE"),
                    (LEFT_EYEBROW, "L BROW"),
                    (RIGHT_EYEBROW, "R BROW"),
                    (MOUTH, "MOUTH"),
                ]:
                    draw_points(
                        image,
                        landmarks,
                        indices,
                        width,
                        height,
                        label,
                    )

            if state == "CAPTURING" and capture_start is not None:
                elapsed = time.monotonic() - capture_start
                duration = STEPS[step_index]["duration"]
                remaining = max(0, int(duration - elapsed + 0.999))

                cv2.putText(
                    image,
                    f"Hold position: {remaining}s",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                if elapsed >= duration:
                    with self.lock:
                        if (
                            self.state == "CAPTURING"
                            and self.step_index == step_index
                        ):
                            if landmarks is None:
                                self.state = "READY"
                                self.capture_start = None
                                self.error = (
                                    "No face detected during capture. "
                                    "Please start this movement again."
                                )
                            else:
                                measurements = calculate_measurements(
                                    landmarks, width, height
                                )
                                step_name = STEPS[step_index]["name"]
                                score = calculate_step_score(
                                    step_name, measurements
                                )

                                self.results[step_name] = {
                                    "measurements": measurements,
                                    "asymmetry": score,
                                }

                                self.step_index += 1
                                self.capture_start = None

                                if self.step_index >= len(STEPS):
                                    self.result = get_facial_result(
                                        self.results
                                    )
                                    self.state = "COMPLETE"
                                else:
                                    self.state = "READY"

        else:
            cv2.putText(
                image,
                "ASSESSMENT COMPLETE",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        return av.VideoFrame.from_ndarray(
            image, format="bgr24"
        )


def render_browser_assessment():
    st.markdown("### Live Facial Motor Assessment")
    st.info(
        "Allow camera access when your browser asks. "
        "Follow each instruction and press Start Movement "
        "when you are ready. Hold the movement for four seconds."
    )

    if "neuro_browser_assessment" not in st.session_state:
        st.session_state.neuro_browser_assessment = (
            BrowserAssessment()
        )

    assessment = st.session_state.neuro_browser_assessment

    ctx = webrtc_streamer(
        key="neuro-facial-assessment",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=assessment.process_frame,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    result = assessment.get_result()

    if result is not None:
        st.success("Facial assessment completed.")
        st.metric(
            "Assessment",
            result["facial_assessment"],
        )
        st.metric(
            "Asymmetry",
            f"{result['asymmetry_percentage']:.2f}%",
        )

        with st.expander("Movement measurements"):
            st.json(result["step_results"])

        if st.button("Restart Facial Assessment"):
            assessment.restart()
            st.rerun()

        return result

    if assessment.error:
        st.warning(assessment.error)

    if not ctx.state.playing:
        st.caption(
            "Start the webcam using the Start button above the video."
        )
    else:
        with assessment.lock:
            state = assessment.state
            step_index = assessment.step_index

        if state == "READY":
            st.write(
                f"**Movement {step_index + 1}/5:** "
                f"{STEPS[step_index]['instruction']}"
            )

            if st.button(
                "Start Movement",
                type="primary",
                key=f"start_movement_{step_index}",
            ):
                assessment.start_step()
                st.rerun()

        elif state == "CAPTURING":
            st.info(
                "Movement is being captured. Keep holding "
                "your position until the four-second capture ends."
            )
            time.sleep(0.5)
            st.rerun()

    return None
