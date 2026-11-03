import threading
import time

import cv2
import requests
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI(title="DiagnoBot Camera Bridge")

ANALYSIS_URL = "http://127.0.0.1:8000/api/v1/analyze/upload"

# One shared webcam instance for preview AND capture
camera = None
camera_lock = threading.Lock()


def get_camera():
    global camera

    if camera is None or not camera.isOpened():
        if camera is not None:
            camera.release()

        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not camera.isOpened():
            camera.release()
            camera = None
            raise HTTPException(503, "Laptop webcam unavailable")

    return camera


def read_shared_frame():
    with camera_lock:
        cam = get_camera()
        success, frame = cam.read()

        if not success or frame is None:
            raise HTTPException(503, "Could not read webcam frame")

        return frame.copy()


@app.get("/health")
def health():
    return {"status": "connected", "camera_bridge": "ready"}


@app.post("/capture")
def capture(
    analysis_mode: str = Form("skin"),
    patient_id: str = Form(""),
    temperature: str = Form(""),
    symptoms: str = Form("")
):
    # Reuse the same webcam as /video. Do not open another camera.
    frame = read_shared_frame()

    success, encoded = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    if not success:
        raise HTTPException(500, "Image encoding failed")

    try:
        response = requests.post(
            ANALYSIS_URL,
            files={
                "file": (
                    "diagnobot_capture.jpg",
                    encoded.tobytes(),
                    "image/jpeg"
                )
            },
            data={
                "analysis_mode": analysis_mode,
                "patient_id": patient_id,
                "temperature": temperature,
                "symptoms": symptoms
            },
            timeout=300
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as exc:
        raise HTTPException(502, f"Analysis backend error: {exc}")


def webcam_stream():
    while True:
        try:
            frame = read_shared_frame()
            frame = cv2.resize(frame, (480, 320))

            success, encoded = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, 65]
            )

            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )

            time.sleep(0.08)

        except Exception as exc:
            print("Webcam stream error:", repr(exc))
            break


@app.get("/video")
def video():
    return StreamingResponse(
        webcam_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.on_event("shutdown")
def release_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
