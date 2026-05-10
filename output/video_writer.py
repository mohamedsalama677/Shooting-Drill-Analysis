"""Thin context manager around cv2.VideoWriter."""

import cv2
import numpy as np


class AnnotatedVideoWriter:
    def __init__(self, path: str, fps: float, frame_size: tuple):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise IOError(f"Failed to open VideoWriter for {path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.release()
