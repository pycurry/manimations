import cv2
import numpy as np


def draw_28x28(brush_width=1.4, scale=8):
    """Display a drawing canvas and return a 28x28 float array on Submit."""
    size = 28
    raster_size = size * scale
    canvas_size = 560
    toolbar_height = 72
    raster = np.full((raster_size, raster_size), 255, dtype=np.uint8)
    stroke_width = max(1, round(brush_width * scale))
    state = {"drawing": False, "last": None, "submitted": False}

    def downsample():
        resized = cv2.resize(
            raster, (size, size), interpolation=cv2.INTER_LANCZOS4
        )
        return resized.astype(np.float32)

    def to_raster_coords(x, y):
        x = round(np.clip(x, 0, canvas_size - 1) * raster_size / canvas_size)
        y = round(np.clip(y, 0, canvas_size - 1) * raster_size / canvas_size)
        return x, y

    def on_mouse(event, x, y, _flags, _parameters):
        if event == cv2.EVENT_LBUTTONDOWN:
            if y >= canvas_size:
                if 20 <= x <= 260:
                    raster.fill(255)
                elif 300 <= x <= 540:
                    state["submitted"] = True
                return
            state["drawing"] = True
            state["last"] = to_raster_coords(x, y)
            cv2.circle(raster, state["last"], stroke_width // 2, 0, -1)
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            current = to_raster_coords(x, y)
            cv2.line(raster, state["last"], current, 0, stroke_width)
            cv2.circle(raster, current, stroke_width // 2, 0, -1)
            state["last"] = current
        elif event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            state["last"] = None

    def render():
        canvas = cv2.resize(
            raster, (canvas_size, canvas_size), interpolation=cv2.INTER_LINEAR
        )
        frame = np.full(
            (canvas_size + toolbar_height, canvas_size, 3),
            (245, 245, 245),
            dtype=np.uint8,
        )
        frame[:canvas_size] = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(frame, (20, 574), (260, 634), (75, 84, 78), -1)
        cv2.rectangle(frame, (300, 574), (540, 634), (30, 112, 92), -1)
        cv2.putText(frame, "Clear (C)", (92, 612), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Submit (Enter)", (330, 612), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    window_name = "Draw a digit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while not state["submitted"]:
            cv2.imshow(window_name, render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("c"), ord("C")):
                raster.fill(255)
            elif key in (10, 13):
                state["submitted"] = True
            elif key == 27:
                raise RuntimeError("Drawing cancelled before submission.")
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("Drawing window closed before submission.")
    finally:
        cv2.destroyWindow(window_name)

    return 1.0 - downsample() / 255.0


def main():
    array = draw_28x28()
    print(array.shape)
    print(array)


if __name__ == "__main__":
    main()


