"""SAM2 interactive explorer — Dash app for testing prompts and propagation."""

import base64
import cv2
import numpy as np
import torch
from pathlib import Path

from dash import Dash, html, dcc, callback, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from sam2.build_sam import build_sam2_video_predictor

# ── Config ──────────────────────────────────────────────────────────────────
VIDEO_PATH = "/n/groups/datta/john/projects/sidb-ratseq/videos/2026_04_03_new_extraction_pipeline/1806757_14wk_1/cropped_height_3x.mp4"
SAM2_CHECKPOINT = "/n/groups/datta/john/repos/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_REPO = "/n/groups/datta/john/repos/sam2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FRAMES = 1000  # Only load first N frames for quick testing

# ── Load video metadata ────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
TOTAL_FRAMES = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), MAX_FRAMES)
FPS = cap.get(cv2.CAP_PROP_FPS)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Extract first N frames as JPEGs for SAM2 init_state
FRAMES_DIR = "/tmp/sam2_explorer_frames"
import os, shutil
if os.path.exists(FRAMES_DIR):
    shutil.rmtree(FRAMES_DIR)
os.makedirs(FRAMES_DIR)

print(f"Extracting {TOTAL_FRAMES} frames to {FRAMES_DIR}...")
cap = cv2.VideoCapture(VIDEO_PATH)
for i in range(TOTAL_FRAMES):
    ret, frame = cap.read()
    if not ret:
        TOTAL_FRAMES = i
        break
    cv2.imwrite(f"{FRAMES_DIR}/{i:06d}.jpg", frame)
cap.release()
print(f"Video: {TOTAL_FRAMES} frames, {WIDTH}x{HEIGHT}, {FPS:.1f} fps")


def read_frame(idx: int) -> np.ndarray:
    """Read a single frame as RGB numpy array."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def frame_to_b64(frame: np.ndarray) -> str:
    """Encode RGB frame as base64 JPEG for plotly image."""
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def make_figure(frame: np.ndarray, mask: np.ndarray | None = None, points: list | None = None) -> go.Figure:
    """Create a plotly figure with frame, optional mask overlay, and click points."""
    h, w = frame.shape[:2]

    fig = go.Figure()

    # Blend mask into frame for display
    display_frame = frame.copy()
    if mask is not None and mask.any():
        overlay_color = np.array([0, 120, 255], dtype=np.uint8)
        alpha = 0.4
        display_frame[mask > 0] = (
            display_frame[mask > 0] * (1 - alpha) + overlay_color * alpha
        ).astype(np.uint8)

    # Frame as layout_image (not clickable, just for display)
    fig.add_layout_image(
        source=frame_to_b64(display_frame),
        x=0, y=0,
        xref="x", yref="y",
        sizex=w, sizey=h,
        xanchor="left", yanchor="top",
        layer="below",
    )

    # Invisible heatmap for click capture (Heatmap fires clickData, Image/layout_image don't)
    fig.add_trace(go.Heatmap(
        z=np.zeros((h, w), dtype=np.uint8),
        x0=0, dx=1, y0=0, dy=1,
        showscale=False,
        opacity=0,
        hoverinfo="x+y",
    ))

    # Click points
    if points:
        pos_x = [p[0] for p in points if p[2] == 1]
        pos_y = [p[1] for p in points if p[2] == 1]
        neg_x = [p[0] for p in points if p[2] == 0]
        neg_y = [p[1] for p in points if p[2] == 0]

        if pos_x:
            fig.add_trace(go.Scatter(
                x=pos_x, y=pos_y, mode="markers",
                marker=dict(size=12, color="lime", symbol="circle",
                            line=dict(width=2, color="white")),
                name="Positive", hoverinfo="skip",
            ))
        if neg_x:
            fig.add_trace(go.Scatter(
                x=neg_x, y=neg_y, mode="markers",
                marker=dict(size=12, color="red", symbol="x",
                            line=dict(width=2, color="white")),
                name="Negative", hoverinfo="skip",
            ))

    fig.update_xaxes(range=[0, w], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[h, 0], showgrid=False, zeroline=False, visible=False, scaleanchor="x")
    fig.update_layout(
        width=w, height=h,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="black",
        showlegend=False,
    )
    return fig


# ── SAM2 predictor ──────────────────────────────────────────────────────────
print("Loading SAM2 model...")
predictor = build_sam2_video_predictor(
    SAM2_CONFIG,
    SAM2_CHECKPOINT,
    device=DEVICE,
)
print("SAM2 model loaded.")

# State
inference_state = None
current_mask = None  # (H, W) bool
prompts = []  # list of (x, y, label)  label: 1=pos, 0=neg
propagated_masks = {}  # frame_idx -> (H, W) bool


def init_session():
    """Initialize SAM2 video predictor session."""
    global inference_state, propagated_masks, prompts, current_mask
    inference_state = predictor.init_state(video_path=FRAMES_DIR)
    propagated_masks = {}
    prompts = []
    current_mask = None
    print("SAM2 session initialized")


def add_points(frame_idx: int, all_prompts: list):
    """Send ALL accumulated points and get mask prediction.

    SAM2's add_new_points_or_box has clear_old_points=True by default,
    so every call replaces previous points. We must send all points each time.
    """
    global current_mask
    points = np.array([[p[0], p[1]] for p in all_prompts], dtype=np.float32)
    labels = np.array([p[2] for p in all_prompts], dtype=np.int32)

    _, obj_ids, masks = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=0,
        points=points,
        labels=labels,
    )
    # masks shape: (num_objects, 1, H, W)
    current_mask = (masks[0, 0] > 0.0).cpu().numpy()
    return current_mask


def propagate():
    """Propagate masks through video."""
    global propagated_masks
    propagated_masks = {}
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
        mask = (masks[0, 0] > 0.0).cpu().numpy()
        propagated_masks[frame_idx] = mask
    print(f"Propagated to {len(propagated_masks)} frames")


# ── Dash App ────────────────────────────────────────────────────────────────
app = Dash(__name__)

app.layout = html.Div([
    html.H2("SAM2 Explorer", style={"textAlign": "center", "color": "#eee"}),

    html.Div([
        html.Div([
            html.Label("Frame:", style={"color": "#ccc"}),
            dcc.Slider(
                id="frame-slider",
                min=0, max=TOTAL_FRAMES - 1, step=1, value=0,
                marks={i: str(i) for i in range(0, TOTAL_FRAMES, max(1, TOTAL_FRAMES // 10))},
                tooltip={"placement": "bottom"},
            ),
        ], style={"flex": "1", "padding": "0 20px"}),

        html.Div([
            html.Button("+ Positive", id="btn-positive", n_clicks=0,
                        style={"backgroundColor": "#22c55e", "color": "white", "margin": "5px", "padding": "8px 16px", "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
            html.Button("- Negative", id="btn-negative", n_clicks=0,
                        style={"backgroundColor": "#ef4444", "color": "white", "margin": "5px", "padding": "8px 16px", "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
            html.Button("Propagate", id="btn-propagate", n_clicks=0,
                        style={"backgroundColor": "#3b82f6", "color": "white", "margin": "5px", "padding": "8px 16px", "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
            html.Button("Reset Session", id="btn-reset", n_clicks=0,
                        style={"backgroundColor": "#666", "color": "white", "margin": "5px", "padding": "8px 16px", "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={"display": "flex", "alignItems": "center", "padding": "10px"}),

    dcc.Graph(
        id="video-frame",
        config={"scrollZoom": False, "displayModeBar": False},
        style={"display": "flex", "justifyContent": "center"},
    ),

    html.Div(id="status", style={"textAlign": "center", "color": "#aaa", "padding": "10px"}),

    # Hidden stores
    dcc.Store(id="click-mode", data="positive"),
    dcc.Store(id="prompts-store", data=[]),
    dcc.Store(id="session-initialized", data=False),
], style={"backgroundColor": "#1a1a1a", "minHeight": "100vh", "fontFamily": "sans-serif"})


@callback(
    Output("click-mode", "data"),
    Input("btn-positive", "n_clicks"),
    Input("btn-negative", "n_clicks"),
    State("click-mode", "data"),
)
def toggle_mode(pos_clicks, neg_clicks, current):
    from dash import ctx
    if not ctx.triggered_id:
        return current
    if ctx.triggered_id == "btn-positive":
        return "positive"
    return "negative"


@callback(
    Output("session-initialized", "data"),
    Output("status", "children", allow_duplicate=True),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_session(n):
    init_session()
    return True, "Session reset. Click on the frame to add prompts."


@callback(
    Output("video-frame", "figure"),
    Output("prompts-store", "data"),
    Output("status", "children"),
    Output("session-initialized", "data", allow_duplicate=True),
    Input("frame-slider", "value"),
    Input("btn-propagate", "n_clicks"),
    State("prompts-store", "data"),
    State("session-initialized", "data"),
    prevent_initial_call="initial_duplicate",
)
def on_frame_change(frame_idx, prop_clicks, stored_prompts, session_init):
    """Handle frame navigation and propagation."""
    from dash import ctx
    global current_mask

    frame = read_frame(frame_idx)

    if ctx.triggered_id == "btn-propagate":
        propagate()
        mask = propagated_masks.get(frame_idx)
        status = f"Propagated to {len(propagated_masks)} frames. Scrub slider to see results."
        fig = make_figure(frame, mask, stored_prompts if stored_prompts else None)
        return fig, stored_prompts, status, session_init

    # Frame navigation
    mask = propagated_masks.get(frame_idx, current_mask if stored_prompts else None)
    show_points = stored_prompts if stored_prompts else None
    fig = make_figure(frame, mask, show_points)
    status = f"Frame {frame_idx}/{TOTAL_FRAMES - 1}"
    return fig, stored_prompts, status, session_init


@callback(
    Output("video-frame", "figure", allow_duplicate=True),
    Output("prompts-store", "data", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("video-frame", "clickData"),
    State("frame-slider", "value"),
    State("click-mode", "data"),
    State("prompts-store", "data"),
    State("session-initialized", "data"),
    prevent_initial_call=True,
)
def on_click(click_data, frame_idx, click_mode, stored_prompts, session_init):
    """Handle clicks on the video frame."""
    if not click_data or not session_init:
        raise PreventUpdate

    point = click_data["points"][0]
    x, y = int(point["x"]), int(point["y"])
    label = 1 if click_mode == "positive" else 0
    stored_prompts.append([x, y, label])

    print(f"Click: frame={frame_idx}, x={x}, y={y}, label={label}, total_prompts={len(stored_prompts)}")
    mask = add_points(frame_idx, stored_prompts)
    mask_pixels = int(mask.sum()) if mask is not None else 0
    print(f"Mask: {mask_pixels} pixels, shape={mask.shape if mask is not None else None}")

    frame = read_frame(frame_idx)
    status = f"Frame {frame_idx} | Point ({x}, {y}) | Mask: {mask_pixels}px"
    fig = make_figure(frame, mask, stored_prompts)
    return fig, stored_prompts, status


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7264)
