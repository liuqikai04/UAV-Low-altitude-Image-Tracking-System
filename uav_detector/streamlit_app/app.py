import os
import cv2
import atexit
import tempfile
import streamlit as st
from PIL import Image

from uav_inference import (
    get_available_models,
    get_model_status,
    get_video_info,
    get_video_frame,
    detect_and_segment_on_frame,
    track_selected_object,
)


def _safe_remove_file(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


@atexit.register
def _cleanup_uploaded_temp_file():
    uploaded_path = st.session_state.get("uploaded_video_path") if hasattr(st, "session_state") else None
    _safe_remove_file(uploaded_path)


def _localize_file_uploader():
    st.markdown(
        """
        <style>
        [data-testid="stFileUploaderDropzoneInstructions"] span {
            font-size: 0;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] span::after {
            content: "拖拽文件到此处";
            font-size: 1rem;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] small {
            font-size: 0;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] small::after {
            content: "单个文件限制 200MB · MP4、AVI、MOV、MKV、MPEG4";
            font-size: 0.875rem;
        }

        [data-testid="stFileUploaderDropzone"] button {
            font-size: 0;
        }

        [data-testid="stFileUploaderDropzone"] button::after {
            content: "浏览文件";
            font-size: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _get_selected_detection(detections, selected_index):
    for det in detections:
        if det["index"] == selected_index:
            return det
    return detections[0] if detections else None


def _highlight_selected_detection(base_frame, selected_det):
    if selected_det is None:
        return base_frame

    highlighted = base_frame.copy()
    x1, y1, x2, y2 = [int(v) for v in selected_det["bbox"]]
    label = f"{selected_det['index']}:{selected_det['class_name']} {selected_det['score']:.2f}"

    cv2.rectangle(highlighted, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        highlighted,
        label,
        (x1, max(16, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )
    return highlighted


st.set_page_config(page_title="无人机单目标跟踪系统", layout="wide")
st.title("无人机低空图像单目标跟踪系统")
_localize_file_uploader()

model_status = get_model_status()
if not model_status["ready"]:
    st.error(model_status["message"])
    st.info("请下载 `yolov8n-seg.pt` 后放到提示路径，再重新运行页面。")
    st.stop()

st.sidebar.header("推理参数")
available_models = get_available_models()
if "yolov8s-seg" in available_models:
    default_model = "yolov8s-seg"
elif "visdrone_sot_best" in available_models:
    default_model = "visdrone_sot_best"
else:
    default_model = available_models[0] if available_models else "yolov8n-seg"
selected_model = st.sidebar.selectbox(
    "检测/分割模型",
    options=available_models if available_models else ["yolov8n-seg"],
    index=(available_models.index(default_model) if available_models else 0),
)
conf_thresh = st.sidebar.slider("置信度阈值", 0.05, 1.0, 0.1, 0.05)
nms_thresh = st.sidebar.slider("NMS IoU 阈值", 0.1, 1.0, 0.45, 0.05)
input_size = st.sidebar.selectbox(
    "输入尺寸（宽 x 高）",
    options=[(640, 640), (896, 896), (1024, 1024), (1024, 1536), (1536, 1536), (2048, 2048)],
    index=4,
)
enhance_small_objects = st.sidebar.checkbox("小目标增强检测（较慢）", value=True)
only_vehicles = st.sidebar.checkbox("仅检测车辆（car/bus/truck/van）", value=True)
with_mask = st.sidebar.checkbox("追踪视频显示分割掩码", value=True)

if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = None
if "uploaded_video_name" not in st.session_state:
    st.session_state.uploaded_video_name = None
if "video_info" not in st.session_state:
    st.session_state.video_info = None
if "detected_frame_idx" not in st.session_state:
    st.session_state.detected_frame_idx = None
if "detections" not in st.session_state:
    st.session_state.detections = []
if "selected_detection_index" not in st.session_state:
    st.session_state.selected_detection_index = None
if "detected_frame_image" not in st.session_state:
    st.session_state.detected_frame_image = None
if "selected_frame" not in st.session_state:
    st.session_state.selected_frame = 0
if "tracking_output_path" not in st.session_state:
    st.session_state.tracking_output_path = None
if "tracking_output_bytes" not in st.session_state:
    st.session_state.tracking_output_bytes = None
if "tracking_target_id" not in st.session_state:
    st.session_state.tracking_target_id = None
if "uploaded_video_bytes" not in st.session_state:
    st.session_state.uploaded_video_bytes = None

video_file = st.file_uploader("1) 上传无人机视频", type=["mp4", "avi", "mov", "mkv"])
if video_file is not None:
    current_name = st.session_state.get("uploaded_video_name")
    if st.session_state.uploaded_video_path is None or current_name != video_file.name:
        _safe_remove_file(st.session_state.uploaded_video_path)
        suffix = os.path.splitext(video_file.name)[1] or ".mp4"
        uploaded_bytes = video_file.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_bytes)
            st.session_state.uploaded_video_path = tmp.name
        st.session_state.uploaded_video_bytes = uploaded_bytes
        st.session_state.uploaded_video_name = video_file.name
        st.session_state.video_info = get_video_info(st.session_state.uploaded_video_path)
        st.session_state.detected_frame_idx = None
        st.session_state.detections = []
        st.session_state.selected_detection_index = None
        st.session_state.detected_frame_image = None
        st.session_state.tracking_output_path = None
        st.session_state.tracking_output_bytes = None
        st.session_state.tracking_target_id = None

if st.session_state.uploaded_video_path and st.session_state.video_info:
    video_path = st.session_state.uploaded_video_path
    info = st.session_state.video_info

    if st.session_state.uploaded_video_bytes is not None:
        st.video(st.session_state.uploaded_video_bytes)
    else:
        st.video(video_path)
    st.caption(
        f"视频信息：{info['width']}x{info['height']} | FPS={info['fps']:.2f} | 总帧数={info['frame_count']}"
    )

    max_frame = max(0, info["frame_count"] - 1)
    st.session_state.selected_frame = min(st.session_state.selected_frame, max_frame)
    frame_step = st.number_input("步长（每次跳转帧数）", min_value=1, max_value=300, value=1, step=1)
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("上一帧", use_container_width=True):
            st.session_state.selected_frame = max(0, st.session_state.selected_frame - int(frame_step))
    with col_next:
        if st.button("下一帧", use_container_width=True):
            st.session_state.selected_frame = min(max_frame, st.session_state.selected_frame + int(frame_step))

    selected_frame = st.slider(
        "2) 拖动进度条选择识别帧",
        min_value=0,
        max_value=max_frame,
        key="selected_frame",
        step=1,
    )
    preview_frame = get_video_frame(video_path, selected_frame)
    preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
    st.image(
        Image.fromarray(preview_rgb),
        caption=f"当前选择帧预览：frame={selected_frame}",
        use_container_width=True,
    )

    if st.button("3) 识别当前帧中的所有目标", use_container_width=True):
        with st.spinner("正在识别..."):
            frame = get_video_frame(video_path, selected_frame)
            vis_frame, detections = detect_and_segment_on_frame(
                frame,
                conf_thresh=conf_thresh,
                nms_thresh=nms_thresh,
                test_size=input_size,
                model_name=selected_model,
                enhance_small_objects=enhance_small_objects,
                only_vehicles=only_vehicles,
            )
        st.session_state.detected_frame_idx = selected_frame
        st.session_state.detections = detections
        st.session_state.selected_detection_index = detections[0]["index"] if detections else None
        st.session_state.detected_frame_image = vis_frame

    if st.session_state.detected_frame_image is not None:
        selected_det = _get_selected_detection(
            st.session_state.detections,
            st.session_state.selected_detection_index,
        )
        display_frame = _highlight_selected_detection(st.session_state.detected_frame_image, selected_det)
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        st.image(Image.fromarray(rgb), caption="识别结果（框内编号用于选择目标）", use_container_width=True)

    if st.session_state.detections:
        detection_indices = [det["index"] for det in st.session_state.detections]
        if st.session_state.selected_detection_index not in detection_indices:
            st.session_state.selected_detection_index = detection_indices[0]

        def _format_detection_option(det_index):
            det = _get_selected_detection(st.session_state.detections, det_index)
            x1, y1, x2, y2 = det["bbox"]
            return f"[{det['index']}] {det['class_name']} 置信度={det['score']:.2f} 边界框=({x1},{y1},{x2},{y2})"

        selected_det_idx = st.selectbox(
            "4) 选择要追踪的目标",
            options=detection_indices,
            format_func=_format_detection_option,
            key="selected_detection_index",
        )
        selected_det = _get_selected_detection(st.session_state.detections, selected_det_idx)

        if st.button("5) 开始追踪并导出视频", use_container_width=True):
            progress = st.progress(0)
            with st.spinner("正在进行单目标追踪，这可能需要一些时间..."):
                out_path, target_track_id = track_selected_object(
                    video_path,
                    selected_frame_idx=st.session_state.detected_frame_idx,
                    selected_bbox=selected_det["bbox"],
                    conf_thresh=conf_thresh,
                    nms_thresh=nms_thresh,
                    test_size=input_size,
                    progress=progress,
                    with_mask=with_mask,
                    model_name=selected_model,
                    enhance_small_objects=enhance_small_objects,
                    only_vehicles=only_vehicles,
                )
            if os.path.exists(out_path):
                st.session_state.tracking_output_path = out_path
                st.session_state.tracking_target_id = target_track_id
            else:
                st.error("追踪输出失败，未找到结果视频。")
    elif st.session_state.detected_frame_idx is not None:
        st.warning("该帧没有检测到可选目标，请换一帧重试。")

    if st.session_state.tracking_output_path is not None and os.path.exists(st.session_state.tracking_output_path):
        if st.session_state.tracking_target_id is None:
            st.warning("在所选帧未能稳定匹配到追踪 ID，结果视频可能不会出现目标框。")
        else:
            st.success(f"追踪完成。目标 ID={st.session_state.tracking_target_id}")
        st.video(st.session_state.tracking_output_path)
        with open(st.session_state.tracking_output_path, "rb") as f:
            st.download_button(
                "下载追踪视频",
                data=f,
                file_name=os.path.basename(st.session_state.tracking_output_path or "track_single.mp4"),
                mime="video/mp4",
                use_container_width=True,
            )
