"""Shared ONNX-based face enhancement utilities for GPEN-BFR models.

Provides session creation, pre/post processing, and the core
enhance-face-via-ONNX pipeline.
"""

import os
import platform
import threading
from typing import Any

import cv2
import numpy as np
import onnxruntime

import modules.globals
from modules.platform_info import OPENVINO_PROVIDER_CONFIG

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Tuned CUDA EP options shared by every session builder. Rationale:
# - arena kNextPowerOfTwo: fewer allocator rounds on large-VRAM cards.
# - cudnn_conv_algo_search HEURISTIC: EXHAUSTIVE autotune stalls startup
#   for zero steady-state gain on fixed-shape models (det/swap/enhancer).
# - do_copy_in_default_stream: async H2D copies without per-copy sync —
#   the inswapper/det pipeline is copy-bound at small batch sizes.
CUDA_PROVIDER_OPTIONS = {
    "arena_extend_strategy": "kNextPowerOfTwo",
    "cudnn_conv_algo_search": "HEURISTIC",
    "do_copy_in_default_stream": "1",
}

# Limit concurrent ONNX calls to avoid VRAM exhaustion on multi-face frames
THREAD_SEMAPHORE = threading.Semaphore(min(max(1, (os.cpu_count() or 1)), 8))


def _cpu_count() -> int:
    return os.cpu_count() or 12


def _is_dml(providers=None) -> bool:
    if providers is None:
        providers = modules.globals.execution_providers
    return any(
        (p[0] if isinstance(p, tuple) else p) == "DmlExecutionProvider"
        for p in providers
    )


def get_session_options(for_providers=None) -> "onnxruntime.SessionOptions":
    """Return a tuned SessionOptions for the active execution provider.

    Tuning rationale (esp. DirectML on Polaris/RX580 + 12-core Intel CPU):

    - ``ORT_ENABLE_ALL`` graph optimisations fuse ops before they reach the
      DML EP, reducing CPU<->GPU partition boundaries (the main DML cost).
    - ``intra_op_num_threads`` high: many DML-fallback ops (Shape, Slice,
      Resize, NMS post-processing) still run on CPU, so the 12-core CPU
      should not be starved. Leaves 1-2 cores for FFmpeg/UI.
    - ``inter_op_num_threads=1`` + ``ORT_SEQUENTIAL``: DML serialises GPU
      work internally (plus the app serialises via ``dml_lock``), so
      parallel branches only add contention on an 8GB RX580.
    - Memory arena/pattern/reuse ON: avoids re-allocation per frame for the
      fixed-shape models used here (det 640x640, swap 128x128, rec 112x112).
    """
    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    cpu = _cpu_count()
    is_dml = _is_dml(for_providers)
    if is_dml:
        # RX580 8GB + 12-core server: feed CPU fallback ops generously.
        opts.intra_op_num_threads = max(4, min(cpu - 1, 16))
        opts.inter_op_num_threads = 1
    else:
        opts.intra_op_num_threads = max(4, min(cpu - 2, 16))
        opts.inter_op_num_threads = 1
    try:
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    except Exception:
        pass
    try:
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True
        opts.enable_mem_reuse = True
    except Exception:
        pass
    try:
        # Share one intra-op pool across all sessions (det/rec/lmk/swap/
        # enhancer). Default (per-session pools) gives N sessions x M
        # threads contending on the same cores — pure oversubscription.
        opts.add_session_config_entry("use_per_session_threads", "0")
    except Exception:
        pass
    if is_dml:
        # Keep arena growth conservative on 8GB VRAM cards: reuse instead
        # of extending aggressively. CUDA keeps the default
        # kNextPowerOfTwo (fewer allocations on large-VRAM cards).
        try:
            opts.add_session_config_entry("arena_extend_strategy", "kSameAsRequested")
        except Exception:
            pass
    opts.log_severity_level = 3
    return opts


def build_provider_config(providers=None):
    """Wrap raw provider name strings with optimised CUDA / CoreML options.

    Providers that are already ``(name, options_dict)`` tuples are passed
    through unchanged.  Non-CUDA providers are left as bare strings.

    Ensures a CPU fallback follows any GPU provider (DML/CUDA/CoreML/
    OpenVINO). Without it, ops unsupported on DirectML fail instead of
    falling back to CPU — a major slowdown/crash source on RX580/Polaris
    where severalanse ops lack DML kernels.
    """
    if providers is None:
        providers = modules.globals.execution_providers

    config = []
    for p in providers:
        if isinstance(p, tuple):
            # Already configured – pass through
            config.append(p)
        elif p == "CUDAExecutionProvider":
            # Tuned options (see CUDA_PROVIDER_OPTIONS). HEURISTIC — not
            # EXHAUSTIVE — cudnn search: fixed-shape models gain nothing
            # from exhaustive autotune but pay seconds at startup.
            config.append(("CUDAExecutionProvider", dict(CUDA_PROVIDER_OPTIONS)))
        elif p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
            config.append((
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "AllowLowPrecisionAccumulationOnGPU": 1,
                },
            ))
        elif p == "OpenVINOExecutionProvider":
            # AUTO lets OpenVINO select the best device
            config.append(OPENVINO_PROVIDER_CONFIG)
        else:
            # DmlExecutionProvider takes no provider options — bare string
            # is correct. Polaris (RX580) lacks fast FP16, so keep FP32
            # models; precision is selected at model-choice time.
            config.append(p)
    # Append CPU fallback if a GPU provider is primary and CPU is missing.
    names = [c[0] if isinstance(c, tuple) else c for c in config]
    if names and names[0] != "CPUExecutionProvider" and "CPUExecutionProvider" not in names:
        try:
            import onnxruntime as _ort

            if "CPUExecutionProvider" in _ort.get_available_providers():
                config.append("CPUExecutionProvider")
        except Exception:
            config.append("CPUExecutionProvider")
    return config


def run_inference(session: onnxruntime.InferenceSession,
                  input_name: str,
                  input_tensor: "np.ndarray") -> "np.ndarray":
    """Run ONNX inference, using IO binding when a GPU session is active.

    IO binding uploads the input directly to device memory and lets ONNX
    Runtime allocate the output on-device (single D2H copy on readback),
    instead of the extra CPU staging allocation of ``session.run``.
    Covers CUDA (``cuda``) and DirectML (``dml``) sessions. Falls back to
    the standard ``session.run`` path for CPU providers or if binding
    fails (e.g. VRAM pressure).
    """
    _providers = session.get_providers()
    _device = (
        "cuda" if "CUDAExecutionProvider" in _providers
        else "dml" if "DmlExecutionProvider" in _providers
        else None
    )
    if _device is not None:
        try:
            io_binding = session.io_binding()

            # Input: numpy → device
            ort_input = onnxruntime.OrtValue.ortvalue_from_numpy(
                input_tensor, _device, 0,
            )
            io_binding.bind_ortvalue_input(input_name, ort_input)

            # Output: allocate on device (avoids a CPU-side allocation)
            output_name = session.get_outputs()[0].name
            io_binding.bind_output(output_name, _device, 0)

            session.run_with_iobinding(io_binding)

            return io_binding.get_outputs()[0].numpy()
        except Exception:
            # Fall back to standard path (e.g. ORT version mismatch,
            # unsupported op, or VRAM pressure)
            pass

    return session.run(None, {input_name: input_tensor})[0]


def create_onnx_session(model_path: str) -> onnxruntime.InferenceSession:
    """Create an ONNX Runtime session with optimised provider config.

    On Apple Silicon, applies CoreML graph optimizations (Pad decomposition,
    Shape/Gather folding, Split decomposition) to reduce CPU↔ANE partition
    boundaries.
    """
    if IS_APPLE_SILICON:
        from modules.onnx_optimize import optimize_for_coreml
        # Infer input shape from the model for Shape/Gather folding
        try:
            import onnx
            m = onnx.load(model_path)
            inp = m.graph.input[0]
            dims = inp.type.tensor_type.shape.dim
            shape = tuple(d.dim_value for d in dims if d.dim_value > 0)
            input_shape = shape if len(shape) == 4 else None
        except Exception:
            input_shape = None
        model_path = optimize_for_coreml(model_path, input_shape=input_shape)

    providers = build_provider_config()
    session_options = get_session_options(providers)
    session = onnxruntime.InferenceSession(
        model_path, sess_options=session_options, providers=providers,
    )
    warmup_session(session)
    return session


def _warmup_dim(axis: int, rank: int, dim) -> int:
    """Resolve one model dim to a concrete warmup size.

    Static dims pass through. Dynamic/symbolic dims (str, None, <=0 —
    e.g. SCRFD's ``[batch, 3, height, width]``) need realistic values:
    batch-like axes → 1, spatial H/W of image tensors → 640 (the
    detection size). A blanket ``1`` produces degenerate inputs like
    1x3x1x1, under which valid models fail — e.g. SCRFD's ``MaxPool_9``
    (k=3) is rejected by DirectML with 80070057.
    """
    if isinstance(dim, int) and dim > 0:
        return dim
    if rank == 4 and axis in (2, 3):
        return 640
    return 1


def warmup_session(session: onnxruntime.InferenceSession) -> None:
    """Run a dummy inference pass to trigger JIT / compile caching."""
    try:
        input_feed = {}
        for inp in session.get_inputs():
            shape = inp.shape
            rank = len(shape)
            concrete = [_warmup_dim(i, rank, d) for i, d in enumerate(shape)]
            input_feed[inp.name] = np.zeros(concrete, dtype=np.float32)
        session.run(None, input_feed)
    except Exception as e:
        print(f"ONNX enhancer warmup skipped (non-fatal): {e}")


def preprocess_face(face_img: np.ndarray, input_size: int) -> np.ndarray:
    """Resize, normalize, and convert a BGR face crop to ONNX input blob.

    GPEN-BFR expects [1, 3, H, W] float32 in RGB, normalized to [-1, 1].
    Math is fused in-place (``x * (2/255) - 1``) to avoid the temporaries
    of the naive ``x / 255 * 2 - 1`` chain (2 fewer full-tensor passes).
    """
    # The warpAffine crop is already input_size x input_size in the normal
    # path — re-resizing to the same size is a pure wasted copy.
    if face_img.shape[1] != input_size or face_img.shape[0] != input_size:
        face_img = cv2.resize(face_img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32)
    blob *= 0.007843137718737125  # 2/255
    blob -= 1.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(blob)


def postprocess_face(output: np.ndarray) -> np.ndarray:
    """Convert ONNX output [1, 3, H, W] float32 back to BGR uint8 image.

    Fused ``(x + 1) * 127.5`` with out= params — one pass instead of three.
    """
    img = output[0].transpose(1, 2, 0)
    np.add(img, 1.0, out=img)
    img *= 127.5
    np.clip(img, 0, 255, out=img)
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _get_face_affine(face: Any, input_size: int):
    """Compute affine transform to align a face to GPEN input space.

    Returns (M, inv_M) — forward and inverse affine matrices.
    """
    template = np.array([
        [0.31556875, 0.4615741],
        [0.68262291, 0.4615741],
        [0.50009375, 0.6405054],
        [0.34947187, 0.8246919],
        [0.65343645, 0.8246919],
    ], dtype=np.float32) * input_size

    landmarks = None
    if hasattr(face, "kps") and face.kps is not None:
        landmarks = face.kps.astype(np.float32)
    elif hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        lm106 = face.landmark_2d_106
        landmarks = np.array([
            lm106[38],  # left eye
            lm106[88],  # right eye
            lm106[86],  # nose tip
            lm106[52],  # left mouth
            lm106[61],  # right mouth
        ], dtype=np.float32)

    if landmarks is None or len(landmarks) < 5:
        return None, None

    M = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)[0]
    if M is None:
        return None, None
    inv_M = cv2.invertAffineTransform(M)
    return M, inv_M


# Feather-mask cache: the blend mask depends only on input_size, but was
# rebuilt (4x linspace + broadcasting over 512x512) on every face of every
# frame. Cache per size like face_enhancer.py does.
_FEATHER_MASK_CACHE: dict = {'mask': None, 'size': 0}


def _get_feather_mask(input_size: int) -> np.ndarray:
    if _FEATHER_MASK_CACHE['size'] != input_size:
        mask = np.ones((input_size, input_size), dtype=np.float32)
        border = max(1, input_size // 16)
        mask[:border, :] = np.linspace(0, 1, border)[:, np.newaxis]
        mask[-border:, :] = np.linspace(1, 0, border)[:, np.newaxis]
        mask[:, :border] = np.minimum(mask[:, :border], np.linspace(0, 1, border)[np.newaxis, :])
        mask[:, -border:] = np.minimum(mask[:, -border:], np.linspace(1, 0, border)[np.newaxis, :])
        # uint8 for the cv2 SIMD blend (1/255 feather quantization is
        # visually identical to the float32 path).
        _FEATHER_MASK_CACHE['mask'] = (mask * 255.0).astype(np.uint8)
        _FEATHER_MASK_CACHE['size'] = input_size
    return _FEATHER_MASK_CACHE['mask']


def enhance_face_onnx(
    frame: np.ndarray,
    face: Any,
    session: onnxruntime.InferenceSession,
    input_size: int,
) -> np.ndarray:
    """Enhance a single face in the frame using an ONNX face restoration model."""
    M, inv_M = _get_face_affine(face, input_size)
    if M is None:
        return frame

    face_crop = cv2.warpAffine(
        frame, M, (input_size, input_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    blob = preprocess_face(face_crop, input_size)
    with THREAD_SEMAPHORE:
        input_name = session.get_inputs()[0].name
        # DML sessions are not safe under concurrent run(); serialise GPU
        # work here so both GPEN variants inherit the guard. No caller of
        # enhance_face_onnx holds dml_lock, so no re-entrancy risk.
        if _is_dml():
            with modules.globals.dml_lock:
                output = run_inference(session, input_name, blob)
        else:
            output = run_inference(session, input_name, blob)
    enhanced = postprocess_face(output)

    # Feathered-edge blend mask, cached per input_size (uint8).
    mask = _get_feather_mask(input_size)

    h, w = frame.shape[:2]
    # Tight bbox of the aligned square in output coords — warp and blend
    # only the crop. The old path ran two full-frame warpAffines plus four
    # full-frame float32 passes per face (~6M px each at 1080p); the crop
    # is typically <5% of that.
    corners = np.array([[0, 0], [input_size, 0],
                        [input_size, input_size], [0, input_size]],
                       dtype=np.float32)
    transformed = (inv_M[:, :2] @ corners.T).T + inv_M[:, 2]
    x1 = max(0, int(np.floor(transformed[:, 0].min())))
    x2 = min(w, int(np.ceil(transformed[:, 0].max())))
    y1 = max(0, int(np.floor(transformed[:, 1].min())))
    y2 = min(h, int(np.ceil(transformed[:, 1].max())))
    if x1 >= x2 or y1 >= y2:
        return frame

    pad = max(1, input_size // 64) + 2
    y1p, y2p = max(0, y1 - pad), min(h, y2 + pad)
    x1p, x2p = max(0, x1 - pad), min(w, x2 + pad)
    crop_w, crop_h = x2p - x1p, y2p - y1p

    inv_crop = inv_M.copy()
    inv_crop[0, 2] -= x1p
    inv_crop[1, 2] -= y1p

    warped_enhanced = cv2.warpAffine(
        enhanced, inv_crop, (crop_w, crop_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask, inv_crop, (crop_w, crop_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    target_crop = frame[y1p:y2p, x1p:x2p]
    # Fused uint8 blend via cv2 SIMD — no float32 round-trip.
    alpha_3c = cv2.merge([warped_mask, warped_mask, warped_mask])
    inv_alpha = 255 - alpha_3c
    a_enh = cv2.multiply(warped_enhanced, alpha_3c, scale=1.0 / 255.0)
    a_tgt = cv2.multiply(target_crop, inv_alpha, scale=1.0 / 255.0)
    frame[y1p:y2p, x1p:x2p] = cv2.add(a_enh, a_tgt)
    return frame
