import os
import sys


def _configure_thread_env() -> None:
    """Set thread-pool env vars before torch/onnxruntime import.

    Maxed-out defaults for a 12-core Intel server CPU + DirectML GPU:

    - OMP/MKL/OpenBLAS threads = logical CPUs minus 2 (leaves room for
      FFmpeg + UI while feeding DML CPU-fallback ops generously).
    - OMP_WAIT_POLICY=ACTIVE + GOMP_CPU_AFFINITY off: reduces wake-up
      latency for per-frame inference (~ms scale matters here).
    - Only sets values the user hasn't already provided.
    """
    cpu = os.cpu_count() or 12
    omp_threads = str(max(4, min(cpu - 2, 22)))
    os.environ.setdefault('OMP_NUM_THREADS', omp_threads)
    os.environ.setdefault('MKL_NUM_THREADS', omp_threads)
    os.environ.setdefault('OPENBLAS_NUM_THREADS', omp_threads)
    os.environ.setdefault('OMP_WAIT_POLICY', 'ACTIVE')
    os.environ.setdefault('OMP_DYNAMIC', 'FALSE')
    # Allow KMP (Intel MKL) to spin briefly instead of sleeping — lower
    # per-inference latency on server CPUs.
    os.environ.setdefault('KMP_BLOCKTIME', '0')
    os.environ.setdefault('KMP_AFFINITY', 'granularity=fine,compact,1,0')


_configure_thread_env()
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
import onnxruntime
try:
    import tensorflow
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules, process_video_in_memory
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

if HAS_TORCH and 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
if HAS_TORCH:
    warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor', default=['face_swapper'], choices=['face_swapper', 'face_enhancer', 'face_enhancer_gpen256', 'face_enhancer_gpen512'], nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--nsfw-filter', help='filter the NSFW image or video', dest='nsfw_filter', action='store_true', default=False)
    program.add_argument('--map-faces', help='map source target faces', dest='map_faces', action='store_true', default=False)
    program.add_argument('--mouth-mask', help='mask the mouth region', dest='mouth_mask', action='store_true', default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('--output-resolution', help='output video size, e.g. 1920x1080, 720p, 4k, or source to keep native size', dest='output_resolution', default='1920x1080')
    program.add_argument('-l', '--lang', help='Ui language', default="en")
    program.add_argument('--live-mirror', help='The live camera display as you see it in the front-facing camera frame', dest='live_mirror', action='store_true', default=False)
    program.add_argument('--live-resizable', help='The live camera frame is resizable', dest='live_resizable', action='store_true', default=False)
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=[suggest_default_execution_provider()], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=None)
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')

    # register deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path, args.output_path)
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = args.source_path or args.target_path or args.output_path
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.map_faces = args.map_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.output_resolution = args.output_resolution
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads
    modules.globals.lang = args.lang

    # The argparse default (None) avoids evaluating suggest_execution_threads()
    # before providers are decoded, and deprecated-arg overrides above may
    # have already set execution_threads.
    if modules.globals.execution_threads is None:
        modules.globals.execution_threads = suggest_execution_threads()

    #for ENHANCER tumblers:
    for enhancer_key in ('face_enhancer', 'face_enhancer_gpen256', 'face_enhancer_gpen512'):
        modules.globals.fp_ui[enhancer_key] = enhancer_key in args.frame_processor

    # translate deprecated args
    if args.source_path_deprecated:
        print('\033[33mArgument -f and --face are deprecated. Use -s and --source instead.\033[0m')
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(args.source_path_deprecated, modules.globals.target_path, args.output_path)
    if args.cpu_cores_deprecated:
        print('\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == 'apple':
        print('\033[33mArgument --gpu-vendor apple is deprecated. Use --execution-provider coreml instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['coreml'])
    if args.gpu_vendor_deprecated == 'nvidia':
        print('\033[33mArgument --gpu-vendor nvidia is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['cuda'])
    if args.gpu_vendor_deprecated == 'amd':
        print('\033[33mArgument --gpu-vendor amd is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['rocm'])
    if args.gpu_threads_deprecated:
        print('\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.gpu_threads_deprecated


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    available = onnxruntime.get_available_providers()
    decoded = [provider for provider, encoded_execution_provider in zip(available, encode_execution_providers(available))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]
    # Always keep a CPU fallback after a GPU provider (DML/CUDA/ROCM/
    # OpenVINO/CoreML). Without it, ops lacking a DirectML kernel fail
    # instead of falling back — fatal on Polaris/RX580.
    if decoded and decoded[0] != 'CPUExecutionProvider' and 'CPUExecutionProvider' in available:
        if 'CPUExecutionProvider' not in decoded:
            decoded.append('CPUExecutionProvider')
    return decoded


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    # Max out for 32GB-class machines: leave ~4GB for OS/VRAM staging.
    try:
        import psutil

        total_gb = psutil.virtual_memory().total // (1024 ** 3)
        if total_gb >= 8:
            return int(max(8, min(total_gb - 4, 28)))
    except Exception:
        pass
    return 16


def suggest_default_execution_provider() -> str:
    """Pick the best available provider: cuda > rocm > coreml > openvino > dml > cpu."""
    available = encode_execution_providers(onnxruntime.get_available_providers())
    for pref in ('cuda', 'rocm', 'coreml', 'openvino', 'dml'):
        if pref in available:
            return pref
    return 'cpu'


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    """Suggest optimal worker count based on hardware and execution provider.

    Maxed out for a 12-core Intel server CPU + 8GB DirectML GPU (RX580):

    - DML serialises GPU inference internally (plus ``dml_lock``), so more
      workers do NOT parallelise inference. But 4-6 workers DO parallelise
      the CPU-side work around it (imread/imwrite, resize, affine warps,
      blending) while one worker holds the GPU. 1 worker (old default)
      starved a 12-core CPU.
    - CPU fallback ops inside DML sessions use intra_op threads (see
      ``get_session_options``), which is separate from this worker count.
    """
    import os

    # Get CPU count
    cpu_count = os.cpu_count() or 12

    if 'DmlExecutionProvider' in modules.globals.execution_providers:
        # 12C/24T -> 6 workers; 12C/12T -> 4 workers; cap at 6 so the 8GB
        # RX580 is never fed more concurrent frames than it can stage.
        return int(max(4, min(6, (cpu_count // 4) or 4)))
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        return 2
    if 'OpenVINOExecutionProvider' in modules.globals.execution_providers:
        return 1

    # For CPU execution, use most cores but leave some for system
    return max(4, min(cpu_count - 2, 16))


def limit_resources() -> None:
    # prevent tensorflow memory leak
    if HAS_TENSORFLOW:
        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
    # limit memory usage
    if modules.globals.max_memory:
        # setrlimit(RLIMIT_DATA) fails with EINVAL on macOS, crashing on launch.
        # See https://github.com/hacksider/Deep-Live-Cam/issues/1848
        if platform.system().lower() == 'darwin':
            return
        memory = modules.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def apply_runtime_tuning() -> None:
    """Apply maxed-out CPU-side tuning after providers/threads are known.

    - OpenCV uses its own thread pool (imread/imwrite/resize/warpAffine
      dominate per-frame CPU time). Default is often capped low; raise it
      to the worker count on DML, or full CPU count otherwise.
    - Torch (used only for optional CUDA blending) gets matching threads.
    """
    cpu = os.cpu_count() or 12
    try:
        import cv2

        if 'DmlExecutionProvider' in modules.globals.execution_providers:
            cv2.setNumThreads(max(4, min(cpu, 16)))
        else:
            cv2.setNumThreads(max(4, min(cpu, 32)))
    except Exception:
        pass
    if HAS_TORCH:
        try:
            import torch

            torch.set_num_threads(max(4, min(cpu - 2, 16)))
            torch.set_num_interop_threads(1)
        except Exception:
            pass


def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.globals.execution_providers and HAS_TORCH:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    print(f'[{scope}] {message}')
    if not modules.globals.headless:
        ui.update_status(message)

def start() -> None:
    """Start processing with performance monitoring."""
    import time
    
    start_time = time.time()
    
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    update_status('Processing...')
    
    # process image to image
    if has_image_extension(modules.globals.target_path):
        if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
            return
        try:
            shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        except Exception as e:
            print("Error copying file:", str(e))
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(modules.globals.source_path, modules.globals.output_path, modules.globals.output_path)
            release_resources()
        if is_image(modules.globals.target_path):
            elapsed = time.time() - start_time
            update_status(f'Processing to image succeed! (Time: {elapsed:.2f}s)')
        else:
            update_status('Processing to image failed!')
        return
    
    # process image to videos
    if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
        return

    # Detect FPS early (needed by both pipelines)
    if modules.globals.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(modules.globals.target_path)
    else:
        fps = 30.0

    video_created = False

    # --- In-memory pipeline (non-map_faces only) ---
    # Reads frames from FFmpeg pipe, processes in memory, encodes directly.
    # Eliminates all per-frame PNG disk I/O for a major speed-up.
    if not modules.globals.map_faces:
        update_status(f'Processing video in-memory at {fps} fps...')
        create_temp(modules.globals.target_path)

        processing_start = time.time()
        video_created = process_video_in_memory(
            modules.globals.source_path,
            modules.globals.target_path,
            fps,
        )
        processing_time = time.time() - processing_start
        release_resources()

        if video_created:
            update_status(f'In-memory processing + encoding completed in {processing_time:.2f}s')

    # --- Disk-based fallback (required for map_faces, or if pipe failed) ---
    if not video_created:
        if not modules.globals.map_faces:
            update_status('Falling back to disk-based processing...')

        extraction_start = time.time()
        create_temp(modules.globals.target_path)
        update_status('Extracting frames...')
        extract_frames(modules.globals.target_path)
        extraction_time = time.time() - extraction_start

        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
        total_frames = len(temp_frame_paths)
        update_status(f'Processing {total_frames} frames with {modules.globals.execution_threads} threads...')

        processing_start = time.time()
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_video(modules.globals.source_path, temp_frame_paths)
            release_resources()
        processing_time = time.time() - processing_start
        fps_processing = total_frames / processing_time if processing_time > 0 else 0
        update_status(f'Frame processing completed in {processing_time:.2f}s ({fps_processing:.2f} fps)')

        encoding_start = time.time()
        update_status(f'Creating video with {fps} fps...')
        video_created = create_video(modules.globals.target_path, fps)
        encoding_time = time.time() - encoding_start
        if video_created:
            update_status(f'Video encoding completed in {encoding_time:.2f}s')

    if not video_created:
        update_status('Video encoding failed. No temporary output video was created.')
        clean_temp(modules.globals.target_path)
        return
    
    # handle audio
    if modules.globals.keep_audio:
        if modules.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(modules.globals.target_path, modules.globals.output_path)
    else:
        move_temp(modules.globals.target_path, modules.globals.output_path)
    
    # clean and validate
    clean_temp(modules.globals.target_path)
    
    total_time = time.time() - start_time
    if is_video(modules.globals.target_path) and modules.globals.output_path and os.path.isfile(modules.globals.output_path):
        update_status(f'Video processing succeeded! Total time: {total_time:.2f}s')
    else:
        update_status('Processing to video failed!')


def destroy(to_quit=True) -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    if to_quit:
        quit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    apply_runtime_tuning()
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    # Pre-load face analyser in main thread before GUI starts
    #from modules.face_analyser import get_face_analyser
    #get_face_analyser()
    limit_resources()
    if modules.globals.headless:
        start()
    else:
        window = ui.init(start, destroy, modules.globals.lang)
        window.mainloop()