# Android YOLOv11 Face Detection (ncnn + Vulkan)

Real-time face detection on Android using YOLOv11 face models, running fully on-device with ncnn inference and Mesa Turnip Vulkan GPU acceleration.

## What It Does

- Runs YOLOv11 face detection models entirely on-device — no cloud, no server required
- Uses ncnn for optimized mobile inference with Vulkan GPU backend (Mesa Turnip)
- Supports multiple model sizes (n/s/m/l) and input resolutions (320/480/640px)
- Configurable confidence threshold, NMS tuning, and camera resolution via in-app settings
- Optional server IP/port settings for a future face-upload feature

## Device Tested

Samsung Galaxy S23 Ultra

## Tech Stack

| Component | Version |
|-----------|---------|
| Android Studio | Latest |
| Android SDK | 34 |
| Android NDK | 29 |
| ncnn | 20260113-android-vulkan |
| OpenCV Mobile | 4.13.0-android |
| Vulkan backend | Mesa Turnip |
| Language | C++ (JNI), Java |

## How to Build

1. **Clone the repo**
   ```bash
   git clone https://github.com/ZeeshanKhalid2k01/andiod-yolo-detector.git
   cd andiod-yolo-detector
   ```

2. **Download the prebuilt SDKs** (not included in repo — too large)
   - [ncnn-20260113-android-vulkan](https://github.com/Tencent/ncnn/releases)
   - [opencv-mobile-4.13.0-android](https://github.com/nihui/opencv-mobile/releases)

3. **Place SDKs in the correct location**
   ```
   app/src/main/jni/ncnn-20260113-android-vulkan/
   app/src/main/jni/opencv-mobile-4.13.0-android/
   ```

4. **Add model files to assets**
   Place your `.ncnn.bin` and `.ncnn.param` model files in:
   ```
   app/src/main/assets/
   ```
   See the Models section below for expected filenames.

5. **Open in Android Studio** and build/run on device.

## Models

Place YOLOv11 face model files in `app/src/main/assets/`. Expected naming pattern:

| Model | Files |
|-------|-------|
| YOLOv11n face | `yolov11n_face.ncnn.bin`, `yolov11n_face.ncnn.param` |
| YOLOv11s face | `yolov11s_face.ncnn.bin`, `yolov11s_face.ncnn.param` |
| YOLOv11m face | `yolov11m_face.ncnn.bin`, `yolov11m_face.ncnn.param` |
| YOLOv11l face | `yolov11l_face.ncnn.bin`, `yolov11l_face.ncnn.param` |

Input resolutions: **320 × 320**, **480 × 480**, **640 × 640** (selectable in settings).

> Model `.bin` and `.param` files are excluded from this repo due to size. Source them from your training pipeline or a model hub.

## Key Fixes & Tuning

Several issues were resolved to get accurate face detection working:

| Fix | Details |
|-----|---------|
| **Square input** | Forced 320×320 letterboxed input — non-square caused misaligned anchors |
| **Format-B tensor decoding** | YOLOv11 face models use format-B output layout; fixed decoder to match |
| **NMS threshold tuning** | Adjusted NMS to suppress overlapping phantom boxes |
| **prob_threshold = 0.55** | Raised confidence cutoff to eliminate low-confidence false positives |
| **Bounds filter** | Added filter to discard anchor boxes outside valid image bounds |

## In-App Settings

| Setting | Description |
|---------|-------------|
| Confidence threshold | Minimum detection score (default 0.55) |
| Model size | n / s / m / l |
| Input resolution | 320 / 480 / 640 px |
| GPU toggle | Enable/disable Vulkan GPU acceleration |
| Server IP / Port | Target endpoint for future face-upload feature |

## Notes

- The `ncnn` and `opencv-mobile` SDK directories are excluded from this repo via `.gitignore`. Download them separately as described above.
- Model `.bin`/`.param` files are also excluded from the repo due to size.
