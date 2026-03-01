// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.yolov8ncnn;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Bundle;
import android.os.Handler;
import android.provider.Settings;
import android.util.Log;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import android.graphics.Bitmap;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends Activity implements SurfaceHolder.Callback
{
    public static final int REQUEST_CAMERA = 100;
    private static final String PREFS_NAME = "FaceDetectSettings";

    private static MainActivity instance;

    private YOLOv8Ncnn yolov8ncnn = new YOLOv8Ncnn();
    private int facing = 0;
    private int captureW = 1920;
    private int captureH = 1080;

    private int current_taskid = 0;
    private int current_model = 0;
    private int current_cpugpu = 1;
    private boolean modelLoaded = false;
    private boolean surfaceReady = false;
    private int intentTaskId = -1; // -1 means not set by intent

    private SurfaceView cameraView;
    private TextView textFps;
    private View uploadStatusDot;
    private TextView textUploadStatus;
    private Button btnToggleUpload;
    private boolean uploadEnabled = false;

    private String serverIp = "";
    private String serverPort = "8080";
    private boolean wifiOnly = true;
    private String deviceId = "";

    // Upload pipeline
    private final RingBuffer ringBuffer = new RingBuffer(5);
    private final OkHttpClient httpClient = new OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .writeTimeout(10, TimeUnit.SECONDS)
            .readTimeout(5, TimeUnit.SECONDS)
            .build();
    private ExecutorService uploadExecutor;
    private volatile boolean uploaderRunning = false;

    private Handler fpsHandler = new Handler();
    private Runnable fpsRunnable;

    // Zoom
    private ScaleGestureDetector scaleGestureDetector;
    private float currentZoom = 1.0f;
    private float maxZoom = 10.0f; // updated from hardware after camera opens
    private TextView textZoomLevel;
    private final Handler zoomHideHandler = new Handler();
    private final Runnable zoomHideRunnable = () -> {
        if (textZoomLevel != null) textZoomLevel.setVisibility(View.GONE);
    };

    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detection);

        instance = this;

        // Read taskid from intent (e.g. 6 for face detection)
        intentTaskId = getIntent().getIntExtra("taskid", -1);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // Camera surface
        cameraView = (SurfaceView) findViewById(R.id.cameraView);
        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        // Top bar
        ImageButton btnBack = (ImageButton) findViewById(R.id.btnBack);
        textFps = (TextView) findViewById(R.id.textFps);

        // Bottom bar
        ImageButton btnSwitchCamera = (ImageButton) findViewById(R.id.btnSwitchCamera);
        uploadStatusDot = findViewById(R.id.uploadStatusDot);
        textUploadStatus = (TextView) findViewById(R.id.textUploadStatus);
        btnToggleUpload = (Button) findViewById(R.id.btnToggleUpload);

        // Load settings from SharedPreferences
        loadSettings();

        // Back button → always return to Home (never exit)
        btnBack.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                navigateToHome();
            }
        });

        // Switch camera
        btnSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                int new_facing = 1 - facing;
                yolov8ncnn.closeCamera();
                yolov8ncnn.setCaptureResolution(captureW, captureH);
                yolov8ncnn.openCamera(new_facing);
                facing  = new_facing;
                maxZoom = yolov8ncnn.getMaxZoom();
                currentZoom = 1.0f;
            }
        });

        // Upload toggle
        btnToggleUpload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                uploadEnabled = !uploadEnabled;
                updateUploadUI();
                Log.i("MainActivity", "Upload " + (uploadEnabled ? "enabled" : "disabled"));
            }
        });

        textZoomLevel = (TextView) findViewById(R.id.textZoomLevel);

        // Pinch-to-zoom
        scaleGestureDetector = new ScaleGestureDetector(this,
                new ScaleGestureDetector.SimpleOnScaleGestureListener() {
            @Override
            public boolean onScale(ScaleGestureDetector detector)
            {
                currentZoom *= detector.getScaleFactor();
                if (currentZoom < 1.0f)    currentZoom = 1.0f;
                if (currentZoom > maxZoom) currentZoom = maxZoom;
                yolov8ncnn.setZoom(currentZoom);

                // Show overlay "2.5× / 100×" so user knows the range
                if (textZoomLevel != null)
                {
                    textZoomLevel.setText(String.format(Locale.US,
                            "%.1f\u00d7 / %.0f\u00d7", currentZoom, maxZoom));
                    textZoomLevel.setVisibility(View.VISIBLE);
                }
                zoomHideHandler.removeCallbacks(zoomHideRunnable);
                zoomHideHandler.postDelayed(zoomHideRunnable, 2000);
                return true;
            }
        });

        cameraView.setOnTouchListener((v, event) -> {
            scaleGestureDetector.onTouchEvent(event);
            return true;
        });

        updateUploadUI();

        // Device identifier for upload metadata
        deviceId = Settings.Secure.getString(getContentResolver(), Settings.Secure.ANDROID_ID);

        // Start background upload worker
        uploaderRunning = true;
        uploadExecutor = Executors.newSingleThreadExecutor();
        uploadExecutor.submit(this::runUploadWorker);

        // Model will be loaded when surface is ready (surfaceChanged)

        // FPS polling runnable — updates every 500ms
        fpsRunnable = new Runnable() {
            @Override
            public void run()
            {
                float fps = yolov8ncnn.getFps();
                if (textFps != null && textFps.getVisibility() == View.VISIBLE)
                {
                    if (fps > 0.f)
                    {
                        textFps.setText(String.format(Locale.US, "%.0f FPS", fps));
                    }
                    else
                    {
                        textFps.setText("-- FPS");
                    }
                }
                fpsHandler.postDelayed(this, 500);
            }
        };
    }

    private void loadSettings()
    {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        serverIp = prefs.getString("server_ip", "");
        serverPort = prefs.getString("server_port", "8080");
        uploadEnabled = prefs.getBoolean("auto_upload", false);
        wifiOnly = prefs.getBoolean("wifi_only", true);

        // model_size: 0=nano, 1=small, 2=medium (3=large for face)
        // resolution: 0=320, 1=480, 2=640
        int modelSize = prefs.getInt("model_size", 0);
        int resolution = prefs.getInt("resolution", 0);

        // Clamp inputs
        if (modelSize < 0 || modelSize > 3) modelSize = 0;
        if (resolution < 0 || resolution > 2) resolution = 0;

        // Determine taskid: use intent extra if set, otherwise default to 0
        if (intentTaskId >= 0)
        {
            current_taskid = intentTaskId;
            // Save last used taskid so HomeActivity can read it for Settings
            prefs.edit().putInt("taskid", intentTaskId).apply();
        }
        else
        {
            current_taskid = 0; // default to coco detection
        }

        if (current_taskid == 6)
        {
            // Face mode: 4 model sizes (n/s/m/l), modelid = resolution * 4 + modelSize
            if (modelSize > 3) modelSize = 0;
            current_model = resolution * 4 + modelSize;
        }
        else
        {
            // Standard yolov8: 3 model sizes (n/s/m), modelid = resolution * 3 + modelSize
            if (modelSize > 2) modelSize = 0;
            current_model = resolution * 3 + modelSize;
        }
        current_cpugpu = prefs.getBoolean("use_gpu", true) ? 1 : 0;

        // Bounds checking — clamp to valid native ranges
        int maxModelId = (current_taskid == 6) ? 11 : 8;
        if (current_taskid < 0 || current_taskid > 6) current_taskid = 0;
        if (current_model < 0 || current_model > maxModelId) current_model = 0;
        if (current_cpugpu < 0 || current_cpugpu > 2) current_cpugpu = 1;

        // Apply box color from settings
        applyBoxColor(prefs.getInt("box_color", 0));

        // Apply confidence threshold from settings (saved as int 0-100, convert to 0.0-1.0)
        int confInt = prefs.getInt("confidence", 55);
        float confFloat = confInt / 100f;
        yolov8ncnn.setConfidenceThreshold(confFloat);
        Log.d("MainActivity", "Confidence threshold set to " + confFloat + " (from pref=" + confInt + ")");

        // Store capture resolution — passed to setCaptureResolution() before camera opens
        captureW = prefs.getInt("camera_capture_w", 1920);
        captureH = prefs.getInt("camera_capture_h", 1080);

        // Apply label visibility from settings
        yolov8ncnn.setShowLabels(prefs.getBoolean("show_labels", true));

        boolean showFps = prefs.getBoolean("show_fps", true);
        if (textFps != null)
        {
            textFps.setVisibility(showFps ? View.VISIBLE : View.GONE);
        }

        Log.d("MainActivity", "Settings loaded: taskid=" + current_taskid
            + " modelid=" + current_model + " cpugpu=" + current_cpugpu
            + " (modelSize=" + modelSize + " resolution=" + resolution + ")");
    }

    private void applyBoxColor(int colorIndex)
    {
        switch (colorIndex)
        {
            case 0: // Green
                yolov8ncnn.setBoxColor(0, 255, 100);
                break;
            case 1: // Blue
                yolov8ncnn.setBoxColor(50, 150, 255);
                break;
            case 2: // Red
                yolov8ncnn.setBoxColor(255, 50, 50);
                break;
            case 3: // Yellow
                yolov8ncnn.setBoxColor(255, 220, 0);
                break;
            case 4: // White
                yolov8ncnn.setBoxColor(255, 255, 255);
                break;
            default:
                yolov8ncnn.setBoxColor(0, 255, 100);
                break;
        }
        Log.d("MainActivity", "Box color set to index " + colorIndex);
    }

    private void updateUploadUI()
    {
        if (uploadEnabled)
        {
            uploadStatusDot.setBackgroundResource(R.drawable.status_dot_green);
            textUploadStatus.setText("Uploading");
            textUploadStatus.setTextColor(getResources().getColor(R.color.accent_green));
            btnToggleUpload.setText("Stop");
            btnToggleUpload.setBackgroundResource(R.drawable.btn_green_pill);
            btnToggleUpload.setTextColor(getResources().getColor(R.color.background_dark));
        }
        else
        {
            uploadStatusDot.setBackgroundResource(R.drawable.status_dot_grey);
            textUploadStatus.setText("Upload Off");
            textUploadStatus.setTextColor(getResources().getColor(R.color.text_secondary));
            btnToggleUpload.setText("Upload");
            btnToggleUpload.setBackgroundResource(R.drawable.btn_dark_pill);
            btnToggleUpload.setTextColor(getResources().getColor(R.color.text_secondary));
        }
    }

    private void reload()
    {
        // Bounds safety net
        int maxModelId = (current_taskid == 6) ? 11 : 8;
        if (current_taskid < 0 || current_taskid > 6) current_taskid = 0;
        if (current_model < 0 || current_model > maxModelId) current_model = 0;
        if (current_cpugpu < 0 || current_cpugpu > 2) current_cpugpu = 1;

        final int taskid = current_taskid;
        final int modelid = current_model;
        final int cpugpu = current_cpugpu;

        Log.d("MainActivity", "loadModel: taskid=" + taskid
            + " modelid=" + modelid + " cpugpu=" + cpugpu);

        // Load model off main thread to avoid jank
        new Thread(new Runnable() {
            @Override
            public void run()
            {
                boolean ret_init = yolov8ncnn.loadModel(getAssets(), taskid, modelid, cpugpu);
                if (!ret_init)
                {
                    Log.e("MainActivity", "yolov8ncnn loadModel failed");
                    modelLoaded = false;
                }
                else
                {
                    Log.d("MainActivity", "yolov8ncnn loadModel success");
                    modelLoaded = true;
                }
            }
        }).start();
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
    {
        yolov8ncnn.setOutputWindow(holder.getSurface());

        // Load model and open camera now that surface is ready
        if (!surfaceReady)
        {
            surfaceReady = true;
            reload();
            new Handler().postDelayed(new Runnable() {
                @Override
                public void run()
                {
                    yolov8ncnn.setCaptureResolution(captureW, captureH);
                    yolov8ncnn.openCamera(facing);
                    maxZoom = yolov8ncnn.getMaxZoom();
                    Log.d("MainActivity", "maxZoom=" + maxZoom + " facing=" + facing);
                }
            }, 300);
        }

        // Start FPS polling
        fpsHandler.removeCallbacks(fpsRunnable);
        fpsHandler.postDelayed(fpsRunnable, 500);
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
        surfaceReady = false;
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED)
        {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        // Reload settings in case user changed them
        loadSettings();
        updateUploadUI();

        // Register JNI callback so C++ can push face frames to us
        yolov8ncnn.registerCallback(this);

        // Restart FPS polling
        fpsHandler.removeCallbacks(fpsRunnable);
        fpsHandler.post(fpsRunnable);

        // If surface is already ready (returning from settings/background), reload and reopen
        if (surfaceReady)
        {
            reload();
            new Handler().postDelayed(new Runnable() {
                @Override
                public void run()
                {
                    yolov8ncnn.setCaptureResolution(captureW, captureH);
                    yolov8ncnn.openCamera(facing);
                    maxZoom = yolov8ncnn.getMaxZoom();
                }
            }, 300);
        }
        // Otherwise, surfaceChanged will handle it when the surface becomes available
    }

    @Override
    public void onPause()
    {
        super.onPause();
        fpsHandler.removeCallbacks(fpsRunnable);
        yolov8ncnn.closeCamera();
        // Clear JNI callback ref so C++ doesn't call into a paused activity
        yolov8ncnn.registerCallback(null);
    }

    @Override
    public void onDestroy()
    {
        super.onDestroy();
        uploaderRunning = false;
        if (uploadExecutor != null)
        {
            uploadExecutor.shutdownNow();
        }
    }

    @Override
    public void onBackPressed()
    {
        navigateToHome();
    }

    private void navigateToHome()
    {
        android.content.Intent intent = new android.content.Intent(this, HomeActivity.class);
        intent.addFlags(android.content.Intent.FLAG_ACTIVITY_CLEAR_TOP
                | android.content.Intent.FLAG_ACTIVITY_SINGLE_TOP);
        startActivity(intent);
        finish();
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
    }

    // Static accessors for JNI / upload service
    public static String getServerIp()
    {
        if (instance != null) return instance.serverIp;
        return "";
    }

    public static String getServerPort()
    {
        if (instance != null) return instance.serverPort;
        return "8080";
    }

    public static boolean isUploadEnabled()
    {
        return instance != null && instance.uploadEnabled;
    }

    // ── JNI callback — called from camera thread for each qualifying face ──────
    // Receives raw RGB bytes; JPEG encoding deferred to upload worker to avoid
    // blocking the camera render thread on every frame.

    public void onFaceReady(byte[] rgbBytes, int width, int height,
                            float x, float y, float w, float h, float conf)
    {
        ringBuffer.push(new FramePacket(rgbBytes, width, height, x, y, w, h, conf));
    }

    // ── Upload worker — runs on single background thread ──────────────────────

    private void runUploadWorker()
    {
        long lastUploadTime = 0;
        while (uploaderRunning)
        {
            FramePacket packet = ringBuffer.poll();
            if (packet == null)
            {
                try { Thread.sleep(50); } catch (InterruptedException e) { break; }
                continue;
            }

            if (!uploadEnabled) continue;

            long now = System.currentTimeMillis();
            if (now - lastUploadTime < 1000) continue;  // rate limit: 1 upload/sec

            if (wifiOnly && !isOnWifi())
            {
                updateStatusDot(false);
                continue;
            }

            boolean ok = uploadPacket(packet);
            if (ok) lastUploadTime = now;
            updateStatusDot(ok);
        }
    }

    private byte[] encodeJpeg(byte[] rgb, int width, int height)
    {
        // Convert raw RGB bytes to ARGB int[] then compress via Android Bitmap
        int[] pixels = new int[width * height];
        for (int i = 0; i < pixels.length; i++)
        {
            int r = rgb[i * 3]     & 0xFF;
            int g = rgb[i * 3 + 1] & 0xFF;
            int b = rgb[i * 3 + 2] & 0xFF;
            pixels[i] = (0xFF << 24) | (r << 16) | (g << 8) | b;
        }
        Bitmap bmp = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, 85, baos);
        bmp.recycle();
        return baos.toByteArray();
    }

    private boolean uploadPacket(FramePacket p)
    {
        if (serverIp == null || serverIp.isEmpty()) return false;
        String url = "http://" + serverIp + ":" + serverPort + "/detect";
        try
        {
            byte[] jpeg = encodeJpeg(p.rgb, p.width, p.height);

            RequestBody body = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("image", "face.jpg",
                            RequestBody.create(jpeg, MediaType.parse("image/jpeg")))
                    .addFormDataPart("box_x",      String.valueOf(p.x))
                    .addFormDataPart("box_y",      String.valueOf(p.y))
                    .addFormDataPart("box_w",      String.valueOf(p.w))
                    .addFormDataPart("box_h",      String.valueOf(p.h))
                    .addFormDataPart("confidence", String.valueOf(p.conf))
                    .addFormDataPart("timestamp",  String.valueOf(p.timestamp))
                    .addFormDataPart("device_id",  deviceId)
                    .build();

            Request request = new Request.Builder().url(url).post(body).build();
            try (Response response = httpClient.newCall(request).execute())
            {
                boolean ok = response.isSuccessful();
                Log.d("MainActivity", "Upload " + (ok ? "OK" : "FAIL")
                        + " HTTP " + response.code() + " → " + url);
                return ok;
            }
        }
        catch (Exception e)
        {
            Log.e("MainActivity", "Upload error: " + e.getMessage());
            return false;
        }
    }

    @SuppressWarnings("deprecation")
    private boolean isOnWifi()
    {
        ConnectivityManager cm = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        if (cm == null) return false;
        NetworkInfo ni = cm.getActiveNetworkInfo();
        return ni != null && ni.isConnected() && ni.getType() == ConnectivityManager.TYPE_WIFI;
    }

    private void updateStatusDot(boolean success)
    {
        runOnUiThread(() -> {
            if (uploadStatusDot != null)
            {
                uploadStatusDot.setBackgroundResource(
                        success ? R.drawable.status_dot_green : R.drawable.status_dot_red);
            }
        });
    }

    // ── Inner classes ─────────────────────────────────────────────────────────

    static class FramePacket
    {
        final byte[] rgb;        // raw RGB bytes (3 bytes/pixel, no padding)
        final int    width;
        final int    height;
        final float  x, y, w, h, conf;
        final long   timestamp;

        FramePacket(byte[] rgb, int width, int height,
                    float x, float y, float w, float h, float conf)
        {
            this.rgb       = rgb;
            this.width     = width;
            this.height    = height;
            this.x         = x;
            this.y         = y;
            this.w         = w;
            this.h         = h;
            this.conf      = conf;
            this.timestamp = System.currentTimeMillis();
        }
    }

    static class RingBuffer
    {
        private final FramePacket[] buf;
        private int readIdx  = 0;
        private int writeIdx = 0;
        private int size     = 0;
        private final int capacity;

        RingBuffer(int capacity)
        {
            this.capacity = capacity;
            this.buf      = new FramePacket[capacity];
        }

        synchronized void push(FramePacket p)
        {
            buf[writeIdx] = p;
            writeIdx = (writeIdx + 1) % capacity;
            if (size < capacity)
            {
                size++;
            }
            else
            {
                // Buffer full — overwrite oldest, advance read pointer
                readIdx = (readIdx + 1) % capacity;
            }
        }

        synchronized FramePacket poll()
        {
            if (size == 0) return null;
            FramePacket p = buf[readIdx];
            readIdx = (readIdx + 1) % capacity;
            size--;
            return p;
        }
    }
}
