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
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;

import java.util.Locale;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class MainActivity extends Activity implements SurfaceHolder.Callback
{
    public static final int REQUEST_CAMERA = 100;
    private static final String PREFS_NAME = "FaceDetectSettings";

    private static MainActivity instance;

    private YOLOv8Ncnn yolov8ncnn = new YOLOv8Ncnn();
    private int facing = 0;

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

    private Handler fpsHandler = new Handler();
    private Runnable fpsRunnable;

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

        // Back button → finish
        btnBack.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                finish();
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
            }
        });

        // Switch camera
        btnSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                int new_facing = 1 - facing;
                yolov8ncnn.closeCamera();
                yolov8ncnn.openCamera(new_facing);
                facing = new_facing;
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

        updateUploadUI();
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
                    yolov8ncnn.openCamera(facing);
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
                    yolov8ncnn.openCamera(facing);
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
    }

    @Override
    public void onBackPressed()
    {
        super.onBackPressed();
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
    }

    // Static accessors for JNI / upload service
    public static String getServerIp()
    {
        if (instance != null)
        {
            return instance.serverIp;
        }
        return "";
    }

    public static String getServerPort()
    {
        if (instance != null)
        {
            return instance.serverPort;
        }
        return "8080";
    }

    public static boolean isUploadEnabled()
    {
        return instance != null && instance.uploadEnabled;
    }
}
