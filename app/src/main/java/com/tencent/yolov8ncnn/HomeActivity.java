package com.tencent.yolov8ncnn;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.animation.AccelerateDecelerateInterpolator;
import android.widget.ImageButton;
import android.widget.Toast;

public class HomeActivity extends Activity
{
    private static final String PREFS_NAME = "FaceDetectSettings";

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);

        View cardObjectDetection = findViewById(R.id.cardObjectDetection);
        View cardFaceDetection = findViewById(R.id.cardFaceDetection);
        View cardAnalytics = findViewById(R.id.cardAnalytics);
        View cardSettings = findViewById(R.id.cardSettings);
        ImageButton btnTopSettings = findViewById(R.id.btnTopSettings);

        // Press animation helper
        View.OnTouchListener pressAnimator = new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event)
            {
                switch (event.getAction())
                {
                    case MotionEvent.ACTION_DOWN:
                        v.animate().scaleX(0.95f).scaleY(0.95f)
                            .setDuration(100)
                            .setInterpolator(new AccelerateDecelerateInterpolator())
                            .start();
                        break;
                    case MotionEvent.ACTION_UP:
                    case MotionEvent.ACTION_CANCEL:
                        v.animate().scaleX(1.0f).scaleY(1.0f)
                            .setDuration(100)
                            .setInterpolator(new AccelerateDecelerateInterpolator())
                            .start();
                        break;
                }
                return false;
            }
        };

        cardObjectDetection.setOnTouchListener(pressAnimator);
        cardFaceDetection.setOnTouchListener(pressAnimator);
        cardSettings.setOnTouchListener(pressAnimator);

        // Card 1 — Object Detection → open camera with taskid=0
        cardObjectDetection.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                try
                {
                    Intent intent = new Intent(HomeActivity.this, MainActivity.class);
                    intent.putExtra("taskid", 0);
                    startActivity(intent);
                    overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
                }
                catch (Exception e)
                {
                    Log.e("HomeActivity", "Error launching object detection", e);
                    Toast.makeText(HomeActivity.this, "Failed to open detection: " + e.getMessage(), Toast.LENGTH_LONG).show();
                }
            }
        });

        // Card 2 — Face Detection → open camera with taskid=6
        cardFaceDetection.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                try
                {
                    Intent intent = new Intent(HomeActivity.this, MainActivity.class);
                    intent.putExtra("taskid", 6);
                    startActivity(intent);
                    overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
                }
                catch (Exception e)
                {
                    Log.e("HomeActivity", "Error launching face detection", e);
                    Toast.makeText(HomeActivity.this, "Failed to open detection: " + e.getMessage(), Toast.LENGTH_LONG).show();
                }
            }
        });

        // Card — Analytics (disabled)
        cardAnalytics.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                Toast.makeText(HomeActivity.this, "Analytics coming soon", Toast.LENGTH_SHORT).show();
            }
        });

        // Card — Settings
        cardSettings.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                launchSettings();
            }
        });

        // Top bar gear icon
        btnTopSettings.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                launchSettings();
            }
        });

        // Entrance animations
        cardObjectDetection.setAlpha(0f);
        cardObjectDetection.setTranslationY(40);
        cardObjectDetection.animate().alpha(1f).translationY(0).setStartDelay(100).setDuration(400).start();

        cardFaceDetection.setAlpha(0f);
        cardFaceDetection.setTranslationY(40);
        cardFaceDetection.animate().alpha(1f).translationY(0).setStartDelay(200).setDuration(400).start();

        cardAnalytics.setAlpha(0f);
        cardAnalytics.setTranslationY(40);
        cardAnalytics.animate().alpha(0.5f).translationY(0).setStartDelay(300).setDuration(400).start();

        cardSettings.setAlpha(0f);
        cardSettings.setTranslationY(40);
        cardSettings.animate().alpha(1f).translationY(0).setStartDelay(400).setDuration(400).start();
    }

    private void launchSettings()
    {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        int savedTaskId = prefs.getInt("taskid", 0);
        Intent intent = new Intent(HomeActivity.this, SettingsActivity.class);
        intent.putExtra("taskid", savedTaskId);
        startActivity(intent);
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
    }
}
