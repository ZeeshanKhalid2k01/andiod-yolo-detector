package com.tencent.yolov8ncnn;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.view.WindowManager;
import android.view.animation.AccelerateDecelerateInterpolator;
import android.widget.ImageView;
import android.widget.TextView;

public class SplashActivity extends Activity
{
    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);

        getWindow().setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        );

        setContentView(R.layout.activity_splash);

        ImageView iconEye = findViewById(R.id.iconEye);
        View glowView = findViewById(R.id.glowView);
        TextView textAppName = findViewById(R.id.textAppName);
        TextView textTagline = findViewById(R.id.textTagline);

        // Animate glow
        glowView.animate()
            .alpha(0.6f)
            .scaleX(1.3f)
            .scaleY(1.3f)
            .setDuration(1200)
            .setInterpolator(new AccelerateDecelerateInterpolator())
            .start();

        // Animate eye icon fade in + scale
        iconEye.animate()
            .alpha(1.0f)
            .scaleX(1.0f)
            .scaleY(1.0f)
            .setStartDelay(200)
            .setDuration(800)
            .setInterpolator(new AccelerateDecelerateInterpolator())
            .start();
        iconEye.setScaleX(0.5f);
        iconEye.setScaleY(0.5f);

        // App name fade in
        textAppName.animate()
            .alpha(1.0f)
            .translationY(0)
            .setStartDelay(600)
            .setDuration(600)
            .start();
        textAppName.setTranslationY(30);

        // Tagline fade in
        textTagline.animate()
            .alpha(1.0f)
            .translationY(0)
            .setStartDelay(900)
            .setDuration(600)
            .start();
        textTagline.setTranslationY(20);

        // Navigate to home after 2.5 seconds
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run()
            {
                Intent intent = new Intent(SplashActivity.this, HomeActivity.class);
                startActivity(intent);
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
                finish();
            }
        }, 2500);
    }
}
