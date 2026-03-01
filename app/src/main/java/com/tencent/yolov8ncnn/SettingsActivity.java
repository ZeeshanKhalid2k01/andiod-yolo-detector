package com.tencent.yolov8ncnn;

import android.app.Activity;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

public class SettingsActivity extends Activity
{
    private static final String PREFS_NAME = "FaceDetectSettings";

    private EditText editServerIp;
    private EditText editServerPort;
    private Switch switchAutoUpload;
    private Switch switchWifiOnly;
    private Spinner spinnerModelSize;
    private Spinner spinnerResolution;
    private SeekBar seekBarConfidence;
    private TextView textConfidenceValue;
    private Switch switchGpu;
    private Switch switchShowFps;
    private Switch switchShowConfidence;
    private Spinner spinnerBoxColor;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);

        // Bind views
        editServerIp = (EditText) findViewById(R.id.editServerIp);
        editServerPort = (EditText) findViewById(R.id.editServerPort);
        switchAutoUpload = (Switch) findViewById(R.id.switchAutoUpload);
        switchWifiOnly = (Switch) findViewById(R.id.switchWifiOnly);
        spinnerModelSize = (Spinner) findViewById(R.id.spinnerModelSize);
        spinnerResolution = (Spinner) findViewById(R.id.spinnerResolution);
        seekBarConfidence = (SeekBar) findViewById(R.id.seekBarConfidence);
        textConfidenceValue = (TextView) findViewById(R.id.textConfidenceValue);
        switchGpu = (Switch) findViewById(R.id.switchGpu);
        switchShowFps = (Switch) findViewById(R.id.switchShowFps);
        switchShowConfidence = (Switch) findViewById(R.id.switchShowConfidence);
        spinnerBoxColor = (Spinner) findViewById(R.id.spinnerBoxColor);
        ImageButton btnBack = (ImageButton) findViewById(R.id.btnBack);
        Button btnSave = (Button) findViewById(R.id.btnSave);
        TextView textSettingsTitle = (TextView) findViewById(R.id.textSettingsTitle);

        // Set up spinners — check if we're in face mode
        SharedPreferences initPrefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        boolean isFaceMode = (getIntent().getIntExtra("taskid", -1) == 6)
            || (initPrefs.getInt("taskid", 0) == 6);

        // Update toolbar title based on mode
        if (isFaceMode)
        {
            textSettingsTitle.setText("Face Detection Settings");
        }
        else
        {
            textSettingsTitle.setText("Object Detection Settings");
        }

        String[] modelSizes;
        if (isFaceMode)
        {
            modelSizes = new String[]{"YOLOv11n (Nano)", "YOLOv11s (Small)", "YOLOv11m (Medium)", "YOLOv11l (Large)"};
        }
        else
        {
            modelSizes = new String[]{"YOLOv8n (Nano)", "YOLOv8s (Small)", "YOLOv8m (Medium)"};
        }
        ArrayAdapter<String> modelAdapter = new ArrayAdapter<>(this,
            android.R.layout.simple_spinner_dropdown_item, modelSizes);
        spinnerModelSize.setAdapter(modelAdapter);

        String[] resolutions = {"320 \u00d7 320 (Fast)", "480 \u00d7 480 (Balanced)", "640 \u00d7 640 (Accurate)"};
        ArrayAdapter<String> resAdapter = new ArrayAdapter<>(this,
            android.R.layout.simple_spinner_dropdown_item, resolutions);
        spinnerResolution.setAdapter(resAdapter);

        String[] boxColors = {"Green", "Blue", "Red", "Yellow", "White"};
        ArrayAdapter<String> colorAdapter = new ArrayAdapter<>(this,
            android.R.layout.simple_spinner_dropdown_item, boxColors);
        spinnerBoxColor.setAdapter(colorAdapter);

        // Confidence slider
        seekBarConfidence.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser)
            {
                float value = progress / 100f;
                textConfidenceValue.setText(String.format("%.2f", value));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // Load saved preferences
        loadPreferences();

        // Back button
        btnBack.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                finish();
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
            }
        });

        // Save button
        btnSave.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                savePreferences();
                Toast.makeText(SettingsActivity.this, "Settings saved", Toast.LENGTH_SHORT).show();
                finish();
                overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
            }
        });
    }

    private void loadPreferences()
    {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);

        editServerIp.setText(prefs.getString("server_ip", ""));
        editServerPort.setText(prefs.getString("server_port", "8080"));
        switchAutoUpload.setChecked(prefs.getBoolean("auto_upload", false));
        switchWifiOnly.setChecked(prefs.getBoolean("wifi_only", true));

        spinnerModelSize.setSelection(prefs.getInt("model_size", 0));
        spinnerResolution.setSelection(prefs.getInt("resolution", 0));
        seekBarConfidence.setProgress(prefs.getInt("confidence", 45));
        switchGpu.setChecked(prefs.getBoolean("use_gpu", true));

        switchShowFps.setChecked(prefs.getBoolean("show_fps", true));
        switchShowConfidence.setChecked(prefs.getBoolean("show_labels", true));
        spinnerBoxColor.setSelection(prefs.getInt("box_color", 0));

        // Update confidence display
        float confValue = prefs.getInt("confidence", 45) / 100f;
        textConfidenceValue.setText(String.format("%.2f", confValue));
    }

    private void savePreferences()
    {
        SharedPreferences.Editor editor = getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit();

        editor.putString("server_ip", editServerIp.getText().toString().trim());
        editor.putString("server_port", editServerPort.getText().toString().trim());
        editor.putBoolean("auto_upload", switchAutoUpload.isChecked());
        editor.putBoolean("wifi_only", switchWifiOnly.isChecked());

        editor.putInt("model_size", spinnerModelSize.getSelectedItemPosition());
        editor.putInt("resolution", spinnerResolution.getSelectedItemPosition());
        editor.putInt("confidence", seekBarConfidence.getProgress());
        editor.putBoolean("use_gpu", switchGpu.isChecked());

        // Persist taskid so MainActivity can pick it up
        int taskid = getIntent().getIntExtra("taskid", -1);
        if (taskid >= 0)
        {
            editor.putInt("taskid", taskid);
        }

        editor.putBoolean("show_fps", switchShowFps.isChecked());
        editor.putBoolean("show_labels", switchShowConfidence.isChecked());
        editor.putInt("box_color", spinnerBoxColor.getSelectedItemPosition());

        editor.apply();
    }

    @Override
    public void onBackPressed()
    {
        super.onBackPressed();
        overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out);
    }
}
