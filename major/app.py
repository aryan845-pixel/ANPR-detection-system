import os
import io
import time
import base64
from flask import Flask, request, jsonify, render_template_string, send_file
from ocr_service import OCRService

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VisionOCR AI - Advanced EasyOCR Image & License Plate Reader</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg-dark: #090d16;
            --bg-card: rgba(17, 24, 39, 0.75);
            --border-glass: rgba(255, 255, 255, 0.08);
            --border-active: rgba(56, 189, 248, 0.5);
            --accent-cyan: #38bdf8;
            --accent-purple: #a855f7;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --gradient-primary: linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #a855f7 100%);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-dim: #64748b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.08) 0%, transparent 40%);
            background-attachment: fixed;
        }

        /* HEADER */
        header {
            background: rgba(11, 15, 25, 0.85);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border-glass);
            padding: 16px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 1000;
        }

        .brand-container {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-logo {
            width: 42px;
            height: 42px;
            background: var(--gradient-primary);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            color: #fff;
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.35);
        }

        .brand-title {
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #ffffff 30%, var(--accent-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-subtitle {
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 400;
        }

        .nav-tabs {
            display: flex;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-glass);
            padding: 4px;
            border-radius: 12px;
            gap: 4px;
        }

        .nav-btn {
            padding: 8px 18px;
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-size: 13px;
            font-weight: 500;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .nav-btn:hover {
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.05);
        }

        .nav-btn.active {
            background: var(--gradient-primary);
            color: #fff;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
        }

        .status-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 6px 14px;
            border-radius: 30px;
            font-size: 13px;
            color: var(--accent-emerald);
            font-weight: 500;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: var(--accent-emerald);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--accent-emerald);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        /* MAIN CONTAINER */
        main {
            flex: 1;
            max-width: 1400px;
            width: 100%;
            margin: 0 auto;
            padding: 28px 24px;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        /* TOOLBAR / CONTROLS */
        .toolbar {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-glass);
            border-radius: 16px;
            padding: 16px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            flex-wrap: wrap;
        }

        .control-group {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }

        .control-label {
            font-size: 13px;
            font-weight: 500;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        select, input[type="text"] {
            background: rgba(9, 13, 22, 0.8);
            color: var(--text-main);
            border: 1px solid var(--border-glass);
            padding: 9px 16px;
            border-radius: 10px;
            font-size: 13px;
            outline: none;
            transition: all 0.2s;
        }

        select:focus, input[type="text"]:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
        }

        .preset-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-glass);
            color: var(--text-main);
            padding: 8px 16px;
            border-radius: 10px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .preset-btn:hover {
            background: rgba(56, 189, 248, 0.15);
            border-color: var(--accent-cyan);
            color: #fff;
            transform: translateY(-1px);
        }

        /* GRID LAYOUT */
        .workspace-grid {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 24px;
        }

        @media (max-width: 1024px) {
            .workspace-grid {
                grid-template-columns: 1fr;
            }
        }

        /* CARDS */
        .glass-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-glass);
            border-radius: 20px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 18px;
            min-height: 540px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border-glass);
            padding-bottom: 14px;
        }

        .card-title {
            font-size: 16px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-main);
        }

        .card-title i {
            color: var(--accent-cyan);
        }

        /* DROPZONE */
        .dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            padding: 40px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(9, 13, 22, 0.5);
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 16px;
            position: relative;
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--accent-cyan);
            background: rgba(56, 189, 248, 0.05);
            box-shadow: 0 0 30px rgba(56, 189, 248, 0.15);
        }

        .drop-icon {
            width: 64px;
            height: 64px;
            background: rgba(56, 189, 248, 0.1);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 26px;
            color: var(--accent-cyan);
            transition: transform 0.3s ease;
        }

        .dropzone:hover .drop-icon {
            transform: scale(1.1) translateY(-4px);
            background: rgba(56, 189, 248, 0.2);
        }

        /* IMAGE CANVAS PREVIEW */
        .preview-container {
            position: relative;
            flex: 1;
            display: none;
            align-items: center;
            justify-content: center;
            background: rgba(9, 13, 22, 0.7);
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid var(--border-glass);
            min-height: 360px;
        }

        .preview-img {
            max-width: 100%;
            max-height: 420px;
            object-fit: contain;
            border-radius: 8px;
        }

        /* BUTTONS */
        .btn-primary {
            background: var(--gradient-primary);
            color: #fff;
            border: none;
            padding: 12px 28px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.25s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            box-shadow: 0 4px 16px rgba(6, 182, 212, 0.35);
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(6, 182, 212, 0.5);
            filter: brightness(1.1);
        }

        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border-glass);
            color: var(--text-main);
            padding: 8px 16px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(255, 255, 255, 0.2);
        }

        /* METRICS BADGES */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
        }

        .metric-card {
            background: rgba(9, 13, 22, 0.6);
            border: 1px solid var(--border-glass);
            padding: 10px 14px;
            border-radius: 12px;
            text-align: center;
        }

        .metric-val {
            font-size: 18px;
            font-weight: 700;
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
        }

        .metric-lbl {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        /* EXTRACTED TEXT DISPLAY */
        .text-output-box {
            background: #090d16;
            border: 1px solid var(--border-glass);
            border-radius: 12px;
            padding: 18px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 15px;
            color: #38bdf8;
            line-height: 1.6;
            flex: 1;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.5);
            min-height: 140px;
        }

        /* BREAKDOWN TABLE */
        .details-table-container {
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid var(--border-glass);
            border-radius: 10px;
            background: rgba(9, 13, 22, 0.5);
        }

        .details-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        .details-table th {
            background: rgba(17, 24, 39, 0.9);
            color: var(--text-muted);
            font-weight: 500;
            padding: 8px 14px;
            text-align: left;
            position: sticky;
            top: 0;
            border-bottom: 1px solid var(--border-glass);
        }

        .details-table td {
            padding: 8px 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }

        .conf-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .conf-high { background: rgba(16, 185, 129, 0.2); color: #34d399; }
        .conf-med { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
        .conf-low { background: rgba(239, 68, 68, 0.2); color: #f87171; }

        /* SPINNER LOADER */
        .spinner {
            display: none;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* WEBCAM & OTHER VIEWS */
        .view-section {
            display: none;
        }

        .view-section.active {
            display: block;
        }

        #webcamVideo {
            width: 100%;
            max-height: 380px;
            border-radius: 12px;
            background: #000;
            object-fit: cover;
        }

        /* HISTORY ITEMS */
        .history-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .history-item {
            background: var(--bg-card);
            border: 1px solid var(--border-glass);
            border-radius: 14px;
            padding: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
        }

        .history-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            color: var(--accent-cyan);
            font-weight: 600;
        }

        .history-time {
            font-size: 12px;
            color: var(--text-muted);
        }

        /* CODE BLOCK */
        .code-block {
            background: #090d16;
            border: 1px solid var(--border-glass);
            border-radius: 12px;
            padding: 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: #e2e8f0;
            overflow-x: auto;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand-container">
            <div class="brand-logo">
                <i class="fa-solid fa-eye"></i>
            </div>
            <div>
                <div class="brand-title">VisionOCR AI</div>
                <div class="brand-subtitle">EasyOCR Engine & License Plate Text Extraction</div>
            </div>
        </div>

        <div class="nav-tabs">
            <button class="nav-btn active" onclick="showTab('scannerTab', this)">
                <i class="fa-solid fa-expand"></i> Image Reader
            </button>
            <button class="nav-btn" onclick="showTab('webcamTab', this); startWebcam();">
                <i class="fa-solid fa-camera"></i> Live Webcam
            </button>
            <button class="nav-btn" onclick="showTab('historyTab', this)">
                <i class="fa-solid fa-clock-rotate-left"></i> History
            </button>
            <button class="nav-btn" onclick="showTab('apiTab', this)">
                <i class="fa-solid fa-code"></i> API Docs
            </button>
        </div>

        <div class="status-pill">
            <div class="pulse-dot"></div>
            Model Ready (PyTorch)
        </div>
    </header>

    <main>
        <!-- SCANNER TAB -->
        <div id="scannerTab" class="view-section active">
            <!-- TOOLBAR -->
            <div class="toolbar" style="margin-bottom: 24px;">
                <div class="control-group">
                    <span class="control-label"><i class="fa-solid fa-wand-magic-sparkles"></i> Quick Samples:</span>
                    <button class="preset-btn" onclick="loadSample('/sample/number_plate', 'sample_number_plate.png')">
                        <i class="fa-solid fa-car-side"></i> Vehicle License Plate (RJ14CV0002)
                    </button>
                    <button class="preset-btn" onclick="loadSample('/sample/document', 'sample_test.png')">
                        <i class="fa-solid fa-file-invoice"></i> Document Sample
                    </button>
                </div>

                <div class="control-group">
                    <span class="control-label"><i class="fa-solid fa-language"></i> Language:</span>
                    <select id="languageSelect">
                        <option value="en">English (en)</option>
                        <option value="en,hi">English + Hindi (en, hi)</option>
                        <option value="en,es">English + Spanish (en, es)</option>
                        <option value="en,fr">English + French (en, fr)</option>
                    </select>
                </div>
            </div>

            <!-- GRID WORKSPACE -->
            <div class="workspace-grid">
                <!-- LEFT CARD: IMAGE PREVIEW -->
                <div class="glass-card">
                    <div class="card-header">
                        <div class="card-title">
                            <i class="fa-regular fa-image"></i> Input Image & Detection Bounding Boxes
                        </div>
                        <button class="btn-secondary" id="resetBtn" onclick="resetInput()" style="display: none;">
                            <i class="fa-solid fa-rotate-left"></i> Reset
                        </button>
                    </div>

                    <!-- DROPZONE -->
                    <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                        <div class="drop-icon">
                            <i class="fa-solid fa-cloud-arrow-up"></i>
                        </div>
                        <div>
                            <p style="font-weight: 600; font-size: 16px;">Drop your image here or click to browse</p>
                            <p style="font-size: 13px; color: var(--text-muted); margin-top: 6px;">Supports License Plates, Documents, Signs (PNG, JPG, WEBP)</p>
                        </div>
                    </div>

                    <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">

                    <!-- PREVIEW CONTAINER -->
                    <div class="preview-container" id="previewContainer">
                        <img id="previewImg" class="preview-img" alt="Uploaded Image Preview">
                    </div>

                    <!-- ACTION BUTTON -->
                    <button class="btn-primary" id="readBtn" onclick="runOCR()" disabled>
                        <span class="spinner" id="spinner"></span>
                        <i class="fa-solid fa-bolt" id="btnIcon"></i>
                        <span id="btnText">Extract Text with EasyOCR</span>
                    </button>
                </div>

                <!-- RIGHT CARD: RESULT DASHBOARD -->
                <div class="glass-card">
                    <div class="card-header">
                        <div class="card-title">
                            <i class="fa-solid fa-font"></i> Extracted Result & Analysis
                        </div>
                        <div style="display: flex; gap: 8px;">
                            <button class="btn-secondary" onclick="copyText()">
                                <i class="fa-regular fa-copy"></i> Copy
                            </button>
                            <button class="btn-secondary" onclick="downloadJSON()">
                                <i class="fa-solid fa-download"></i> JSON
                            </button>
                        </div>
                    </div>

                    <!-- METRICS -->
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-val" id="metricWords">0</div>
                            <div class="metric-lbl">Words Found</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-val" id="metricChars">0</div>
                            <div class="metric-lbl">Characters</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-val" id="metricConf">0%</div>
                            <div class="metric-lbl">Avg Confidence</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-val" id="metricTime">0ms</div>
                            <div class="metric-lbl">Proc. Time</div>
                        </div>
                    </div>

                    <!-- FULL TEXT DISPLAY -->
                    <div id="fullTextDisplay" class="text-output-box">Upload an image and click 'Extract Text' to read content...</div>

                    <!-- DETAILED WORD TABLE -->
                    <div>
                        <div style="font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 8px;">
                            Detailed Breakdown & Coordinates
                        </div>
                        <div class="details-table-container">
                            <table class="details-table">
                                <thead>
                                    <tr>
                                        <th>#</th>
                                        <th>Detected Text</th>
                                        <th>Confidence</th>
                                    </tr>
                                </thead>
                                <tbody id="detailsTableBody">
                                    <tr>
                                        <td colspan="3" style="text-align: center; color: var(--text-dim); padding: 14px;">No text extracted yet</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- WEBCAM TAB -->
        <div id="webcamTab" class="view-section">
            <div class="glass-card" style="max-width: 800px; margin: 0 auto;">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fa-solid fa-video"></i> Live Webcam License Plate & Document Scanner
                    </div>
                </div>
                <video id="webcamVideo" autoplay playsinline></video>
                <div style="display: flex; gap: 12px; justify-content: center; margin-top: 10px;">
                    <button class="btn-primary" onclick="captureWebcamFrame()">
                        <i class="fa-solid fa-camera"></i> Capture & Scan Frame
                    </button>
                    <button class="btn-secondary" onclick="stopWebcam()">
                        <i class="fa-solid fa-power-off"></i> Stop Camera
                    </button>
                </div>
            </div>
        </div>

        <!-- HISTORY TAB -->
        <div id="historyTab" class="view-section">
            <div class="glass-card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fa-solid fa-history"></i> Recent Extractions Log
                    </div>
                    <button class="btn-secondary" onclick="clearHistory()">
                        <i class="fa-solid fa-trash"></i> Clear History
                    </button>
                </div>
                <div class="history-list" id="historyList">
                    <div style="text-align: center; color: var(--text-muted); padding: 40px;">No scan history yet. Try scanning an image!</div>
                </div>
            </div>
        </div>

        <!-- API DOCS TAB -->
        <div id="apiTab" class="view-section">
            <div class="glass-card">
                <div class="card-header">
                    <div class="card-title">
                        <i class="fa-solid fa-code"></i> VisionOCR REST API Integration Guide
                    </div>
                </div>
                <p style="color: var(--text-muted); font-size: 14px;">Send images directly to our local EasyOCR backend via standard HTTP POST request.</p>

                <div style="margin-top: 16px;">
                    <h4 style="color: var(--accent-cyan); margin-bottom: 6px;">Python Integration Example</h4>
                    <div class="code-block">
import requests

url = "http://127.0.0.1:5000/api/ocr"
files = {"file": open("license_plate.png", "rb")}
data = {"languages": "en"}

response = requests.post(url, files=files, data=data)
result = response.json()

print("Extracted Text:", result["full_text"])
                    </div>
                </div>

                <div style="margin-top: 16px;">
                    <h4 style="color: var(--accent-purple); margin-bottom: 6px;">cURL Terminal Command</h4>
                    <div class="code-block">
curl -X POST -F "file=@your_image.png" -F "languages=en" http://127.0.0.1:5000/api/ocr
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        let currentFile = null;
        let lastResultData = null;
        let webcamStream = null;

        // Navigation
        function showTab(tabId, btn) {
            document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            btn.classList.add('active');

            if (tabId !== 'webcamTab') {
                stopWebcam();
            }
            if (tabId === 'historyTab') {
                loadHistory();
            }
        }

        // Dropzone & File Handling
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const previewImg = document.getElementById('previewImg');
        const readBtn = document.getElementById('readBtn');
        const resetBtn = document.getElementById('resetBtn');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => {
            dropzone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); });
        });

        ['dragenter', 'dragover'].forEach(ev => {
            dropzone.addEventListener(ev, () => dropzone.classList.add('dragover'));
        });

        ['dragleave', 'drop'].forEach(ev => {
            dropzone.addEventListener(ev, () => dropzone.classList.remove('dragover'));
        });

        dropzone.addEventListener('drop', e => {
            const files = e.dataTransfer.files;
            if (files.length > 0) handleFile(files[0]);
        });

        function handleFileSelect(e) {
            if (e.target.files.length > 0) handleFile(e.target.files[0]);
        }

        function handleFile(file) {
            currentFile = file;
            const reader = new FileReader();
            reader.onload = e => {
                previewImg.src = e.target.result;
                dropzone.style.display = 'none';
                previewContainer.style.display = 'flex';
                resetBtn.style.display = 'inline-flex';
                readBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }

        function resetInput() {
            currentFile = null;
            previewImg.src = '';
            dropzone.style.display = 'flex';
            previewContainer.style.display = 'none';
            resetBtn.style.display = 'none';
            readBtn.disabled = true;
            fileInput.value = '';
            document.getElementById('fullTextDisplay').innerText = "Upload an image and click 'Extract Text' to read content...";
            document.getElementById('detailsTableBody').innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-dim); padding: 14px;">No text extracted yet</td></tr>';
            document.getElementById('metricWords').innerText = '0';
            document.getElementById('metricChars').innerText = '0';
            document.getElementById('metricConf').innerText = '0%';
            document.getElementById('metricTime').innerText = '0ms';
        }

        async function loadSample(sampleUrl, filename) {
            try {
                const response = await fetch(sampleUrl);
                const blob = await response.blob();
                const file = new File([blob], filename, { type: 'image/png' });
                handleFile(file);
            } catch (e) {
                alert("Failed to load sample image: " + e.message);
            }
        }

        // OCR Execution
        async function runOCR() {
            if (!currentFile) return;

            const startTime = performance.now();
            readBtn.disabled = true;
            document.getElementById('spinner').style.display = 'inline-block';
            document.getElementById('btnIcon').style.display = 'none';
            document.getElementById('btnText').innerText = 'Processing with EasyOCR...';

            const formData = new FormData();
            formData.append('file', currentFile);
            formData.append('languages', document.getElementById('languageSelect').value);

            try {
                const response = await fetch('/api/ocr', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                const endTime = performance.now();
                const procTime = Math.round(endTime - startTime);

                lastResultData = data;

                if (data.success) {
                    const text = data.full_text || "(No readable text detected in image)";
                    document.getElementById('fullTextDisplay').innerText = text;

                    if (data.annotated_image_base64) {
                        previewImg.src = data.annotated_image_base64;
                    }

                    // Calculate Metrics
                    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
                    const chars = text.length;
                    
                    let avgConf = 0;
                    if (data.results && data.results.length > 0) {
                        const sumConf = data.results.reduce((acc, r) => acc + (r.confidence || 0), 0);
                        avgConf = Math.round((sumConf / data.results.length) * 100);
                    }

                    document.getElementById('metricWords').innerText = words;
                    document.getElementById('metricChars').innerText = chars;
                    document.getElementById('metricConf').innerText = avgConf + '%';
                    document.getElementById('metricTime').innerText = procTime + 'ms';

                    // Populate Details Table
                    const tbody = document.getElementById('detailsTableBody');
                    if (data.results && data.results.length > 0) {
                        tbody.innerHTML = data.results.map((item, idx) => {
                            const conf = Math.round(item.confidence * 100);
                            let confClass = 'conf-high';
                            if (conf < 50) confClass = 'conf-low';
                            else if (conf < 75) confClass = 'conf-med';

                            return `
                                <tr>
                                    <td>${idx + 1}</td>
                                    <td style="color: #fff; font-weight: 600; font-family: 'JetBrains Mono', monospace;">${escapeHtml(item.text)}</td>
                                    <td><span class="conf-badge ${confClass}">${conf}%</span></td>
                                </tr>
                            `;
                        }).join('');
                    } else {
                        tbody.innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-dim); padding: 14px;">No text regions identified</td></tr>';
                    }

                    // Save to history
                    saveToHistory(text, avgConf);
                } else {
                    document.getElementById('fullTextDisplay').innerText = "Error: " + (data.error || "Failed to process image");
                }
            } catch (err) {
                document.getElementById('fullTextDisplay').innerText = "Server Error: " + err.message;
            } finally {
                readBtn.disabled = false;
                document.getElementById('spinner').style.display = 'none';
                document.getElementById('btnIcon').style.display = 'inline-block';
                document.getElementById('btnText').innerText = 'Extract Text with EasyOCR';
            }
        }

        // Copy & Download
        function copyText() {
            const txt = document.getElementById('fullTextDisplay').innerText;
            if (txt && !txt.startsWith("Upload an image") && !txt.startsWith("Error")) {
                navigator.clipboard.writeText(txt);
                alert("Extracted text copied to clipboard!");
            }
        }

        function downloadJSON() {
            if (!lastResultData) return alert("No result available to download");
            const blob = new Blob([JSON.stringify(lastResultData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ocr_result_${Date.now()}.json`;
            a.click();
        }

        // Webcam Stream
        async function startWebcam() {
            try {
                webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
                document.getElementById('webcamVideo').srcObject = webcamStream;
            } catch (err) {
                alert("Unable to access camera: " + err.message);
            }
        }

        function stopWebcam() {
            if (webcamStream) {
                webcamStream.getTracks().forEach(track => track.stop());
                webcamStream = null;
            }
        }

        function captureWebcamFrame() {
            const video = document.getElementById('webcamVideo');
            if (!video.srcObject) return;

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);

            canvas.toBlob(blob => {
                const file = new File([blob], "webcam_frame.png", { type: "image/png" });
                showTab('scannerTab', document.querySelectorAll('.nav-btn')[0]);
                handleFile(file);
                runOCR();
            }, 'image/png');
        }

        // Local Storage History
        function saveToHistory(text, conf) {
            let hist = JSON.parse(localStorage.getItem('vision_ocr_history') || '[]');
            hist.unshift({
                text: text,
                confidence: conf,
                time: new Date().toLocaleTimeString() + ', ' + new Date().toLocaleDateString()
            });
            if (hist.length > 20) hist.pop();
            localStorage.setItem('vision_ocr_history', JSON.stringify(hist));
        }

        function loadHistory() {
            const container = document.getElementById('historyList');
            let hist = JSON.parse(localStorage.getItem('vision_ocr_history') || '[]');
            if (hist.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 40px;">No scan history yet. Try scanning an image!</div>';
                return;
            }

            container.innerHTML = hist.map(item => `
                <div class="history-item">
                    <div>
                        <div class="history-text">${escapeHtml(item.text)}</div>
                        <div class="history-time"><i class="fa-regular fa-clock"></i> ${item.time} • Confidence: ${item.confidence}%</div>
                    </div>
                    <button class="btn-secondary" onclick="navigator.clipboard.writeText('${escapeJs(item.text)}')">
                        <i class="fa-regular fa-copy"></i> Copy
                    </button>
                </div>
            `).join('');
        }

        function clearHistory() {
            localStorage.removeItem('vision_ocr_history');
            loadHistory();
        }

        function escapeHtml(text) {
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }

        function escapeJs(text) {
            return text.replace(/'/g, "\\'").replace(/"/g, '\\"');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/sample/number_plate')
def get_sample_number_plate():
    path = os.path.join(os.path.dirname(__file__), 'sample_number_plate.png')
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    return "Sample not found", 404

@app.route('/sample/document')
def get_sample_document():
    path = os.path.join(os.path.dirname(__file__), 'sample_test.png')
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    return "Sample not found", 404

@app.route('/api/ocr', methods=['POST'])
def ocr_api():
    """
    API Endpoint for EasyOCR text extraction.
    Supports multipart form-data (file upload) or JSON (base64 string).
    """
    try:
        languages = ('en',)
        
        if request.content_type and 'application/json' in request.content_type:
            data = request.get_json() or {}
            lang_param = data.get('languages', 'en')
            if isinstance(lang_param, str):
                languages = tuple(l.strip() for l in lang_param.split(','))
            elif isinstance(lang_param, list):
                languages = tuple(lang_param)
                
            img_b64 = data.get('image_base64')
            if not img_b64:
                return jsonify({"success": False, "error": "Missing 'image_base64' in JSON body"}), 400
                
            res = OCRService.read_text(img_b64, languages=languages)
            return jsonify(res)

        else:
            lang_param = request.form.get('languages', 'en')
            if isinstance(lang_param, str):
                languages = tuple(l.strip() for l in lang_param.split(','))
                
            if 'file' not in request.files:
                return jsonify({"success": False, "error": "No image file provided in 'file' parameter"}), 400
                
            file = request.files['file']
            if file.filename == '':
                return jsonify({"success": False, "error": "Empty filename"}), 400

            file_bytes = file.read()
            res = OCRService.read_text(file_bytes, languages=languages)
            return jsonify(res)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[VisionOCR] Server running on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
