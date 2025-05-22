'use client';

import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';

export default function VideoStreamer() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [caption, setCaption] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [mode, setMode] = useState<"image" | "caption">("caption");

  useEffect(() => {
    // Only run in the browser
    if (typeof window === 'undefined') return;
    if (!('mediaDevices' in navigator) || !navigator.mediaDevices?.getUserMedia) {
      console.warn("mediaDevices API not available in this environment.");
      return;
    }

    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Failed to access camera:", err);
      }
    };

    setupCamera();
  }, []);

  // Stream frames
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;

    if (isStreaming) {
      interval = setInterval(captureAndSendFrame, 1000); // every 1s
    }

    return () => clearInterval(interval);
  }, [isStreaming, mode]);

  const captureAndSendFrame = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob) return;

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      try {
        if (mode === "image") {
          const response = await axios.post("http://10.236.237.55:8000/process-frame", formData, {
            responseType: "blob",
          });

          const url = URL.createObjectURL(response.data);
          setProcessedImage(url);
          setCaption(null);
        } else {
          const response = await axios.post("http://10.236.237.55:8000/justJSON", formData);
          setCaption(response.data.caption);
          setProcessedImage(null);
        }
      } catch (err) {
        console.error("Frame send error", err);
      }
    }, "image/jpeg");
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Real-Time Video Captioning</h1>

      <div style={{ position: "relative", width: 400 }}>
        <video ref={videoRef} autoPlay muted playsInline style={{ width: "100%", borderRadius: 8 }} />
        {caption && (
          <div
            style={{
              position: "absolute",
              top: 10,
              left: 10,
              backgroundColor: "rgba(0,0,0,0.6)",
              color: "white",
              padding: "4px 8px",
              borderRadius: "4px",
              fontSize: "14px",
            }}
          >
            {caption}
          </div>
        )}
      </div>

      <canvas ref={canvasRef} style={{ display: "none" }} />

      <div style={{ marginTop: 10 }}>
        <button onClick={() => setIsStreaming(!isStreaming)}>
          {isStreaming ? "Stop" : "Start"} Streaming
        </button>

        <label style={{ marginLeft: 16 }}>
          <input
            type="radio"
            name="mode"
            value="caption"
            checked={mode === "caption"}
            onChange={() => setMode("caption")}
          />
          Overlay Caption
        </label>

        <label style={{ marginLeft: 8 }}>
          <input
            type="radio"
            name="mode"
            value="image"
            checked={mode === "image"}
            onChange={() => setMode("image")}
          />
          Show Processed Image
        </label>
      </div>

      {processedImage && (
        <div style={{ marginTop: 20 }}>
          <h2>Processed Frame:</h2>
          <img src={processedImage} alt="Processed" style={{ maxWidth: "100%" }} />
        </div>
      )}
    </div>
  );
}
