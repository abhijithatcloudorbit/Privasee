// File: src/pages/dashboard/DashboardPage.jsx
// Notion-style clean dashboard layout (Pure CSS) — Professional UI

import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion } from "framer-motion";
import Navbar from "../../components/layouts/Navbar.jsx";
import "./dashboard.css";

export default function DashboardPage() {
  const [files, setFiles] = useState([]);
  const [rejected, setRejected] = useState("");
  const [preview, setPreview] = useState(null);
  const [progress, setProgress] = useState(0);
  const [activity, setActivity] = useState([]);

  const simulateUpload = () => {
    setProgress(0);
    let val = 0;
    const t = setInterval(() => {
      val += 8;
      setProgress(val);
      if (val >= 100) clearInterval(t);
    }, 120);
  };

  const onDrop = useCallback((accepted, rejectedFiles) => {
    if (rejectedFiles?.length) {
      setRejected("Only PDF, PNG, JPG, JPEG allowed.");
    } else setRejected("");

    if (accepted.length > 0) {
      simulateUpload();

      const timestamp = new Date().toLocaleTimeString();
      const mapped = accepted.map(f => ({
        file: f,
        url: URL.createObjectURL(f),
        type: f.type.includes("pdf") ? "pdf" : "image",
        time: timestamp
      }));

      setFiles(prev => [...mapped, ...prev]);
      setActivity(prev => [{ text: `Uploaded ${accepted[0].name}`, time: timestamp }, ...prev]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: true,
    accept: {
      "application/pdf": [".pdf"],
      "image/png": [".png"],
      "image/jpeg": [".jpg", ".jpeg"],
    },
  });

  return (
    <>
      <Navbar />

      <div className="dash-wrapper">
        {/* SIDEBAR */}
        <aside className="dash-sidebar">
          <h2 className="side-heading">Dashboard</h2>
          <ul className="side-nav">
            <li>Overview</li>
            <li>Uploads</li>
            <li>Activity</li>
            <li>Settings</li>
          </ul>
        </aside>

        {/* MAIN */}
        <main className="dash-main">
          <h1 className="page-title">Upload Center</h1>

          {/* GRID LAYOUT */}
          <div className="grid-2col">
            {/* Storage Card */}
            <section className="card storage-card">
              <h3 className="card-title">Storage Used</h3>
              <div className="storage-circle-wrapper">
                <svg width="110" height="110">
                  <circle cx="55" cy="55" r="45" stroke="#e5e7eb" strokeWidth="8" fill="none" />
                  <circle
                    cx="55"
                    cy="55"
                    r="45"
                    stroke="#4f46e5"
                    strokeWidth="8"
                    fill="none"
                    strokeDasharray={282}
                    strokeDashoffset={282 - files.length * 20}
                    strokeLinecap="round"
                  />
                </svg>
              </div>
              <p className="storage-value">{files.length * 10}MB / 100MB</p>
            </section>

            {/* Upload Zone */}
            <section className="card">
              <motion.div
                {...getRootProps()}
                className={`drop-zone ${isDragActive ? "active-drop" : ""}`}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.98 }}
              >
                <input {...getInputProps()} />
                {isDragActive ? (
                  <p className="drop-text-active">Release to upload</p>
                ) : (
                  <>
                    <p className="drop-text">Drag & drop your files</p>
                    <p className="drop-sub">PDF, PNG, JPG, JPEG only</p>
                  </>
                )}
              </motion.div>

              {progress > 0 && progress < 100 && (
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${progress}%` }}></div>
                </div>
              )}

              {rejected && <p className="error-msg">{rejected}</p>}
            </section>
          </div>

          {/* Recent Uploads */}
          {files.length > 0 && (
            <section className="card">
              <h3 className="card-title">Recent Uploads</h3>
              <ul className="file-list">
                {files.map((f, i) => (
                  <li key={i} className="file-row" onClick={() => setPreview(f)}>
                    <span className="file-icon">{f.type === "pdf" ? "📄" : "🖼️"}</span>
                    <span className="file-name">{f.file.name}</span>
                    <span className="file-size">{Math.round(f.file.size / 1024)} KB</span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Activity Log */}
          <section className="card">
            <h3 className="card-title">Activity Log</h3>
            <ul className="activity-list">
              {activity.map((act, idx) => (
                <li key={idx} className="activity-row">
                  {act.time} — {act.text}
                </li>
              ))}
            </ul>
          </section>
        </main>
      </div>

      {/* Preview Modal */}
      {preview && (
        <div className="modal-overlay" onClick={() => setPreview(null)}>
          <div className="modal-box">
            <h3 className="modal-title">{preview.file.name}</h3>

            {preview.type === "image" ? (
              <img src={preview.url} className="modal-img" />
            ) : (
              <iframe src={preview.url} className="modal-pdf" />
            )}
          </div>
        </div>
      )}
    </>
  );
}

/* Should restructure the code and create reusable components and should use tailwindcss */