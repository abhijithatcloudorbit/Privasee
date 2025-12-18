"use client"

import type React from "react"
import { useState } from "react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"

import { Upload, X, FileImage, Shield, Settings, Info } from "lucide-react"

// 🔌 Backend API helpers
import { uploadFile } from "@/lib/api/upload"
import { startProcessing } from "@/lib/api/process"
import { getJobStatus } from "@/lib/api/status"

export default function UploadPage() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isDragging, setIsDragging] = useState(false)

  const [processing, setProcessing] = useState(false)
  const [progress, setProgress] = useState(0)

  const [selectedModels, setSelectedModels] = useState({
    face: true,
    person: true,
    text: true,
    medical: false,
    licensePlate: false,
  })

  const [complianceMode, setComplianceMode] =
    useState<"strict" | "moderate" | "custom">("moderate")

  /* ---------------- Drag & Drop ---------------- */

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files)
    setSelectedFiles((prev) => [...prev, ...files])
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files)
      setSelectedFiles((prev) => [...prev, ...files])
    }
  }

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const formatBytes = (bytes: number) =>
    (bytes / (1024 * 1024)).toFixed(2) + " MB"

  const toggleModel = (model: keyof typeof selectedModels) => {
    setSelectedModels((prev) => ({ ...prev, [model]: !prev[model] }))
  }

  /* ---------------- Download Helper ---------------- */

  async function downloadResult(jobId: string) {
    const res = await fetch(
      `${process.env.NEXT_PUBLIC_PROCESSING_API}/result/${jobId}`
    )

    if (!res.ok) {
      throw new Error("Failed to download processed file")
    }

    const blob = await res.blob()

    // Try to extract filename from headers
    const disposition = res.headers.get("content-disposition")
    let filename = "processed_file"

    if (disposition && disposition.includes("filename=")) {
      filename = disposition.split("filename=")[1].replace(/"/g, "")
    }

    const url = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
    window.URL.revokeObjectURL(url)
  }

  /* ---------------- Process Button Logic ---------------- */

  async function handleProcessClick() {
    if (selectedFiles.length === 0) return

    try {
      setProcessing(true)
      setProgress(0)

      // 1️⃣ Upload file (MVP = first file)
      const uploadRes = await uploadFile(selectedFiles[0])
      const fileId = uploadRes.file_id

      // 2️⃣ Start processing
      const processRes = await startProcessing(fileId)
      const jobId = processRes.job_id

      // 3️⃣ Poll job status
      const interval = setInterval(async () => {
        const status = await getJobStatus(jobId)
        setProgress(status.progress)

        if (status.status === "COMPLETED") {
          clearInterval(interval)
          setProcessing(false)

          // 4️⃣ Download processed file
          await downloadResult(jobId)
        }
      }, 1500)
    } catch (err) {
      console.error(err)
      setProcessing(false)
    }
  }

  /* ---------------- UI ---------------- */

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-semibold tracking-tight">
          Secure Data Upload
        </h1>
        <p className="text-muted-foreground mt-2">
          Configure privacy controls and upload files for compliant anonymization
        </p>
      </div>

      {/* Detection Controls */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Settings className="size-5 text-primary" />
            <CardTitle>Privasee Detection Controls</CardTitle>
          </div>
          <CardDescription>
            Select which privacy detectors to apply
          </CardDescription>
        </CardHeader>

        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {(
              [
                ["face", "Face Detection"],
                ["person", "Person Detection"],
                ["text", "Text Redaction"],
                ["medical", "Medical Data"],
                ["licensePlate", "License Plates"],
              ] as const
            ).map(([key, label]) => (
              <button
                key={key}
                onClick={() => toggleModel(key)}
                className={`p-4 rounded-lg border-2 text-left transition-all ${
                  selectedModels[key]
                    ? "border-primary bg-primary/5"
                    : "border-border hover:border-primary/50"
                }`}
              >
                <Badge
                  variant={selectedModels[key] ? "default" : "outline"}
                  className="mb-2"
                >
                  {label}
                </Badge>
                <p className="text-sm text-muted-foreground">
                  Enable {label.toLowerCase()}
                </p>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Compliance Mode */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Shield className="size-5 text-primary" />
            <CardTitle>Compliance Mode</CardTitle>
          </div>
        </CardHeader>

        <CardContent className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {(["strict", "moderate", "custom"] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => setComplianceMode(mode)}
              className={`p-4 rounded-lg border-2 text-left ${
                complianceMode === mode
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50"
              }`}
            >
              <Badge
                variant={complianceMode === mode ? "default" : "outline"}
                className="mb-2"
              >
                {mode.toUpperCase()}
              </Badge>
              <p className="text-sm text-muted-foreground">
                {mode} privacy enforcement
              </p>
            </button>
          ))}
        </CardContent>
      </Card>

      {/* File Upload */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Upload className="size-5 text-primary" />
            <CardTitle>File Upload</CardTitle>
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-xl p-12 text-center ${
              isDragging
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/50"
            }`}
          >
            <input
              id="file-input"
              type="file"
              multiple
              accept="image/*,.pdf"
              onChange={handleFileInput}
              className="hidden"
            />
            <label htmlFor="file-input" className="cursor-pointer">
              <FileImage className="mx-auto mb-4 size-10 text-primary" />
              <p className="font-medium">
                Drag files here or click to select
              </p>
            </label>
          </div>

          {selectedFiles.map((file, i) => (
            <div
              key={i}
              className="flex items-center gap-4 p-3 rounded-lg bg-secondary/30"
            >
              <FileImage className="size-6 text-primary" />
              <div className="flex-1">
                <p className="truncate">{file.name}</p>
                <p className="text-xs text-muted-foreground">
                  {formatBytes(file.size)}
                </p>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => removeFile(i)}
              >
                <X className="size-4" />
              </Button>
            </div>
          ))}

          {/* Process Button */}
          <div className="flex gap-4 pt-4">
            <Button
              size="lg"
              className="flex-1 h-12"
              disabled={selectedFiles.length === 0 || processing}
              onClick={handleProcessClick}
            >
              <Upload className="size-4" />
              {processing
                ? `Processing ${progress}%`
                : `Process ${selectedFiles.length} Files`}
            </Button>

            <Button variant="outline" size="lg">
              <Info className="size-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
