"use client"

import type React from "react"
import { useState, useRef } from "react"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

import { Upload, FileImage, Shield, Settings } from "lucide-react"

import { uploadFile } from "@/lib/api/upload"
import { startProcessing } from "@/lib/api/process"
import { getJobStatus } from "@/lib/api/status"
import { getResultBlobUrl } from "@/lib/api/result"

export default function UploadPage() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [processing, setProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [outputUrl, setOutputUrl] = useState<string | null>(null)

  const pollingRef = useRef<NodeJS.Timeout | null>(null)

  const [selectedModels, setSelectedModels] = useState({
    face: true,
    person: true,
    text: true,
    medical: false,
    licensePlate: false,
  })

  const [complianceMode, setComplianceMode] =
    useState<"strict" | "moderate" | "custom">("moderate")

  const resolveProcessingMode = () => {
    if (selectedModels.licensePlate) return "license_plate"
    if (selectedModels.face) return "face"
    return "face"
  }

  async function handleProcessClick() {
    if (selectedFiles.length === 0) return

    try {
      setProcessing(true)
      setProgress(0)
      setOutputUrl(null)

      // 1️⃣ Upload
      const uploadRes = await uploadFile(selectedFiles[0])

      // 2️⃣ Start processing
      const mode = resolveProcessingMode()
      const processRes = await startProcessing(uploadRes.file_id, mode)

      if (!processRes?.job_id) {
        throw new Error("Processing did not return a job_id")
      }

      const jobId = processRes.job_id

      // 3️⃣ Poll status
      pollingRef.current = setInterval(async () => {
        try {
          const statusRes = await getJobStatus(jobId)
          setProgress(statusRes.progress ?? 0)

          if (statusRes.status === "FAILED") {
            clearInterval(pollingRef.current!)
            setProcessing(false)
            console.error("Processing failed")
          }

          if (statusRes.status === "COMPLETED") {
            clearInterval(pollingRef.current!)
            setProcessing(false)

            // 4️⃣ Fetch result blob
            const url = await getResultBlobUrl(jobId)
            setOutputUrl(url)
          }
        } catch (err) {
          clearInterval(pollingRef.current!)
          setProcessing(false)
          console.error("Status polling failed", err)
        }
      }, 1500)
    } catch (err) {
      setProcessing(false)
      console.error("Upload or processing failed", err)
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-4xl font-semibold tracking-tight">
          Secure Data Upload
        </h1>
        <p className="text-muted-foreground mt-2">
          Configure privacy controls and upload files for compliant anonymization
        </p>
      </div>

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
                onClick={() =>
                  setSelectedModels((prev) => ({
                    ...prev,
                    [key]: !prev[key],
                  }))
                }
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

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Upload className="size-5 text-primary" />
            <CardTitle>File Upload</CardTitle>
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          <input
            type="file"
            accept="image/*"
            onChange={(e) =>
              e.target.files && setSelectedFiles([e.target.files[0]])
            }
          />

          <Button
            size="lg"
            className="w-full h-12"
            disabled={selectedFiles.length === 0 || processing}
            onClick={handleProcessClick}
          >
            {processing ? `Processing ${progress}%` : "Upload & Process"}
          </Button>

          {outputUrl && (
            <div className="pt-6">
              <h3 className="text-lg font-medium mb-3">
                Processed Output
              </h3>
              <img
                src={outputUrl}
                alt="Processed result"
                className="rounded-lg border max-w-full"
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
