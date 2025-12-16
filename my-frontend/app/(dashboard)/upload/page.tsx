"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"
import { Upload, X, FileImage, Shield, Settings, Info } from "lucide-react"

export default function UploadPage() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [selectedModels, setSelectedModels] = useState({
    face: true,
    person: true,
    text: true,
    medical: false,
    licensePlate: false,
  })
  const [complianceMode, setComplianceMode] = useState<"strict" | "moderate" | "custom">("moderate")

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

  const formatBytes = (bytes: number) => {
    return (bytes / (1024 * 1024)).toFixed(2) + " MB"
  }

  const toggleModel = (model: keyof typeof selectedModels) => {
    setSelectedModels((prev) => ({ ...prev, [model]: !prev[model] }))
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-semibold tracking-tight text-balance">Secure Data Upload</h1>
        <p className="text-muted-foreground mt-2">Configure privacy controls and upload files for compliant anonymization</p>
      </div>

      {/* Model Selection */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Settings className="size-5 text-primary" />
            <CardTitle>Privasee Detection Controls</CardTitle>
          </div>
          <CardDescription>Select which privacy detectors to apply during anonymization</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button
              onClick={() => toggleModel("face")}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                selectedModels.face ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <Badge variant={selectedModels.face ? "default" : "outline"}>Face Detection</Badge>
                <div
                  className={`size-5 rounded-full border-2 flex items-center justify-center ${
                    selectedModels.face ? "border-primary bg-primary" : "border-border"
                  }`}
                >
                  {selectedModels.face && <div className="size-2 rounded-full bg-background" />}
                </div>
              </div>
              <p className="text-sm text-muted-foreground">Detect and anonymize human faces</p>
            </button>

            <button
              onClick={() => toggleModel("person")}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                selectedModels.person ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <Badge variant={selectedModels.person ? "default" : "outline"}>Person Detection</Badge>
                <div
                  className={`size-5 rounded-full border-2 flex items-center justify-center ${
                    selectedModels.person ? "border-primary bg-primary" : "border-border"
                  }`}
                >
                  {selectedModels.person && <div className="size-2 rounded-full bg-background" />}
                </div>
              </div>
              <p className="text-sm text-muted-foreground">Detect and anonymize individuals in images</p>
            </button>

            <button
              onClick={() => toggleModel("text")}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                selectedModels.text ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <Badge variant={selectedModels.text ? "default" : "outline"}>Text Redaction</Badge>
                <div
                  className={`size-5 rounded-full border-2 flex items-center justify-center ${
                    selectedModels.text ? "border-primary bg-primary" : "border-border"
                  }`}
                >
                  {selectedModels.text && <div className="size-2 rounded-full bg-background" />}
                </div>
              </div>
              <p className="text-sm text-muted-foreground">Detect and anonymize sensitive text content</p>
            </button>

            <button
              onClick={() => toggleModel("medical")}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                selectedModels.medical ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <Badge variant={selectedModels.medical ? "default" : "outline"}>Medical Data</Badge>
                <div
                  className={`size-5 rounded-full border-2 flex items-center justify-center ${
                    selectedModels.medical ? "border-primary bg-primary" : "border-border"
                  }`}
                >
                  {selectedModels.medical && <div className="size-2 rounded-full bg-background" />}
                </div>
              </div>
              <p className="text-sm text-muted-foreground">Anonymization for medical and clinical data</p>
            </button>

            <button
              onClick={() => toggleModel("licensePlate")}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                selectedModels.licensePlate ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <Badge variant={selectedModels.licensePlate ? "default" : "outline"}>License Plates</Badge>
                <div
                  className={`size-5 rounded-full border-2 flex items-center justify-center ${
                    selectedModels.licensePlate ? "border-primary bg-primary" : "border-border"
                  }`}
                >
                  {selectedModels.licensePlate && <div className="size-2 rounded-full bg-background" />}
                </div>
              </div>
              <p className="text-sm text-muted-foreground">Detect and anonymize vehicle identifiers</p>
            </button>
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
          <CardDescription>Select the privacy enforcement level</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button
              onClick={() => setComplianceMode("strict")}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                complianceMode === "strict" ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
              }`}
            >
              <Badge variant={complianceMode === "strict" ? "default" : "outline"} className="mb-2">
                Strict
              </Badge>
              <p className="text-sm text-muted-foreground">Maximum privacy enforcement for regulated data</p>
            </button>

            <button
              onClick={() => setComplianceMode("moderate")}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                complianceMode === "moderate" ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
              }`}
            >
              <Badge variant={complianceMode === "moderate" ? "default" : "outline"} className="mb-2">
                Moderate
              </Badge>
              <p className="text-sm text-muted-foreground">Balanced privacy for general compliance use</p>
            </button>

            <button
              onClick={() => setComplianceMode("custom")}
              className={`p-4 rounded-lg border-2 transition-all text-left ${
                complianceMode === "custom" ? "border-primary bg-primary/5" : "border-border hover:border-primary/50"
              }`}
            >
              <Badge variant={complianceMode === "custom" ? "default" : "outline"} className="mb-2">
                Custom
              </Badge>
              <p className="text-sm text-muted-foreground">Custom privacy configuration</p>
            </button>
          </div>
        </CardContent>
      </Card>

      {/* File Upload */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Upload className="size-5 text-primary" />
            <CardTitle>File Upload</CardTitle>
          </div>
          <CardDescription>Upload images or documents for privacy-safe processing</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Drop Zone */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-xl p-12 text-center transition-all cursor-pointer ${
              isDragging ? "border-primary bg-primary/5 scale-[1.02]" : "border-border hover:border-primary/50"
            }`}
          >
            <input
              type="file"
              multiple
              accept="image/*"
              onChange={handleFileInput}
              className="hidden"
              id="file-input"
            />
            <label htmlFor="file-input" className="cursor-pointer">
              <div className="flex flex-col items-center gap-4">
                <div className="size-16 rounded-2xl bg-primary/10 flex items-center justify-center">
                  <FileImage className="size-8 text-primary" />
                </div>
                <div>
                  <p className="text-lg font-medium mb-1">Drag files here or click to select</p>
                  <p className="text-sm text-muted-foreground">Supports: JPG, PNG, PDF, TIFF (Max 50MB per file)</p>
                </div>
              </div>
            </label>
          </div>

          {/* Selected Files */}
          {selectedFiles.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-base">Selected Files ({selectedFiles.length})</Label>
                <Button variant="ghost" size="sm" onClick={() => setSelectedFiles([])}>
                  Clear All
                </Button>
              </div>

              <div className="space-y-2">
                {selectedFiles.map((file, index) => (
                  <div
                    key={index}
                    className="flex items-center gap-4 p-4 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors"
                  >
                    <FileImage className="size-8 text-primary shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="font-medium truncate">{file.name}</p>
                      <p className="text-sm text-muted-foreground">{formatBytes(file.size)}</p>
                    </div>
                    <Button variant="ghost" size="icon" onClick={() => removeFile(index)}>
                      <X className="size-4" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Process Button */}
          <div className="flex items-center gap-4 pt-4">
            <Button
              size="lg"
              className="flex-1 h-12 gap-2 shadow-lg shadow-primary/20"
              disabled={selectedFiles.length === 0}
            >
              <Upload className="size-4" />
              Process {selectedFiles.length} {selectedFiles.length === 1 ? "File" : "Files"}
            </Button>
            <Button variant="outline" size="lg" className="h-12 bg-transparent">
              <Info className="size-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
