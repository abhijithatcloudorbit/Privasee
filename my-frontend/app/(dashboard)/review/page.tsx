"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { reviewResults } from "@/lib/mock-data"
import { Check, X, Eye, EyeOff, Download, RotateCcw } from "lucide-react"

export default function ReviewPage() {
  const [currentResult] = useState(reviewResults[0])
  const [sliderPosition, setSliderPosition] = useState(50)
  const [blurIntensity, setBlurIntensity] = useState(80)
  const [showDetections, setShowDetections] = useState(true)
  const [hiddenDetections, setHiddenDetections] = useState<Set<string>>(new Set())

  const toggleDetection = (id: string) => {
    const newHidden = new Set(hiddenDetections)
    if (newHidden.has(id)) {
      newHidden.delete(id)
    } else {
      newHidden.add(id)
    }
    setHiddenDetections(newHidden)
  }

  const getDetectionColor = (type: string) => {
    switch (type) {
      case "face":
        return "bg-cyan-500"
      case "person":
        return "bg-blue-500"
      case "text":
        return "bg-purple-500"
      case "medical":
        return "bg-red-500"
      case "license-plate":
        return "bg-yellow-500"
      default:
        return "bg-gray-500"
    }
  }

  const getDetectionLabel = (type: string) => {
    switch (type) {
      case "face":
        return "Face"
      case "person":
        return "Person"
      case "text":
        return "Text"
      case "medical":
        return "Medical"
      case "license-plate":
        return "License Plate"
      default:
        return type
    }
  }

  const visibleDetections = currentResult.detections.filter((d) => !hiddenDetections.has(d.id))

  return (
    <div className="min-h-screen p-4 md:p-6 lg:p-8">
      <div className="mb-6">
        <h1 className="text-3xl font-semibold text-white mb-2">Result Review & Preview</h1>
        <p className="text-zinc-400">Review detected sensitive content and adjust anonymization settings</p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Main Preview Area */}
        <div className="xl:col-span-2 space-y-4">
          {/* Before/After Comparison */}
          <Card className="bg-zinc-900 border-zinc-800 p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-semibold text-white">{currentResult.filename}</h2>
                <p className="text-sm text-zinc-400">
                  {currentResult.detections.length} detections • {currentResult.processingTime}s processing time
                </p>
              </div>
              <Badge variant="outline" className="border-cyan-500/50 text-cyan-500">
                {currentResult.complianceMode}
              </Badge>
            </div>

            {/* Image Comparison Slider */}
            <div className="relative aspect-video bg-black rounded-lg overflow-hidden mb-4">
              <div className="absolute inset-0">
                <img
                  src={currentResult.originalUrl || "/placeholder.svg"}
                  alt="Original"
                  className="w-full h-full object-cover"
                />

                {/* Detection Overlays on Original */}
                {showDetections &&
                  visibleDetections.map((detection) => (
                    <div
                      key={detection.id}
                      className={`absolute border-2 ${getDetectionColor(detection.type)} ${getDetectionColor(detection.type).replace("bg-", "border-")}`}
                      style={{
                        left: `${detection.boundingBox.x}%`,
                        top: `${detection.boundingBox.y}%`,
                        width: `${detection.boundingBox.width}%`,
                        height: `${detection.boundingBox.height}%`,
                      }}
                    >
                      <div
                        className={`absolute -top-6 left-0 px-2 py-1 rounded text-xs font-medium text-white ${getDetectionColor(detection.type)}`}
                      >
                        {getDetectionLabel(detection.type)} {detection.confidence}%
                      </div>
                    </div>
                  ))}
              </div>

              {/* Processed Image Overlay */}
              <div
                className="absolute inset-0 overflow-hidden"
                style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
              >
                <img
                  src={currentResult.processedUrl || "/placeholder.svg"}
                  alt="Processed"
                  className="w-full h-full object-cover"
                  style={{ filter: `blur(${blurIntensity / 10}px)` }}
                />
              </div>

              {/* Slider Handle */}
              <div
                className="absolute top-0 bottom-0 w-1 bg-cyan-500 cursor-ew-resize"
                style={{ left: `${sliderPosition}%` }}
              >
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-cyan-500 rounded-full flex items-center justify-center shadow-lg">
                  <div className="w-4 h-4 border-2 border-white rounded-full" />
                </div>
              </div>

              {/* Labels */}
              <div className="absolute bottom-4 left-4 px-3 py-1 bg-black/70 rounded text-sm font-medium text-white">
                Original
              </div>
              <div className="absolute bottom-4 right-4 px-3 py-1 bg-black/70 rounded text-sm font-medium text-cyan-500">
                Processed
              </div>
            </div>

            {/* Comparison Slider */}
            <div className="space-y-2">
              <label className="text-sm text-zinc-400">Before/After Comparison</label>
              <Slider
                value={[sliderPosition]}
                onValueChange={(value) => setSliderPosition(value[0])}
                max={100}
                step={1}
                className="w-full"
              />
            </div>
          </Card>

          {/* Blur Intensity Control */}
          <Card className="bg-zinc-900 border-zinc-800 p-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-white">Blur Intensity</label>
                <span className="text-sm text-cyan-500">{blurIntensity}%</span>
              </div>
              <Slider
                value={[blurIntensity]}
                onValueChange={(value) => setBlurIntensity(value[0])}
                max={100}
                step={1}
                className="w-full"
              />
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setBlurIntensity(40)}
                  className="border-zinc-700 text-zinc-300 hover:bg-zinc-800"
                >
                  Low
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setBlurIntensity(70)}
                  className="border-zinc-700 text-zinc-300 hover:bg-zinc-800"
                >
                  Medium
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setBlurIntensity(100)}
                  className="border-zinc-700 text-zinc-300 hover:bg-zinc-800"
                >
                  High
                </Button>
              </div>
            </div>
          </Card>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-3">
            <Button className="bg-green-600 hover:bg-green-700 text-white">
              <Check className="w-4 h-4 mr-2" />
              Approve & Export
            </Button>
            <Button variant="outline" className="border-red-600 text-red-600 hover:bg-red-600/10 bg-transparent">
              <X className="w-4 h-4 mr-2" />
              Reject
            </Button>
            <Button variant="outline" className="border-zinc-700 text-zinc-300 hover:bg-zinc-800 bg-transparent">
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset
            </Button>
            <Button
              variant="outline"
              className="border-zinc-700 text-zinc-300 hover:bg-zinc-800 ml-auto bg-transparent"
            >
              <Download className="w-4 h-4 mr-2" />
              Download
            </Button>
          </div>
        </div>

        {/* Detection Management Sidebar */}
        <div className="space-y-4">
          {/* Detection Toggle */}
          <Card className="bg-zinc-900 border-zinc-800 p-4">
            <Button
              variant="outline"
              className="w-full border-zinc-700 text-zinc-300 hover:bg-zinc-800 bg-transparent"
              onClick={() => setShowDetections(!showDetections)}
            >
              {showDetections ? (
                <>
                  <EyeOff className="w-4 h-4 mr-2" />
                  Hide Detections
                </>
              ) : (
                <>
                  <Eye className="w-4 h-4 mr-2" />
                  Show Detections
                </>
              )}
            </Button>
          </Card>

          {/* Detections List */}
          <Card className="bg-zinc-900 border-zinc-800 p-4">
            <h3 className="text-sm font-semibold text-white mb-4">
              Detected Objects ({visibleDetections.length}/{currentResult.detections.length})
            </h3>
            <div className="space-y-2">
              {currentResult.detections.map((detection) => {
                const isHidden = hiddenDetections.has(detection.id)
                return (
                  <div
                    key={detection.id}
                    className={`p-3 rounded-lg border transition-all ${
                      isHidden ? "bg-zinc-950 border-zinc-800 opacity-50" : "bg-zinc-800 border-zinc-700"
                    }`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div className={`w-3 h-3 rounded ${getDetectionColor(detection.type)}`} />
                        <span className="text-sm font-medium text-white">{getDetectionLabel(detection.type)}</span>
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => toggleDetection(detection.id)}
                        className="h-6 px-2 text-xs"
                      >
                        {isHidden ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                      </Button>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-zinc-400">Confidence</span>
                      <span className="text-cyan-500 font-medium">{detection.confidence}%</span>
                    </div>
                    <div className="mt-2 h-1 bg-zinc-950 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${getDetectionColor(detection.type)}`}
                        style={{ width: `${detection.confidence}%` }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          </Card>

          {/* Statistics */}
          <Card className="bg-zinc-900 border-zinc-800 p-4">
            <h3 className="text-sm font-semibold text-white mb-3">Detection Summary</h3>
            <div className="space-y-2">
              {[
                {
                  label: "Faces",
                  count: currentResult.detections.filter((d) => d.type === "face").length,
                  color: "bg-cyan-500",
                },
                {
                  label: "People",
                  count: currentResult.detections.filter((d) => d.type === "person").length,
                  color: "bg-blue-500",
                },
                {
                  label: "Text",
                  count: currentResult.detections.filter((d) => d.type === "text").length,
                  color: "bg-purple-500",
                },
                {
                  label: "License Plates",
                  count: currentResult.detections.filter((d) => d.type === "license-plate").length,
                  color: "bg-yellow-500",
                },
              ].map((item) => (
                <div key={item.label} className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded ${item.color}`} />
                    <span className="text-zinc-400">{item.label}</span>
                  </div>
                  <span className="text-white font-medium">{item.count}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
