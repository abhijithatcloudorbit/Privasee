"use client"

import { useState, useEffect } from "react"
import { processingQueue } from "@/lib/mock-data"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { CheckCircle2, Clock, Loader2, FileImage, Pause, Activity, Zap } from "lucide-react"
import Image from "next/image"

export default function ProcessingPage() {
  const [queue, setQueue] = useState(processingQueue)
  const [isPaused, setIsPaused] = useState(false)

  useEffect(() => {
    if (isPaused) return

    const interval = setInterval(() => {
      setQueue((prev) =>
        prev.map((item) => {
          if (item.status === "completed") return item

          const newProgress = Math.min(item.progress + Math.random() * 3, 100)
          const newStatus = newProgress >= 100 ? "completed" : "processing"

          return {
            ...item,
            progress: newProgress,
            status: newStatus,
            models: item.models.map((model) => {
              if (model.status === "completed") return model
              if (model.status === "pending" && newProgress > 50) {
                return { ...model, status: "processing", confidence: Math.floor(Math.random() * 20 + 80) }
              }
              if (model.status === "processing" && Math.random() > 0.5) {
                return {
                  ...model,
                  status: "completed",
                  confidence: Math.floor(Math.random() * 10 + 90),
                  detections: Math.floor(Math.random() * 15),
                }
              }
              return model
            }),
          }
        }),
      )
    }, 1500)

    return () => clearInterval(interval)
  }, [isPaused])

  const activeJobs = queue.filter((item) => item.status === "processing").length
  const completedJobs = queue.filter((item) => item.status === "completed").length
  const totalDetections = queue.reduce((sum, item) => sum + item.models.reduce((s, m) => s + (m.detections || 0), 0), 0)

  const getModelIcon = (modelName: string) => {
    if (modelName.includes("Face")) return "👤"
    if (modelName.includes("Person")) return "🚶"
    if (modelName.includes("Text")) return "📝"
    if (modelName.includes("Medical")) return "🏥"
    if (modelName.includes("License")) return "🚗"
    return "🔍"
  }

  const getStatusColor = (status: string) => {
    if (status === "completed") return "text-green-400"
    if (status === "processing") return "text-cyan-400"
    return "text-zinc-500"
  }

  const getStatusBadge = (status: string) => {
    if (status === "completed") return "bg-green-500/10 text-green-400 border-green-500/20"
    if (status === "processing") return "bg-cyan-500/10 text-cyan-400 border-cyan-500/20"
    return "bg-zinc-500/10 text-zinc-400 border-zinc-500/20"
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold text-white">Anonymization Queue</h1>
          <p className="text-zinc-400 mt-1">Monitoring privacy anonymization and compliance processing</p>
        </div>
        <div className="flex gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsPaused(!isPaused)}
            className="border-zinc-700 hover:bg-zinc-800"
          >
            {isPaused ? (
              <>
                <Activity className="mr-2 h-4 w-4" />
                Resume Processing
              </>
            ) : (
              <>
                <Pause className="mr-2 h-4 w-4" />
                Pause Processing
              </>
            )}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-zinc-900 border-zinc-800 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-zinc-400">Active Processing Jobs</p>
              <p className="text-3xl font-semibold text-white mt-1">{activeJobs}</p>
            </div>
            <div className="bg-cyan-500/10 p-3 rounded-lg">
              <Loader2 className="h-6 w-6 text-cyan-400 animate-spin" />
            </div>
          </div>
        </Card>

        <Card className="bg-zinc-900 border-zinc-800 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-zinc-400">Completed Jobs</p>
              <p className="text-3xl font-semibold text-white mt-1">{completedJobs}</p>
            </div>
            <div className="bg-green-500/10 p-3 rounded-lg">
              <CheckCircle2 className="h-6 w-6 text-green-400" />
            </div>
          </div>
        </Card>

        <Card className="bg-zinc-900 border-zinc-800 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-zinc-400">otal Privacy Detections</p>
              <p className="text-3xl font-semibold text-white mt-1">{totalDetections}</p>
            </div>
            <div className="bg-purple-500/10 p-3 rounded-lg">
              <Zap className="h-6 w-6 text-purple-400" />
            </div>
          </div>
        </Card>
      </div>

      <div className="space-y-4">
        {queue.map((item) => (
          <Card key={item.id} className="bg-zinc-900 border-zinc-800 p-6 hover:border-zinc-700 transition-colors">
            <div className="space-y-4">
              {/* File header */}
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-4">
                  <div className="relative w-20 h-20 bg-zinc-800 rounded-lg overflow-hidden shrink-0">
                    {item.thumbnail ? (
                      <Image
                        src={item.thumbnail || "/placeholder.svg"}
                        alt={item.filename}
                        fill
                        className="object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <FileImage className="h-8 w-8 text-zinc-600" />
                      </div>
                    )}
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white">{item.filename}</h3>
                    <p className="text-sm text-zinc-400 mt-1">{(item.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Badge className={getStatusBadge(item.status)}>
                    {item.status === "completed" ? (
                      <CheckCircle2 className="mr-1 h-3 w-3" />
                    ) : (
                      <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                    )}
                    {item.status === "completed" ? "Completed" : "Processing"}
                  </Badge>
                  {item.status === "processing" && (
                    <span className="text-sm text-zinc-400">{item.estimatedTime} remaining</span>
                  )}
                </div>
              </div>

              {/* Overall progress */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-zinc-300">Overall Progress</span>
                  <span className="text-sm font-semibold text-white">{Math.floor(item.progress)}%</span>
                </div>
                <Progress value={item.progress} className="h-2" />
              </div>

              {/* Model-specific status grid */}
              <div className="grid grid-cols-1 md:grid-cols-5 gap-3 pt-2">
                {item.models.map((model, idx) => (
                  <div
                    key={idx}
                    className={`bg-zinc-800/50 rounded-lg p-3 border ${
                      model.status === "completed"
                        ? "border-green-500/20"
                        : model.status === "processing"
                          ? "border-cyan-500/20"
                          : "border-zinc-700"
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-lg">{getModelIcon(model.name)}</span>
                      <span
                        className={`h-2 w-2 rounded-full ${
                          model.status === "completed"
                            ? "bg-green-400 animate-pulse"
                            : model.status === "processing"
                              ? "bg-cyan-400 animate-pulse"
                              : "bg-zinc-600"
                        }`}
                      />
                    </div>
                    <p className="text-xs font-medium text-zinc-300 mb-1">{model.name}</p>
                    <div className="flex items-center justify-between">
                      <span className={`text-xs font-semibold ${getStatusColor(model.status)}`}>
                        {model.status === "completed" ? "Done" : model.status === "processing" ? "Active" : "Pending"}
                      </span>
                      {model.confidence > 0 && <span className="text-xs text-zinc-500">{model.confidence}%</span>}
                    </div>
                    {model.detections !== undefined && model.detections > 0 && (
                      <p className="text-xs text-zinc-400 mt-1">{model.detections} detected</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </Card>
        ))}
      </div>

      {queue.length === 0 && (
        <Card className="bg-zinc-900 border-zinc-800 p-12">
          <div className="text-center">
            <Clock className="h-12 w-12 text-zinc-600 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">No Active Processing</h3>
            <p className="text-sm text-zinc-400">Upload files to start processing</p>
          </div>
        </Card>
      )}
    </div>
  )
}
