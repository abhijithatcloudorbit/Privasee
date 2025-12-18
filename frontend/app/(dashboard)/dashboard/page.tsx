import { MetricCard } from "@/components/metric-card"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { dashboardMetrics, recentUploads } from "@/lib/mock-data"
import {
  Upload,
  Shield,
  Activity,
  Zap,
  Eye,
  User,
  FileText,
  Pill,
  Car,
  ArrowRight,
  CheckCircle2,
  Loader2,
  XCircle,
} from "lucide-react"
import Link from "next/link"

export default function DashboardPage() {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="size-4 text-green-500" />
      case "processing":
        return <Loader2 className="size-4 text-cyan-400 animate-spin" />
      case "failed":
        return <XCircle className="size-4 text-red-500" />
      default:
        return null
    }
  }

  const formatBytes = (bytes: number) => {
    return (bytes / (1024 * 1024)).toFixed(2) + " MB"
  }

  const formatTime = (date: Date) => {
    const now = Date.now()
    const diff = now - date.getTime()
    const minutes = Math.floor(diff / 60000)
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h ago`
    return `${Math.floor(hours / 24)}d ago`
  }

  return (
    <main className="p-6 lg:p-8">
      <div className="max-w-400 mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl lg:text-4xl font-semibold tracking-tight text-balance text-white">
              Privasee Dashboard
            </h1>
            <p className="text-zinc-400 mt-1 text-sm lg:text-base">
              Monitoring AI-based image anonymization and compliance status
            </p>
          </div>
          <Link href="/upload">
            <Button size="lg" className="gap-2 shadow-none bg-cyan-500 hover:bg-cyan-600 text-black">
              <Upload className="size-4" />
              <span className="hidden sm:inline">Upload File</span>
            </Button>
          </Link>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
          <MetricCard
            title="Files Processed"
            value={dashboardMetrics.totalUploads.toLocaleString()}
            subtitle="All time"
            icon={Upload}
            trend={{ value: 12.5, isPositive: true }}
          />
          <MetricCard
            title="Detections by Privasee"
            value={dashboardMetrics.totalDetections.toLocaleString()}
            subtitle="Objects anonymized"
            icon={Eye}
            trend={{ value: 8.3, isPositive: true }}
          />
          <MetricCard
            title="Active Jobs"
            value={dashboardMetrics.activeProcessing}
            subtitle="Active jobs"
            icon={Activity}
          />
          <MetricCard
            title="Compliance Coverage"
            value={`${dashboardMetrics.complianceScore}%`}
            subtitle="Privacy rating"
            icon={Shield}
            trend={{ value: 2.1, isPositive: true }}
          />
          <MetricCard
            title="Avg Processing Time"
            value={`${dashboardMetrics.avgProcessingTime}s`}
            subtitle="Per upload"
            icon={Zap}
          />
        </div>

        {/* Detection Breakdown */}
        <Card className="border-zinc-800 bg-zinc-950">
          <CardHeader>
            <CardTitle className="text-white">Sensitive Data Detection Breakdown</CardTitle>
            <CardDescription>Total sensitive content detected across all uploads</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
              <div className="flex items-center gap-3 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800">
                <div className="size-10 rounded-lg bg-cyan-500/10 flex items-center justify-center shrink-0">
                  <User className="size-5 text-cyan-400" />
                </div>
                <div>
                  <p className="text-xl font-semibold text-white">
                    {dashboardMetrics.detectionBreakdown.faces.toLocaleString()}
                  </p>
                  <p className="text-xs text-zinc-400">Faces</p>
                </div>
              </div>

              <div className="flex items-center gap-3 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800">
                <div className="size-10 rounded-lg bg-cyan-500/10 flex items-center justify-center shrink-0">
                  <User className="size-5 text-cyan-400" />
                </div>
                <div>
                  <p className="text-xl font-semibold text-white">
                    {dashboardMetrics.detectionBreakdown.people.toLocaleString()}
                  </p>
                  <p className="text-xs text-zinc-400">People</p>
                </div>
              </div>

              <div className="flex items-center gap-3 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800">
                <div className="size-10 rounded-lg bg-cyan-500/10 flex items-center justify-center shrink-0">
                  <FileText className="size-5 text-cyan-400" />
                </div>
                <div>
                  <p className="text-xl font-semibold text-white">
                    {dashboardMetrics.detectionBreakdown.text.toLocaleString()}
                  </p>
                  <p className="text-xs text-zinc-400">Text</p>
                </div>
              </div>

              <div className="flex items-center gap-3 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800">
                <div className="size-10 rounded-lg bg-cyan-500/10 flex items-center justify-center shrink-0">
                  <Pill className="size-5 text-cyan-400" />
                </div>
                <div>
                  <p className="text-xl font-semibold text-white">
                    {dashboardMetrics.detectionBreakdown.medical.toLocaleString()}
                  </p>
                  <p className="text-xs text-zinc-400">Medical</p>
                </div>
              </div>

              <div className="flex items-center gap-3 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800">
                <div className="size-10 rounded-lg bg-cyan-500/10 flex items-center justify-center shrink-0">
                  <Car className="size-5 text-cyan-400" />
                </div>
                <div>
                  <p className="text-xl font-semibold text-white">
                    {dashboardMetrics.detectionBreakdown.licensePlates.toLocaleString()}
                  </p>
                  <p className="text-xs text-zinc-400">Plates</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Recent Uploads */}
        <Card className="border-zinc-800 bg-zinc-950">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-white">Recent Processing Jobs</CardTitle>
              <CardDescription>Latest image anonymization and compliance checks</CardDescription>
            </div>
            <Link href="/history">
              <Button variant="ghost" size="sm" className="gap-2 text-zinc-400 hover:text-white">
                View All
                <ArrowRight className="size-4" />
              </Button>
            </Link>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {recentUploads.map((upload) => (
                <div
                  key={upload.id}
                  className="flex items-center gap-4 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800 hover:bg-zinc-900/80 transition-colors"
                >
                  <img
                    src={upload.thumbnail || "/placeholder.svg?height=64&width=64"}
                    alt={upload.filename}
                    className="size-14 rounded-lg object-cover bg-zinc-800 shrink-0"
                  />

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <p className="font-medium truncate text-white text-sm">{upload.filename}</p>
                      {getStatusIcon(upload.status)}
                    </div>
                    <div className="flex items-center gap-2 text-xs text-zinc-400 flex-wrap">
                      <span>{formatBytes(upload.size)}</span>
                      <span>•</span>
                      <span>{formatTime(upload.uploadedAt)}</span>
                      <span>•</span>
                      <span className="capitalize">{upload.complianceMode}</span>
                    </div>
                  </div>

                  {upload.status === "completed" && (
                    <div className="text-right shrink-0">
                      <p className="text-lg font-semibold text-white">{upload.detections}</p>
                      <p className="text-xs text-zinc-400">privasee filters</p>
                      <p className="text-xs text-cyan-400 mt-1">{upload.processingTime}s</p>
                    </div>
                  )}

                  {upload.status === "processing" && (
                    <div className="shrink-0">
                      <Badge variant="outline" className="border-cyan-500 text-cyan-400">
                        Processing
                      </Badge>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  )
}
