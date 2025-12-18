"use client"

import { MetricCard } from "@/components/metric-card"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { dashboardMetrics, recentUploads } from "@/lib/mock-data"
import { Shield, AlertTriangle, Eye, Activity } from "lucide-react"

export default function InsightsPage() {
  // ---- Derived insights (purely from existing mock data) ----
  const totalUploads = recentUploads.length

  const strictUsageCount = recentUploads.filter(
    (u) => u.complianceMode === "Strict"
  ).length

  const strictUsagePercent =
    totalUploads > 0 ? Math.round((strictUsageCount / totalUploads) * 100) : 0

  const mostDetectedType = Object.entries(dashboardMetrics.detectionBreakdown).sort(
    (a, b) => b[1] - a[1]
  )[0]

  const highRiskUploads = recentUploads.filter(
    (u) => u.complianceMode === "Strict" && u.detections > 10
  ).length

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-semibold tracking-tight text-white">
          Privacy Intelligence
        </h1>
        <p className="text-zinc-400 mt-1">
          Insights into visual privacy risk and compliance posture
        </p>
      </div>

      {/* Top Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="High-Risk Uploads"
          value={highRiskUploads}
          subtitle="Strict mode with elevated detections"
          icon={AlertTriangle}
        />

        <MetricCard
          title="Most Detected Privacy Type"
          value={mostDetectedType?.[0] ?? "N/A"}
          subtitle="Across all processed uploads"
          icon={Eye}
        />

        <MetricCard
          title="Strict Mode Usage"
          value={`${strictUsagePercent}%`}
          subtitle="Of total uploads"
          icon={Shield}
        />

        <MetricCard
          title="Compliance Coverage"
          value={`${dashboardMetrics.complianceScore}%`}
          subtitle="Overall privacy posture"
          icon={Activity}
        />
      </div>

      {/* Risk Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Risk Distribution Summary</CardTitle>
          <CardDescription>
            Breakdown of detected privacy-sensitive content
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="p-4 rounded-lg border border-border bg-muted/30">
              <p className="text-sm text-muted-foreground mb-1">Faces</p>
              <p className="text-xl font-semibold">
                {dashboardMetrics.detectionBreakdown.faces.toLocaleString()}
              </p>
            </div>

            <div className="p-4 rounded-lg border border-border bg-muted/30">
              <p className="text-sm text-muted-foreground mb-1">Text</p>
              <p className="text-xl font-semibold">
                {dashboardMetrics.detectionBreakdown.text.toLocaleString()}
              </p>
            </div>

            <div className="p-4 rounded-lg border border-border bg-muted/30">
              <p className="text-sm text-muted-foreground mb-1">Medical</p>
              <p className="text-xl font-semibold">
                {dashboardMetrics.detectionBreakdown.medical.toLocaleString()}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Compliance Posture */}
      <Card>
        <CardHeader>
          <CardTitle>Compliance Posture</CardTitle>
          <CardDescription>
            Current regulatory alignment based on processing history
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline" className="border-green-500 text-green-400">
              GDPR aligned
            </Badge>
            <Badge variant="outline" className="border-green-500 text-green-400">
              HIPAA-ready
            </Badge>
            <Badge variant="outline" className="border-green-500 text-green-400">
              DPDP compliant
            </Badge>
          </div>

          <p className="text-sm text-muted-foreground">
            Compliance posture is inferred from anonymization configuration,
            detection coverage, and historical processing behavior.
          </p>
        </CardContent>
      </Card>

      {/* Recent Changes */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Privacy Trends</CardTitle>
          <CardDescription>
            Notable changes observed in recent processing activity
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li>• Increased use of Strict compliance mode in recent uploads</li>
            <li>• Higher frequency of medical data detection</li>
            <li>• Stable overall processing and compliance coverage</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
