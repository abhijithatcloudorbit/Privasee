"use client"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Shield, Download, FileCheck, AlertTriangle, CheckCircle, TrendingUp } from "lucide-react"

export default function CompliancePage() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Compliance Reports</h1>
          <p className="text-muted-foreground mt-2">Monitor compliance scores and generate audit reports</p>
        </div>
        <Button>
          <Download className="size-4 mr-2" />
          Export Report
        </Button>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        <Card className="border-primary/20 bg-primary/5">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardDescription>Overall Compliance</CardDescription>
              <Shield className="size-5 text-primary" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-primary">98.7%</div>
            <p className="text-xs text-muted-foreground mt-1 flex items-center gap-1">
              <TrendingUp className="size-3 text-green-500" />
              <span className="text-green-500">+2.1%</span> from last month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardDescription>Passed Audits</CardDescription>
              <CheckCircle className="size-5 text-green-500" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">1,247</div>
            <p className="text-xs text-muted-foreground mt-1">All time uploads</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardDescription>Flagged Items</CardDescription>
              <AlertTriangle className="size-5 text-amber-500" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">16</div>
            <p className="text-xs text-muted-foreground mt-1">Requires manual review</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Compliance Standards</CardTitle>
          <CardDescription>Your adherence to privacy regulations and standards</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileCheck className="size-4 text-primary" />
                <span className="font-medium">GDPR Compliance</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-sm text-muted-foreground">99.2%</span>
                <Badge className="bg-green-500/20 text-green-500 hover:bg-green-500/30">Excellent</Badge>
              </div>
            </div>
            <Progress value={99.2} className="h-2" />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileCheck className="size-4 text-primary" />
                <span className="font-medium">HIPAA Compliance</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-sm text-muted-foreground">98.5%</span>
                <Badge className="bg-green-500/20 text-green-500 hover:bg-green-500/30">Excellent</Badge>
              </div>
            </div>
            <Progress value={98.5} className="h-2" />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileCheck className="size-4 text-primary" />
                <span className="font-medium">CCPA Compliance</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-sm text-muted-foreground">97.8%</span>
                <Badge className="bg-green-500/20 text-green-500 hover:bg-green-500/30">Good</Badge>
              </div>
            </div>
            <Progress value={97.8} className="h-2" />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileCheck className="size-4 text-primary" />
                <span className="font-medium">ISO 27001 Standards</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-sm text-muted-foreground">96.4%</span>
                <Badge className="bg-amber-500/20 text-amber-500 hover:bg-amber-500/30">Good</Badge>
              </div>
            </div>
            <Progress value={96.4} className="h-2" />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recent Audit Logs</CardTitle>
          <CardDescription>Latest compliance audit activity</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[
              { action: "Batch anonymization completed", timestamp: "2 hours ago", status: "success" },
              { action: "Manual review requested", timestamp: "5 hours ago", status: "warning" },
              { action: "Compliance report generated", timestamp: "1 day ago", status: "success" },
              { action: "GDPR audit passed", timestamp: "2 days ago", status: "success" },
              { action: "Policy update applied", timestamp: "3 days ago", status: "info" },
            ].map((log, index) => (
              <div key={index} className="flex items-center justify-between border-b border-border pb-3 last:border-0">
                <div className="flex items-center gap-3">
                  <div
                    className={`size-2 rounded-full ${
                      log.status === "success"
                        ? "bg-green-500"
                        : log.status === "warning"
                          ? "bg-amber-500"
                          : "bg-cyan-500"
                    }`}
                  />
                  <span className="text-sm">{log.action}</span>
                </div>
                <span className="text-xs text-muted-foreground">{log.timestamp}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
