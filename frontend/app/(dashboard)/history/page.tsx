"use client"

import React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { mockUploads } from "@/lib/mock-data"
import { Clock, Download, Eye, Search } from "lucide-react"

export default function HistoryPage() {
  const [searchQuery, setSearchQuery] = React.useState("")

  const filteredUploads = mockUploads.filter((upload) =>
    upload.filename.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Batch History</h1>
        <p className="text-muted-foreground mt-2">View and manage your anonymization job history</p>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
          <Input
            placeholder="Search by filename..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>
      </div>

      <div className="grid gap-4">
        {filteredUploads.map((upload) => (
          <Card key={upload.id} className="hover:shadow-md transition-shadow">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div className="space-y-1">
                  <CardTitle className="text-lg">{upload.filename}</CardTitle>
                  <CardDescription className="flex items-center gap-2">
                    <Clock className="size-3" />
                    {new Date(upload.uploadedAt).toLocaleString()}
                  </CardDescription>
                </div>
                <Badge
                  variant={
                    upload.status === "completed"
                      ? "default"
                      : upload.status === "processing"
                        ? "secondary"
                        : "destructive"
                  }
                  className={
                    upload.status === "completed"
                      ? "bg-primary/20 text-primary hover:bg-primary/30"
                      : upload.status === "processing"
                        ? "bg-cyan-500/20 text-cyan-500 hover:bg-cyan-500/30"
                        : ""
                  }
                >
                  {upload.status}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-6 text-sm">
                  <div>
                    <span className="text-muted-foreground">Size:</span>{" "}
                    <span className="font-medium">{(upload.size / 1024 / 1024).toFixed(2)} MB</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Compliance:</span>{" "}
                    <span className="font-medium capitalize">{upload.complianceMode}</span>
                  </div>
                  {upload.detections && (
                    <div>
                      <span className="text-muted-foreground">Detections:</span>{" "}
                      <span className="font-medium">{upload.detections}</span>
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm">
                    <Eye className="size-4 mr-2" />
                    View
                  </Button>
                  {upload.status === "completed" && (
                    <Button variant="outline" size="sm">
                      <Download className="size-4 mr-2" />
                      Download
                    </Button>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
