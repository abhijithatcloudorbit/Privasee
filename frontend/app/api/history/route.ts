import { NextResponse } from "next/server"
import { mockUploads } from "@/lib/mock-data"

export async function GET() {
  try {
    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 400))

    // Add more historical data
    const historicalData = [
      ...mockUploads,
      {
        id: "hist-1",
        filename: "pune_office_cctv.mp4",
        size: 12500000,
        uploadedAt: new Date(Date.now() - 1000 * 60 * 60 * 2),
        status: "completed" as const,
        detections: 34,
        processingTime: 5.2,
        thumbnail: "/placeholder.svg?height=200&width=200",
        complianceMode: "Strict Mode",
      },
      {
        id: "hist-2",
        filename: "kolkata_hospital_scan.pdf",
        size: 890000,
        uploadedAt: new Date(Date.now() - 1000 * 60 * 60 * 5),
        status: "completed" as const,
        detections: 15,
        processingTime: 3.1,
        thumbnail: "/placeholder.svg?height=200&width=200",
        complianceMode: "Moderate Mode",
      },
      {
        id: "hist-3",
        filename: "chennai_street_view.jpg",
        size: 4200000,
        uploadedAt: new Date(Date.now() - 1000 * 60 * 60 * 24),
        status: "completed" as const,
        detections: 28,
        processingTime: 4.5,
        thumbnail: "/placeholder.svg?height=200&width=200",
        complianceMode: "Custom Mode",
      },
    ]

    return NextResponse.json(
      {
        success: true,
        data: historicalData,
        total: historicalData.length,
      },
      { status: 200 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to fetch history" }, { status: 500 })
  }
}
