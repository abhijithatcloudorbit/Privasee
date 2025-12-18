import { type NextRequest, NextResponse } from "next/server"
import { recentUploadsUpdated } from "@/lib/mock-data"

export async function GET() {
  try {
    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 400))

    return NextResponse.json(
      {
        success: true,
        data: recentUploadsUpdated,
        total: recentUploadsUpdated.length,
      },
      { status: 200 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to fetch uploads" }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("file") as File
    const complianceMode = formData.get("complianceMode") as string
    const models = formData.get("models") as string

    // Simulate file upload processing
    await new Promise((resolve) => setTimeout(resolve, 1000))

    if (!file) {
      return NextResponse.json({ success: false, error: "No file provided" }, { status: 400 })
    }

    // Mock successful upload
    const newUpload = {
      id: "upload-" + Date.now(),
      filename: file.name,
      size: file.size,
      uploadedAt: new Date(),
      status: "processing" as const,
      detections: 0,
      thumbnail: "/placeholder.svg?height=200&width=200",
      complianceMode: complianceMode || "Strict Mode",
      models: models ? JSON.parse(models) : [],
    }

    return NextResponse.json(
      {
        success: true,
        data: newUpload,
        message: "File uploaded successfully",
      },
      { status: 201 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "File upload failed" }, { status: 500 })
  }
}
