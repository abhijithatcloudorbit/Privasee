import { NextResponse } from "next/server"
import { processingQueue } from "@/lib/mock-data"

export async function GET() {
  try {
    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 300))

    return NextResponse.json(
      {
        success: true,
        data: processingQueue,
        activeJobs: processingQueue.filter((item) => item.status === "processing").length,
        completedJobs: processingQueue.filter((item) => item.status === "completed").length,
      },
      { status: 200 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to fetch processing queue" }, { status: 500 })
  }
}
