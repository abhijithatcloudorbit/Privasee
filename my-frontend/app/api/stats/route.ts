import { NextResponse } from "next/server"
import { dashboardMetrics } from "@/lib/mock-data"

export async function GET() {
  try {
    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 300))

    return NextResponse.json(
      {
        success: true,
        data: dashboardMetrics,
      },
      { status: 200 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to fetch stats" }, { status: 500 })
  }
}
