import { NextResponse } from "next/server"
import { reviewResults } from "@/lib/mock-data"

export async function GET() {
  try {
    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 400))

    return NextResponse.json(
      {
        success: true,
        data: reviewResults,
        total: reviewResults.length,
      },
      { status: 200 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to fetch review results" }, { status: 500 })
  }
}
