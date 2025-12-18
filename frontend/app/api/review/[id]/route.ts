import { type NextRequest, NextResponse } from "next/server"
import { reviewResults } from "@/lib/mock-data"

export async function GET(request: NextRequest, { params }: { params: { id: string } }) {
  try {
    const { id } = params

    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 300))

    const result = reviewResults.find((r) => r.id === id)

    if (!result) {
      return NextResponse.json({ success: false, error: "Review result not found" }, { status: 404 })
    }

    return NextResponse.json(
      {
        success: true,
        data: result,
      },
      { status: 200 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to fetch review result" }, { status: 500 })
  }
}

export async function PATCH(request: NextRequest, { params }: { params: { id: string } }) {
  try {
    const { id } = params
    const body = await request.json()
    const { action } = body // 'approve' or 'reject'

    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 500))

    return NextResponse.json(
      {
        success: true,
        message: `Review ${action}d successfully`,
        data: { id, action, timestamp: new Date() },
      },
      { status: 200 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to update review" }, { status: 500 })
  }
}
