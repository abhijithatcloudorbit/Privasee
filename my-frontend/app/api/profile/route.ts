import { type NextRequest, NextResponse } from "next/server"
import { DEMO_CREDENTIALS } from "@/lib/mock-data"

export async function GET() {
  try {
    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 300))

    return NextResponse.json(
      {
        success: true,
        data: DEMO_CREDENTIALS.user,
      },
      { status: 200 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to fetch profile" }, { status: 500 })
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const body = await request.json()

    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 500))

    // Mock successful profile update
    return NextResponse.json(
      {
        success: true,
        data: { ...DEMO_CREDENTIALS.user, ...body },
        message: "Profile updated successfully",
      },
      { status: 200 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to update profile" }, { status: 500 })
  }
}
