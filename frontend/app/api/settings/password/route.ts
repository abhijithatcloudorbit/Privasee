import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { currentPassword, newPassword } = body

    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 600))

    // Basic validation
    if (!currentPassword || !newPassword) {
      return NextResponse.json({ success: false, error: "All fields are required" }, { status: 400 })
    }

    // Mock password validation
    if (currentPassword !== "demo123") {
      return NextResponse.json({ success: false, error: "Current password is incorrect" }, { status: 401 })
    }

    return NextResponse.json(
      {
        success: true,
        message: "Password changed successfully",
      },
      { status: 200 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to change password" }, { status: 500 })
  }
}
