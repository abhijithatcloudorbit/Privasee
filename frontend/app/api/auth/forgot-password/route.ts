import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { email } = body

    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 600))

    if (!email) {
      return NextResponse.json({ success: false, error: "Email is required" }, { status: 400 })
    }

    // Mock successful password reset email
    return NextResponse.json(
      {
        success: true,
        message: "Password reset instructions sent to your email",
      },
      { status: 200 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "Password reset failed" }, { status: 500 })
  }
}
