import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { email, password, name } = body

    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 800))

    // Basic validation
    if (!email || !password || !name) {
      return NextResponse.json({ success: false, error: "All fields are required" }, { status: 400 })
    }

    // Mock successful signup
    return NextResponse.json(
      {
        success: true,
        user: {
          id: "user-" + Date.now(),
          name,
          email,
          role: "User",
          avatar: name.substring(0, 2).toUpperCase(),
          department: "General",
          joinedDate: new Date(),
        },
        token: "mock-jwt-token-" + Date.now(),
      },
      { status: 201 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "Signup failed" }, { status: 500 })
  }
}
