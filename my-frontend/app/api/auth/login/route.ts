import { type NextRequest, NextResponse } from "next/server"
import { DEMO_CREDENTIALS } from "@/lib/mock-data"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { email, password } = body

    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 500))

    // Check credentials
    if (email === DEMO_CREDENTIALS.email && password === DEMO_CREDENTIALS.password) {
      return NextResponse.json(
        {
          success: true,
          user: DEMO_CREDENTIALS.user,
          token: "mock-jwt-token-" + Date.now(),
        },
        { status: 200 },
      )
    }

    return NextResponse.json({ success: false, error: "Invalid credentials" }, { status: 401 })
  } catch (error) {
    return NextResponse.json({ success: false, error: "Login failed" }, { status: 500 })
  }
}
