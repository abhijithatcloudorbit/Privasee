import { type NextRequest, NextResponse } from "next/server"
import { detectionModels, complianceModes } from "@/lib/mock-data"

export async function GET() {
  try {
    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 300))

    const settings = {
      detectionModels,
      complianceModes,
      notifications: {
        email: true,
        push: false,
        sms: false,
      },
      preferences: {
        theme: "dark",
        language: "en",
        timezone: "Asia/Kolkata",
      },
    }

    return NextResponse.json(
      {
        success: true,
        data: settings,
      },
      { status: 200 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to fetch settings" }, { status: 500 })
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const body = await request.json()

    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 500))

    return NextResponse.json(
      {
        success: true,
        data: body,
        message: "Settings updated successfully",
      },
      { status: 200 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to update settings" }, { status: 500 })
  }
}
