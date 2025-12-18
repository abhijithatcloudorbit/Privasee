import { NextResponse } from "next/server"

export async function GET() {
  try {
    // Simulate network delay
    await new Promise((resolve) => setTimeout(resolve, 400))

    const complianceData = {
      overallScore: 98.7,
      standards: [
        { name: "GDPR", score: 99.2, status: "compliant" },
        { name: "HIPAA", score: 98.5, status: "compliant" },
        { name: "CCPA", score: 97.8, status: "compliant" },
        { name: "ISO 27001", score: 99.0, status: "compliant" },
      ],
      auditLogs: [
        {
          id: "audit-1",
          action: "Data Anonymization",
          user: "Arjun Patel",
          timestamp: new Date(Date.now() - 1000 * 60 * 15),
          status: "success",
          details: "Processed 12 files with strict compliance mode",
        },
        {
          id: "audit-2",
          action: "Policy Update",
          user: "Priya Sharma",
          timestamp: new Date(Date.now() - 1000 * 60 * 60),
          status: "success",
          details: "Updated face detection sensitivity threshold",
        },
        {
          id: "audit-3",
          action: "Export Report",
          user: "Rahul Kumar",
          timestamp: new Date(Date.now() - 1000 * 60 * 60 * 3),
          status: "success",
          details: "Generated compliance report for Q4 2024",
        },
      ],
      reports: [
        {
          id: "report-1",
          name: "Monthly Compliance Report - December 2024",
          generatedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7),
          type: "monthly",
          format: "PDF",
        },
        {
          id: "report-2",
          name: "GDPR Audit Report - Q4 2024",
          generatedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 15),
          type: "audit",
          format: "PDF",
        },
        {
          id: "report-3",
          name: "HIPAA Compliance Certificate",
          generatedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 30),
          type: "certificate",
          format: "PDF",
        },
      ],
    }

    return NextResponse.json(
      {
        success: true,
        data: complianceData,
      },
      { status: 200 },
    )
  } catch (error) {
    return NextResponse.json({ success: false, error: "Failed to fetch compliance data" }, { status: 500 })
  }
}
