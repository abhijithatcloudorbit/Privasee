"use client"

import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Shield,
  Upload,
  Eye,
  CheckCircle2,
  Lock,
  FileCheck,
  ArrowRight,
} from "lucide-react"

export default function TrustPage() {
  return (
    <main className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-6 lg:px-8 py-10 space-y-12">

        {/* HERO */}
        <section className="space-y-6">
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline">Privacy by Design</Badge>
            <Badge variant="outline">Enterprise Ready</Badge>
            <Badge variant="outline">Compliance Focused</Badge>
          </div>

          <h1 className="text-4xl lg:text-5xl font-semibold tracking-tight text-balance">
            Protect Sensitive Visual Data <br className="hidden sm:block" />
            <span className="text-muted-foreground">
              before it creates risk
            </span>
          </h1>

          <p className="text-lg text-muted-foreground max-w-2xl">
            Privasee automatically detects and anonymizes privacy-sensitive elements
            in images and documents, enabling organizations to safely use and share
            visual data in regulated environments.
          </p>

          {/* Primary CTAs */}
          <div className="flex flex-wrap gap-4 pt-2">
            <Link href="/dashboard?demo=true">
            <Button size="lg" className="gap-2">
                View Live Demo
                <ArrowRight className="size-4" />
            </Button>
            </Link>

            <Link href="/insights">
              <Button size="lg" variant="outline" className="gap-2">
                See Privacy Intelligence
                <Eye className="size-4" />
              </Button>
            </Link>
          </div>
        </section>

        {/* HOW IT WORKS */}
        <section className="space-y-6">
          <h2 className="text-2xl font-semibold">How Privasee Works</h2>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader>
                <Upload className="size-6 text-primary" />
                <CardTitle>Upload</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Upload images or documents securely for processing.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <Eye className="size-6 text-primary" />
                <CardTitle>Detect</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Identify faces, text, medical data, and visual identifiers.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <Shield className="size-6 text-primary" />
                <CardTitle>Anonymize</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Apply privacy controls based on compliance requirements.
                </CardDescription>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CheckCircle2 className="size-6 text-primary" />
                <CardTitle>Deliver</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Receive privacy-safe outputs ready for sharing or analysis.
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* TRUST & DATA HANDLING */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <Lock className="size-6 text-primary" />
              <CardTitle>What We Process</CardTitle>
              <CardDescription>
                Privasee detects only what is required to protect privacy
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p>• Human faces and individuals</p>
              <p>• Textual identifiers and documents</p>
              <p>• Medical and clinical information</p>
              <p>• Vehicle identifiers (e.g., license plates)</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <FileCheck className="size-6 text-primary" />
              <CardTitle>What We Don’t Do</CardTitle>
              <CardDescription>
                Clear boundaries to protect customer trust
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p>• We don’t retain raw files after processing</p>
              <p>• We don’t use customer data to train models</p>
              <p>• We don’t share data with third parties</p>
              <p>• We don’t expose unmasked outputs</p>
            </CardContent>
          </Card>
        </section>

        {/* COMPLIANCE */}
        <section className="space-y-4">
          <h2 className="text-2xl font-semibold">Designed for Regulated Environments</h2>

          <div className="flex flex-wrap gap-2">
            <Badge variant="outline">GDPR</Badge>
            <Badge variant="outline">HIPAA</Badge>
            <Badge variant="outline">DPDP Act (India)</Badge>
          </div>

          <p className="text-muted-foreground max-w-2xl">
            Privasee supports privacy-first workflows for organizations operating
            under global and regional data protection regulations.
            Compliance posture is reinforced through anonymization, auditability,
            and controlled data handling.
          </p>
        </section>

        {/* FINAL CTA */}
        <section className="border-t border-border pt-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
          <div>
            <h3 className="text-xl font-semibold">See Privasee in Action</h3>
            <p className="text-muted-foreground">
              Explore the demo experience or review privacy insights.
            </p>
          </div>

          <div className="flex gap-3">
            <Link href="/dashboard?demo=true">
                <Button size="lg">Launch Demo</Button>
            </Link>
            <Link href="/upload">
              <Button size="lg" variant="outline">
                Try Upload Flow
              </Button>
            </Link>
          </div>
        </section>

      </div>
    </main>
  )
}
