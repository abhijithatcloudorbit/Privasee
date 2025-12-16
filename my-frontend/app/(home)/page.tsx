"use client"

import Link from "next/link"
import { Shield, Eye, Lock, Zap, CheckCircle2, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { useRouter } from "next/navigation"

export default function HomePage() {
  const router = useRouter()

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="border-b border-border/40">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="size-10 bg-primary/20 rounded-xl flex items-center justify-center">
              <Shield className="size-5 text-primary" />
            </div>
            <span className="text-xl font-semibold">Privacy Shield</span>
          </div>

          <div className="flex items-center gap-3">
            <Link href="/login">
              <Button variant="ghost">Sign In</Button>
            </Link>
            <Link href="/signup">
              <Button>Get Started</Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-6 py-24 md:py-32">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-8">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20">
                <Zap className="size-4 text-primary" />
                <span className="text-sm font-medium text-primary">
                  AI-Powered Privacy Protection
                </span>
              </div>

              <h1 className="text-5xl md:text-6xl font-semibold tracking-tight text-balance leading-tight">
                Anonymize Sensitive Content{" "}
                <span className="text-primary">Instantly</span>
              </h1>

              <p className="text-xl text-muted-foreground text-balance leading-relaxed">
                Enterprise-grade ML detection for faces, people, text, medical data,
                and license plates. Ensure compliance with automatic blur and
                redaction.
              </p>

              <div className="flex items-center gap-4">
                <Link href="/signup">
                  <Button
                    size="lg"
                    className="h-12 px-8 gap-2 shadow-lg shadow-primary/20"
                  >
                    Start Free Trial
                    <ArrowRight className="size-4" />
                  </Button>
                </Link>

                <Button
                  size="lg"
                  variant="outline"
                  className="h-12 px-8 bg-transparent"
                  onClick={() => router.push("/trust")}
                >
                  View Demo
                </Button>
              </div>

              <div className="flex items-center gap-6 text-sm text-muted-foreground">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="size-4 text-primary" />
                  <span>No credit card required</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="size-4 text-primary" />
                  <span>Enterprise ready</span>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="relative rounded-2xl overflow-hidden border border-border/40 shadow-2xl">
                <img
                  src="/surveillance-camera-footage.jpg"
                  alt="Privacy Shield Dashboard"
                  className="w-full h-auto"
                />
                <div className="absolute inset-0 bg-linear-to-t from-background/80 to-transparent" />
              </div>

              {/* Floating Stats */}
              <div className="absolute -bottom-6 -left-6 bg-card border border-border rounded-xl p-4 shadow-xl">
                <div className="flex items-center gap-3">
                  <div className="size-12 bg-primary/10 rounded-lg flex items-center justify-center">
                    <Eye className="size-5 text-primary" />
                  </div>
                  <div>
                    <p className="text-2xl font-semibold">1.2M+</p>
                    <p className="text-xs text-muted-foreground">Detections</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/40 py-8">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <Shield className="size-4" />
              <span>© 2025 Privacy Shield. All rights reserved.</span>
            </div>
            <div className="flex items-center gap-6">
              <Link href="#" className="hover:text-foreground transition-colors">
                Privacy Policy
              </Link>
              <Link href="#" className="hover:text-foreground transition-colors">
                Terms
              </Link>
              <Link href="#" className="hover:text-foreground transition-colors">
                Contact
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
