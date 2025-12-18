"use client"

import type React from "react"

import { useState } from "react"
import Link from "next/link"
import { Shield, ArrowLeft, CheckCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card } from "@/components/ui/card"

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isSuccess, setIsSuccess] = useState(false)

  const handleResetPassword = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    // Simulate API call
    setTimeout(() => {
      setIsSuccess(true)
      setIsLoading(false)
    }, 1500)
  }

  if (isSuccess) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background p-4">
        <div className="w-full max-w-md space-y-8">
          <div className="text-center space-y-3">
            <div className="flex justify-center">
              <div className="size-16 bg-primary/20 rounded-2xl flex items-center justify-center">
                <CheckCircle className="size-8 text-primary" />
              </div>
            </div>
            <h1 className="text-3xl font-semibold tracking-tight">Check Your Email</h1>
            <p className="text-muted-foreground">
              We've sent password reset instructions to <span className="font-medium text-foreground">{email}</span>
            </p>
          </div>

          <Card className="p-6 space-y-4">
            <p className="text-sm text-muted-foreground text-center">
              Didn't receive the email? Check your spam folder or try again.
            </p>
            <Button variant="outline" className="w-full bg-transparent" onClick={() => setIsSuccess(false)}>
              Try Another Email
            </Button>
          </Card>

          <div className="text-center">
            <Link href="/login" className="text-sm text-primary hover:underline inline-flex items-center gap-2">
              <ArrowLeft className="size-4" />
              Back to login
            </Link>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <div className="w-full max-w-md space-y-8">
        {/* Logo and Header */}
        <div className="text-center space-y-3">
          <div className="flex justify-center">
            <div className="size-16 bg-primary/20 rounded-2xl flex items-center justify-center">
              <Shield className="size-8 text-primary" />
            </div>
          </div>
          <h1 className="text-3xl font-semibold tracking-tight">Reset Password</h1>
          <p className="text-muted-foreground">
            Enter your email address and we'll send you instructions to reset your password
          </p>
        </div>

        {/* Reset Form */}
        <Card className="p-6">
          <form onSubmit={handleResetPassword} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email">Email Address</Label>
              <Input
                id="email"
                type="email"
                placeholder="arjun.patel@privacyshield.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="h-11"
              />
            </div>

            <Button type="submit" className="w-full h-11" disabled={isLoading}>
              {isLoading ? "Sending instructions..." : "Send Reset Instructions"}
            </Button>
          </form>
        </Card>

        {/* Back to Login */}
        <div className="text-center">
          <Link href="/login" className="text-sm text-primary hover:underline inline-flex items-center gap-2">
            <ArrowLeft className="size-4" />
            Back to login
          </Link>
        </div>
      </div>
    </div>
  )
}
