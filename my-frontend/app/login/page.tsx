"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import { Shield, Eye, EyeOff } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card } from "@/components/ui/card"
import { DEMO_CREDENTIALS } from "@/lib/mock-data"

export default function LoginPage() {
  const router = useRouter()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setIsLoading(true)

    // Simulate API call
    setTimeout(() => {
      if (email === DEMO_CREDENTIALS.email && password === DEMO_CREDENTIALS.password) {
        localStorage.setItem("user", JSON.stringify(DEMO_CREDENTIALS.user))
        router.push("/dashboard")
      } else {
        setError("Invalid credentials. Use demo@privacyshield.com / demo123")
      }
      setIsLoading(false)
    }, 800)
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <div className="w-full max-w-md space-y-8">
        {/* Logo and Header */}
        <div className="text-center space-y-3">
          <Link href="/" className="inline-flex justify-center">
            <div className="size-16 bg-primary/20 rounded-2xl flex items-center justify-center">
              <Shield className="size-8 text-primary" />
            </div>
          </Link>
          <h1 className="text-3xl font-semibold tracking-tight">Welcome Back</h1>
          <p className="text-muted-foreground">Sign in to Privacy Shield to continue</p>
        </div>

        {/* Demo Credentials Card */}
        <Card className="p-4 bg-primary/5 border-primary/20">
          <p className="text-sm font-medium text-primary mb-2">Demo Credentials</p>
          <div className="space-y-1 text-sm text-muted-foreground">
            <p>
              Email: <span className="font-mono text-foreground">demo@privacyshield.com</span>
            </p>
            <p>
              Password: <span className="font-mono text-foreground">demo123</span>
            </p>
          </div>
        </Card>

        {/* Login Form */}
        <Card className="p-6">
          <form onSubmit={handleLogin} className="space-y-4">
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

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="h-11 pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showPassword ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                </button>
              </div>
            </div>

            {error && (
              <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20">
                <p className="text-sm text-destructive">{error}</p>
              </div>
            )}

            <div className="flex items-center justify-between text-sm">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" className="size-4 rounded border-border" />
                <span className="text-muted-foreground">Remember me</span>
              </label>
              <Link href="/forgot-password" className="text-primary hover:underline">
                Forgot password?
              </Link>
            </div>

            <Button type="submit" className="w-full h-11" disabled={isLoading}>
              {isLoading ? "Signing in..." : "Sign In"}
            </Button>
          </form>
        </Card>

        {/* Sign Up Link */}
        <p className="text-center text-sm text-muted-foreground">
          {"Don't have an account? "}
          <Link href="/signup" className="text-primary hover:underline font-medium">
            Sign up
          </Link>
        </p>
      </div>
    </div>
  )
}
