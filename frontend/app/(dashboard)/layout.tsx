"use client"

import type React from "react"
import { useEffect, useState } from "react"
import { useRouter, usePathname } from "next/navigation"
import { SidebarNav } from "@/components/sidebar-nav"

import { ToastProvider } from "@/components/ui/use-toast"
import { Toaster } from "@/components/ui/toaster"

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const router = useRouter()
  const pathname = usePathname()

  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isChecking, setIsChecking] = useState(true)
  const [isCollapsed, setIsCollapsed] = useState(false)

  // Client-side auth guard (mock auth for MVP)
  useEffect(() => {
    const user = localStorage.getItem("user")
    const isDemo = window.location.search.includes("demo=true")

    if (!user && !isDemo) {
      router.push("/login")
    } else {
      setIsAuthenticated(true)
    }
    setIsChecking(false)
  }, [router, pathname])

  // Sync sidebar collapsed state across tabs
  useEffect(() => {
    const handleStorage = () => {
      const collapsed = localStorage.getItem("sidebarCollapsed") === "true"
      setIsCollapsed(collapsed)
    }

    handleStorage()
    window.addEventListener("storage", handleStorage)
    const interval = setInterval(handleStorage, 100)

    return () => {
      window.removeEventListener("storage", handleStorage)
      clearInterval(interval)
    }
  }, [])

  if (isChecking || !isAuthenticated) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="text-center space-y-4">
          <div className="size-8 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-sm text-zinc-500">Verifying session…</p>
        </div>
      </div>
    )
  }

  return (
    <ToastProvider>
      <div className="flex min-h-screen bg-background">
        <SidebarNav />
        <main
          className={`flex-1 w-full min-w-0 p-6 lg:p-8 transition-all duration-300 ${
            isCollapsed ? "lg:ml-16" : "lg:ml-56"
          }`}
        >
          <div className="max-w-7xl mx-auto">{children}</div>
        </main>
      </div>

      {/* Toast renderer */}
      <Toaster />
    </ToastProvider>
  )
}
