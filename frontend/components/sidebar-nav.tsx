"use client"

import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { cn } from "@/lib/utils"
import {
  LayoutDashboard,
  Upload,
  Eye,
  Edit3,
  Clock,
  Settings,
  Shield,
  Code,
  LogOut,
  ChevronLeft,
  Menu,
  X,
  Radar,
} from "lucide-react"
import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"

const navItems = [
  { href: "/insights", label: "Privacy Intelligence", icon: Radar },
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/upload", label: "Upload File", icon: Upload },
  { href: "/processing", label: "Anonymization", icon: Eye },
  { href: "/review", label: "Manual Review", icon: Edit3 },
  { href: "/history", label: "Audit History", icon: Clock },
  { href: "/compliance", label: "Compliance Logs", icon: Shield },
  { href: "/api", label: "API", icon: Code },
  { href: "/settings", label: "Settings", icon: Settings },
]

export function SidebarNav() {
  const pathname = usePathname()
  const router = useRouter()
  const [user, setUser] = useState<{ name: string; avatar: string; role: string } | null>(null)
  const [isMobileOpen, setIsMobileOpen] = useState(false)
  const [isCollapsed, setIsCollapsed] = useState(false)

  useEffect(() => {
    const storedUser = localStorage.getItem("user")
    if (storedUser) {
      setUser(JSON.parse(storedUser))
    }
    const collapsed = localStorage.getItem("sidebarCollapsed")
    setIsCollapsed(collapsed === "true")
  }, [])

  useEffect(() => {
    setIsMobileOpen(false)
  }, [pathname])

  const toggleCollapse = () => {
    const newState = !isCollapsed
    setIsCollapsed(newState)
    localStorage.setItem("sidebarCollapsed", String(newState))
  }

  const handleLogout = () => {
    localStorage.removeItem("user")
    router.push("/login")
  }

  return (
    <>
      {/* Mobile Menu Button */}
      <button
        onClick={() => setIsMobileOpen(!isMobileOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 size-10 bg-zinc-900 border border-zinc-800 rounded-lg flex items-center justify-center"
      >
        {isMobileOpen ? <X className="size-5" /> : <Menu className="size-5" />}
      </button>

      {/* Mobile Overlay */}
      {isMobileOpen && (
        <div className="lg:hidden fixed inset-0 bg-black/80 z-40" onClick={() => setIsMobileOpen(false)} />
      )}

      {/* Sidebar - Added dynamic width based on collapsed state */}
      <aside
        className={cn(
          "fixed left-0 top-0 h-screen bg-zinc-950 border-r border-zinc-800 flex flex-col z-40 transition-all duration-300",
          isCollapsed ? "w-16" : "w-56",
          isMobileOpen ? "translate-x-0" : "max-lg:-translate-x-full",
        )}
      >
        {/* Header - Responsive header layout */}
        <div className="h-14 px-3 border-b border-zinc-800 flex items-center gap-2 shrink-0">
          <div className="size-8 bg-cyan-500/20 rounded-lg flex items-center justify-center shrink-0">
            <Shield className="size-4 text-cyan-400" />
          </div>
          {!isCollapsed && (
            <div className="flex-1 min-w-0">
              <h1 className="text-xs font-semibold text-white truncate">Privasee</h1>
              <p className="text-[10px] text-zinc-400 truncate">AI Privacy Platform</p>
            </div>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleCollapse}
            className="hidden lg:flex size-7 text-zinc-400 hover:text-white"
          >
            <ChevronLeft className={cn("size-4 transition-transform", isCollapsed && "rotate-180")} />
          </Button>
        </div>

        {/* Navigation - Centered icons when collapsed */}
        <nav className="flex-1 p-2 space-y-1 overflow-y-auto">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = pathname === item.href

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-2.5 py-2 rounded-lg text-sm font-medium transition-all",
                  isActive ? "bg-cyan-500/20 text-cyan-400" : "text-zinc-400 hover:text-white hover:bg-zinc-900",
                  isCollapsed && "justify-center",
                )}
                title={isCollapsed ? item.label : undefined}
              >
                <Icon className="size-5 shrink-0" />
                {!isCollapsed && <span className="truncate">{item.label}</span>}
              </Link>
            )
          })}
        </nav>

        {/* Footer - Compact footer for collapsed state */}
        <div className="p-2 border-t border-zinc-800 space-y-2 shrink-0">
          {!isCollapsed ? (
            <>
              <Link href="/profile">
                <div className="px-2.5 py-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 transition-colors cursor-pointer">
                  <div className="flex items-center gap-2">
                    <div className="size-8 rounded-full bg-cyan-500/20 flex items-center justify-center shrink-0">
                      <span className="text-[10px] font-semibold text-cyan-400">{user?.avatar || "AP"}</span>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-white truncate">{user?.name || "Arjun Patel"}</p>
                      <p className="text-[10px] text-zinc-400 truncate">{user?.role || "Admin"}</p>
                    </div>
                  </div>
                </div>
              </Link>

              <button
                onClick={handleLogout}
                className="w-full flex items-center gap-2 px-2.5 py-2 rounded-lg text-sm font-medium text-zinc-400 hover:text-white hover:bg-zinc-900 transition-colors"
              >
                <LogOut className="size-4 shrink-0" />
                <span className="text-xs">Sign Out</span>
              </button>
            </>
          ) : (
            <>
              <Link href="/profile" className="flex justify-center" title="Profile">
                <div className="size-8 rounded-full bg-cyan-500/20 flex items-center justify-center">
                  <span className="text-[10px] font-semibold text-cyan-400">{user?.avatar || "AP"}</span>
                </div>
              </Link>

              <button
                onClick={handleLogout}
                className="w-full flex justify-center p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-900 transition-colors"
                title="Logout"
              >
                <LogOut className="size-4" />
              </button>
            </>
          )}
        </div>
      </aside>
    </>
  )
}
