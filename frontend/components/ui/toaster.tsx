"use client"

import { useToast } from "./use-toast"
import { X } from "lucide-react"

export function Toaster() {
  const { toasts, dismiss } = useToast()

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-3">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`w-80 rounded-lg border p-4 shadow-lg transition-all ${
            toast.variant === "destructive"
              ? "border-red-500 bg-red-50 text-red-900"
              : "border-border bg-background text-foreground"
          }`}
        >
          <div className="flex items-start justify-between gap-2">
            <div>
              {toast.title && (
                <p className="font-semibold leading-none">
                  {toast.title}
                </p>
              )}
              {toast.description && (
                <p className="mt-1 text-sm text-muted-foreground">
                  {toast.description}
                </p>
              )}
            </div>
            <button
              onClick={() => dismiss(toast.id)}
              className="text-muted-foreground hover:text-foreground"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}
