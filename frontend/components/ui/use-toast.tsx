"use client"

import * as React from "react"

export type ToastVariant = "default" | "destructive"

export type Toast = {
  id: string
  title?: string
  description?: string
  variant?: ToastVariant
}

type ToastContextValue = {
  toasts: Toast[]
  toast: (toast: Omit<Toast, "id">) => void
  dismiss: (id: string) => void
}

const ToastContext = React.createContext<ToastContextValue | null>(null)

function generateId() {
  return crypto.randomUUID()
}

export function ToastProvider({
  children,
}: {
  children: React.ReactNode
}) {
  const [toasts, setToasts] = React.useState<Toast[]>([])

  const toast = React.useCallback(
    ({ title, description, variant = "default" }: Omit<Toast, "id">) => {
      const id = generateId()

      setToasts((prev) => [
        ...prev,
        { id, title, description, variant },
      ])

      // Auto-dismiss after 4 seconds
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id))
      }, 4000)
    },
    []
  )

  const dismiss = React.useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const value = React.useMemo(
    () => ({ toasts, toast, dismiss }),
    [toasts, toast, dismiss]
  )

  return (
    <ToastContext.Provider value={value}>
      {children}
    </ToastContext.Provider>
  )
}

export function useToast() {
  const context = React.useContext(ToastContext)

  if (!context) {
    throw new Error(
      "useToast must be used inside <ToastProvider>"
    )
  }

  return context
}
