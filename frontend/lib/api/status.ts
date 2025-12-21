import { supabase } from "@/lib/supabase"

export async function getJobStatus(jobId: string): Promise<{
  status: "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED"
  progress: number
}> {
  const {
    data: { session },
    error: sessionError,
  } = await supabase.auth.getSession()

  if (sessionError || !session) {
    throw new Error("User not authenticated")
  }

  const apiBase = process.env.NEXT_PUBLIC_PROCESSING_API
  if (!apiBase) {
    throw new Error("NEXT_PUBLIC_PROCESSING_API is not defined")
  }

  const res = await fetch(`${apiBase}/status/${jobId}`, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${session.access_token}`,
    },
    cache: "no-store",
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Status fetch failed (${res.status}): ${text}`)
  }

  return res.json()
}
