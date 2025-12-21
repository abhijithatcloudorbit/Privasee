import { supabase } from "@/lib/supabase"

export async function getResultBlobUrl(jobId: string): Promise<string> {
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

  const res = await fetch(`${apiBase}/result/${jobId}`, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${session.access_token}`,
    },
    cache: "no-store",
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Result fetch failed (${res.status}): ${text}`)
  }

  const blob = await res.blob()
  return URL.createObjectURL(blob)
}
