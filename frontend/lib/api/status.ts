import { supabase } from "@/lib/supabase"

export async function getJobStatus(jobId: string) {
  const {
    data: { session },
  } = await supabase.auth.getSession()

  if (!session) {
    throw new Error("User not authenticated")
  }

  const res = await fetch(
    `${process.env.NEXT_PUBLIC_PROCESSING_API}/status/${jobId}`,
    {
      headers: {
        Authorization: `Bearer ${session.access_token}`,
      },
    }
  )

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Status fetch failed: ${text}`)
  }

  return res.json() as Promise<{
    status: "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED"
    progress: number
  }>
}
