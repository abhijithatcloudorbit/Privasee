import { supabase } from "@/lib/supabase"

export async function getResultBlobUrl(jobId: string): Promise<string> {
  const {
    data: { session },
  } = await supabase.auth.getSession()

  if (!session) {
    throw new Error("User not authenticated")
  }

  const res = await fetch(
    `${process.env.NEXT_PUBLIC_PROCESSING_API}/result/${jobId}`,
    {
      headers: {
        Authorization: `Bearer ${session.access_token}`,
      },
      cache: "no-store",
    }
  )

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Result fetch failed: ${text}`)
  }

  const blob = await res.blob()
  return URL.createObjectURL(blob)
}
