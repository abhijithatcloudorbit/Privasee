import { supabase } from "@/lib/supabase"

export async function startProcessing(
  fileId: string,
  mode: "face" | "license_plate"
): Promise<{ job_id: string }> {
  const params = new URLSearchParams({
    file_id: fileId,
    mode,
  })

  const {
    data: { session },
    error: sessionError,
  } = await supabase.auth.getSession()

  if (sessionError || !session) {
    throw new Error("User not authenticated")
  }

  const res = await fetch(
    `${process.env.NEXT_PUBLIC_PROCESSING_API}/process?${params.toString()}`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${session.access_token}`,
      },
    }
  )

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Process failed: ${res.status} ${text}`)
  }

  return res.json()
}
