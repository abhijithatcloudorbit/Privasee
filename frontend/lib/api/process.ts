export async function startProcessing(
  fileId: string,
  mode: "face" | "license_plate"
) {
  const params = new URLSearchParams({
    file_id: fileId,
    mode: mode,
  })

  const res = await fetch(
    `${process.env.NEXT_PUBLIC_PROCESSING_API}/process?${params.toString()}`,
    { method: "POST" }
  )

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Process failed: ${res.status} ${text}`)
  }

  return res.json()
}
