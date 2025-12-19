export async function startProcessing(
  fileId: string,
  rawFileKey: string
) {
  const params = new URLSearchParams({
    file_id: fileId,
    raw_file_key: rawFileKey,
  })

  const res = await fetch(
    `${process.env.NEXT_PUBLIC_PROCESSING_API}/process?${params.toString()}`,
    { method: "POST" }
  )

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Process failed: ${res.status} ${text}`)
  }

  return res.json() as Promise<{
    job_id: string
    status: string
  }>
}
