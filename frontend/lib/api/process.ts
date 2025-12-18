export async function startProcessing(fileId: string) {
  const res = await fetch(
    `${process.env.NEXT_PUBLIC_PROCESSING_API}/process?file_id=${fileId}`,
    {
      method: "POST",
    }
  )

  if (!res.ok) {
    throw new Error("Processing failed")
  }

  return res.json() as Promise<{
    job_id: string
    status: string
  }>
}
