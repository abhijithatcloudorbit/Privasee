export async function getJobStatus(jobId: string) {
  const res = await fetch(
    `${process.env.NEXT_PUBLIC_PROCESSING_API}/status/${jobId}`
  )

  if (!res.ok) {
    throw new Error("Status fetch failed")
  }

  return res.json() as Promise<{
    status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED"
    progress: number
  }>
}
