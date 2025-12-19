export function getResultUrl(jobId: string): string {
  if (!process.env.NEXT_PUBLIC_PROCESSING_API) {
    throw new Error("NEXT_PUBLIC_PROCESSING_API is not defined")
  }

  return `${process.env.NEXT_PUBLIC_PROCESSING_API}/result/${jobId}?v=${Date.now()}`
}
