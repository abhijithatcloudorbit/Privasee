export function getResultUrl(jobId: string) {
  return `${process.env.NEXT_PUBLIC_PROCESSING_API}/result/${jobId}`
}
