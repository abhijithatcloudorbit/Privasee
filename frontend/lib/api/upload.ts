export async function uploadFile(file: File): Promise<{ file_id: string }> {
  const apiBase = process.env.NEXT_PUBLIC_PROCESSING_API
  if (!apiBase) {
    throw new Error("NEXT_PUBLIC_PROCESSING_API is not defined")
  }

  const formData = new FormData()
  formData.append("file", file)

  const res = await fetch(`${apiBase}/upload`, {
    method: "POST",
    body: formData,
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Upload failed (${res.status}): ${text}`)
  }

  return res.json()
}
