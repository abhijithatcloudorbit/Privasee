export async function uploadFile(file: File) {
  const formData = new FormData()
  formData.append("file", file)

  const res = await fetch(
    `${process.env.NEXT_PUBLIC_PROCESSING_API}/upload`,
    {
      method: "POST",
      body: formData,
    }
  )

  if (!res.ok) {
    throw new Error("Upload failed")
  }

  // IMPORTANT:
  // We extend the response shape but do not remove existing fields
  return res.json() as Promise<{
    file_id: string
    filename: string
    raw_file_key: string   // ✅ NEW (non-breaking addition)
  }>
}
