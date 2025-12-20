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
    const text = await res.text()
    throw new Error(text)
  }

  return res.json()
}
