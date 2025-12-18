"use client"

import React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Code, Copy, Eye, EyeOff, Key, RefreshCw, CheckCircle } from "lucide-react"

export default function APIPage() {
  const [apiKey, setApiKey] = React.useState("")
  const [showKey, setShowKey] = React.useState(false)
  const [copied, setCopied] = React.useState(false)

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleRegenerateKey = () => {
    const newKey = `sk_live_${Math.random().toString(36).substring(2, 15)}${Math.random().toString(36).substring(2, 15)}`
    setApiKey(newKey)
  }

  const codeExamples = {
    curl: `curl -X POST https://api.privacyshield.com/v1/anonymize \\
  -H "Authorization: Bearer ${apiKey}" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_url": "https://example.com/image.jpg",
    "models": ["face", "text"],
    "compliance_mode": "strict"
  }'`,
    javascript: `const response = await fetch('https://api.privacyshield.com/v1/anonymize', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer ${apiKey}',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    image_url: 'https://example.com/image.jpg',
    models: ['face', 'text'],
    compliance_mode: 'strict'
  })
});

const data = await response.json();`,
    python: `import requests

response = requests.post(
    'https://api.privacyshield.com/v1/anonymize',
    headers={
        'Authorization': f'Bearer ${apiKey}',
        'Content-Type': 'application/json'
    },
    json={
        'image_url': 'https://example.com/image.jpg',
        'models': ['face', 'text'],
        'compliance_mode': 'strict'
    }
)

data = response.json()`,
  }

  const [selectedLanguage, setSelectedLanguage] = React.useState<keyof typeof codeExamples>("curl")

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">API Configuration</h1>
        <p className="text-muted-foreground mt-2">Integrate Privacy Shield into your applications</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Key className="size-5" />
            API Key Management
          </CardTitle>
          <CardDescription>Use this key to authenticate API requests</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-2">
            <div className="flex-1">
              <Input type={showKey ? "text" : "password"} value={apiKey} readOnly className="font-mono text-sm" />
            </div>
            <Button variant="outline" size="icon" onClick={() => setShowKey(!showKey)}>
              {showKey ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
            </Button>
            <Button variant="outline" size="icon" onClick={() => handleCopy(apiKey)}>
              {copied ? <CheckCircle className="size-4 text-green-500" /> : <Copy className="size-4" />}
            </Button>
            <Button variant="outline" onClick={handleRegenerateKey}>
              <RefreshCw className="size-4 mr-2" />
              Regenerate
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Keep your API key secure and never share it publicly. Regenerating will invalidate the previous key.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>API Usage Statistics</CardTitle>
          <CardDescription>Monitor your API consumption</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="border border-border rounded-lg p-4">
              <p className="text-sm text-muted-foreground">Requests Today</p>
              <p className="text-2xl font-bold mt-1">1,247</p>
              <Badge className="mt-2 bg-primary/20 text-primary">Active</Badge>
            </div>
            <div className="border border-border rounded-lg p-4">
              <p className="text-sm text-muted-foreground">Rate Limit</p>
              <p className="text-2xl font-bold mt-1">10,000/day</p>
              <Badge className="mt-2 bg-green-500/20 text-green-500">87% Available</Badge>
            </div>
            <div className="border border-border rounded-lg p-4">
              <p className="text-sm text-muted-foreground">Success Rate</p>
              <p className="text-2xl font-bold mt-1">99.8%</p>
              <Badge className="mt-2 bg-cyan-500/20 text-cyan-500">Excellent</Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Code className="size-5" />
                Code Examples
              </CardTitle>
              <CardDescription>Quick start integration examples</CardDescription>
            </div>
            <div className="flex gap-2">
              <Button
                variant={selectedLanguage === "curl" ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedLanguage("curl")}
              >
                cURL
              </Button>
              <Button
                variant={selectedLanguage === "javascript" ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedLanguage("javascript")}
              >
                JavaScript
              </Button>
              <Button
                variant={selectedLanguage === "python" ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedLanguage("python")}
              >
                Python
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="relative">
            <Button
              variant="outline"
              size="sm"
              className="absolute right-2 top-2 z-10 bg-transparent"
              onClick={() => handleCopy(codeExamples[selectedLanguage])}
            >
              {copied ? <CheckCircle className="size-4 mr-2 text-green-500" /> : <Copy className="size-4 mr-2" />}
              {copied ? "Copied!" : "Copy"}
            </Button>
            <pre className="bg-secondary/50 border border-border rounded-lg p-4 overflow-x-auto">
              <code className="text-xs font-mono">{codeExamples[selectedLanguage]}</code>
            </pre>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>API Endpoints</CardTitle>
          <CardDescription>Available API endpoints and methods</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[
              { method: "POST", endpoint: "/v1/anonymize", description: "Upload and anonymize image" },
              { method: "GET", endpoint: "/v1/jobs/:id", description: "Get job status and results" },
              { method: "GET", endpoint: "/v1/jobs", description: "List all anonymization jobs" },
              { method: "DELETE", endpoint: "/v1/jobs/:id", description: "Delete a job and its data" },
              { method: "GET", endpoint: "/v1/usage", description: "Get API usage statistics" },
            ].map((endpoint, index) => (
              <div
                key={index}
                className="flex items-center justify-between border border-border rounded-lg p-3 hover:bg-secondary/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <Badge
                    variant={
                      endpoint.method === "POST" ? "default" : endpoint.method === "GET" ? "secondary" : "destructive"
                    }
                  >
                    {endpoint.method}
                  </Badge>
                  <code className="text-sm font-mono">{endpoint.endpoint}</code>
                </div>
                <span className="text-sm text-muted-foreground">{endpoint.description}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
