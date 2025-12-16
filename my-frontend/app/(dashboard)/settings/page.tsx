"use client"

import React from "react"
import { SidebarNav } from "@/components/sidebar-nav"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Bell, Shield, Palette } from "lucide-react"

export default function SettingsPage() {
  const [settings, setSettings] = React.useState({
    emailNotifications: true,
    processingAlerts: true,
    weeklyReports: false,
    twoFactorAuth: false,
    apiAccess: true,
    darkMode: true,
  })

  const toggleSetting = (key: keyof typeof settings) => {
    setSettings((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  return (
    <div className="flex min-h-screen">
      <SidebarNav />

      <main className="flex-1 ml-64 p-8">
        <div className="max-w-4xl mx-auto space-y-8">
          <div>
            <h1 className="text-4xl font-semibold tracking-tight text-balance">Settings</h1>
            <p className="text-muted-foreground mt-2">Manage your application preferences and configurations</p>
          </div>

          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Bell className="size-5 text-primary" />
                <CardTitle>Notifications</CardTitle>
              </div>
              <CardDescription>Configure how you receive updates and alerts</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="email-notifications">Email Notifications</Label>
                  <p className="text-sm text-muted-foreground">Receive email updates about your uploads</p>
                </div>
                <Switch
                  id="email-notifications"
                  checked={settings.emailNotifications}
                  onCheckedChange={() => toggleSetting("emailNotifications")}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="processing-alerts">Processing Alerts</Label>
                  <p className="text-sm text-muted-foreground">Get notified when processing completes</p>
                </div>
                <Switch
                  id="processing-alerts"
                  checked={settings.processingAlerts}
                  onCheckedChange={() => toggleSetting("processingAlerts")}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="weekly-reports">Weekly Reports</Label>
                  <p className="text-sm text-muted-foreground">Receive weekly analytics summaries</p>
                </div>
                <Switch
                  id="weekly-reports"
                  checked={settings.weeklyReports}
                  onCheckedChange={() => toggleSetting("weeklyReports")}
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Shield className="size-5 text-primary" />
                <CardTitle>Security</CardTitle>
              </div>
              <CardDescription>Manage your account security settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="two-factor">Two-Factor Authentication</Label>
                  <p className="text-sm text-muted-foreground">Add an extra layer of security</p>
                </div>
                <Switch
                  id="two-factor"
                  checked={settings.twoFactorAuth}
                  onCheckedChange={() => toggleSetting("twoFactorAuth")}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="api-access">API Access</Label>
                  <p className="text-sm text-muted-foreground">Allow third-party API integrations</p>
                </div>
                <Switch
                  id="api-access"
                  checked={settings.apiAccess}
                  onCheckedChange={() => toggleSetting("apiAccess")}
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Palette className="size-5 text-primary" />
                <CardTitle>Appearance</CardTitle>
              </div>
              <CardDescription>Customize your application interface</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="dark-mode">Dark Mode</Label>
                  <p className="text-sm text-muted-foreground">Use dark theme throughout the app</p>
                </div>
                <Switch id="dark-mode" checked={settings.darkMode} onCheckedChange={() => toggleSetting("darkMode")} />
              </div>
            </CardContent>
          </Card>

          <div className="flex items-center gap-3">
            <Button>Save Changes</Button>
            <Button variant="outline">Reset to Defaults</Button>
          </div>
        </div>
      </main>
    </div>
  )
}
