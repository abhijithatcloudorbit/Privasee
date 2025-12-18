"use client"

import React from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { User, Mail, Briefcase, MapPin, Calendar, Shield } from "lucide-react"
import Link from "next/link"

export default function ProfilePage() {
  const [user, setUser] = React.useState({
    name: "Arjun Patel",
    email: "arjun.patel@privacyshield.com",
    role: "Admin",
    department: "Security Operations",
    location: "Mumbai, India",
    joinedDate: "January 2024",
    avatar: "AP",
  })

  React.useEffect(() => {
    const storedUser = localStorage.getItem("user")
    console.log("[v0] Profile - loaded user:", storedUser)
    if (storedUser) {
      const userData = JSON.parse(storedUser)
      setUser((prev) => ({ ...prev, ...userData }))
    }
  }, [])

  return (
    <div className="p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <div>
          <h1 className="text-4xl font-semibold tracking-tight text-balance">Profile</h1>
          <p className="text-muted-foreground mt-2">Manage your account information and preferences</p>
        </div>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-4">
              <div className="size-20 rounded-full bg-primary/20 flex items-center justify-center">
                <span className="text-2xl font-semibold text-primary">{user.avatar}</span>
              </div>
              <div>
                <CardTitle>{user.name}</CardTitle>
                <CardDescription>{user.email}</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <Label htmlFor="name">Full Name</Label>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-secondary/30">
                  <User className="size-4 text-muted-foreground" />
                  <span className="text-sm">{user.name}</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-secondary/30">
                  <Mail className="size-4 text-muted-foreground" />
                  <span className="text-sm">{user.email}</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="role">Role</Label>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-secondary/30">
                  <Shield className="size-4 text-muted-foreground" />
                  <span className="text-sm">{user.role}</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="department">Department</Label>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-secondary/30">
                  <Briefcase className="size-4 text-muted-foreground" />
                  <span className="text-sm">{user.department}</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="location">Location</Label>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-secondary/30">
                  <MapPin className="size-4 text-muted-foreground" />
                  <span className="text-sm">{user.location}</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="joined">Joined Date</Label>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-secondary/30">
                  <Calendar className="size-4 text-muted-foreground" />
                  <span className="text-sm">{user.joinedDate}</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3 pt-4">
              <Link href="/change-password">
                <Button variant="outline">Change Password</Button>
              </Link>
              <Link href="/settings">
                <Button>Edit Profile</Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
