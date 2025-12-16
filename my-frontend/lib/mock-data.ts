export interface DetectionModel {
  id: string
  name: string
  type: "face" | "person" | "text" | "medical" | "license-plate"
  enabled: boolean
  accuracy: number
}

export interface ComplianceMode {
  id: string
  name: string
  description: string
  level: "strict" | "moderate" | "custom"
}

export interface Upload {
  id: string
  filename: string
  size: number
  uploadedAt: Date
  status: "processing" | "completed" | "failed"
  detections: number
  processingTime?: number
  thumbnail: string
  complianceMode: string
}

export interface DashboardMetrics {
  totalUploads: number
  totalDetections: number
  activeProcessing: number
  complianceScore: number
  avgProcessingTime: number
  detectionBreakdown: {
    faces: number
    people: number
    text: number
    medical: number
    licensePlates: number
  }
}

export interface User {
  id: string
  name: string
  email: string
  role: string
  avatar: string
  department: string
  joinedDate: Date
}

export interface ProcessingQueueItem {
  id: string
  filename: string
  size: number
  thumbnail: string
  progress: number
  status: "processing" | "completed"
  estimatedTime: string
  models: {
    name: string
    status: "completed" | "processing" | "pending"
    confidence: number
    detections: number
  }[]
}

export interface ReviewResult {
  id: string
  filename: string
  originalUrl: string
  processedUrl: string
  processingTime: number
  complianceMode: string
  detections: {
    id: string
    type: "face" | "person" | "text" | "medical" | "license-plate"
    confidence: number
    boundingBox: { x: number; y: number; width: number; height: number }
  }[]
}

export const detectionModels: DetectionModel[] = [
  { id: "face", name: "Face Detection", type: "face", enabled: true, accuracy: 98.5 },
  { id: "person", name: "Person Detection", type: "person", enabled: true, accuracy: 96.2 },
  { id: "text", name: "Text Recognition", type: "text", enabled: false, accuracy: 94.7 },
  { id: "medical", name: "Medical Info", type: "medical", enabled: false, accuracy: 97.1 },
  { id: "license", name: "License Plates", type: "license-plate", enabled: false, accuracy: 99.3 },
]

export const complianceModes: ComplianceMode[] = [
  {
    id: "strict",
    name: "Strict Mode",
    description: "Maximum privacy - Blur all detected sensitive content",
    level: "strict",
  },
  {
    id: "moderate",
    name: "Moderate Mode",
    description: "Balanced approach - Blur high-confidence detections only",
    level: "moderate",
  },
  {
    id: "custom",
    name: "Custom Mode",
    description: "Manual review - Review each detection before applying",
    level: "custom",
  },
]

export const mockUsers: User[] = [
  {
    id: "1",
    name: "Arjun Patel",
    email: "arjun.patel@privacyshield.com",
    role: "Admin",
    avatar: "AP",
    department: "Security Operations",
    joinedDate: new Date(2023, 5, 15),
  },
  {
    id: "2",
    name: "Priya Sharma",
    email: "priya.sharma@privacyshield.com",
    role: "Analyst",
    avatar: "PS",
    department: "Compliance",
    joinedDate: new Date(2023, 8, 10),
  },
  {
    id: "3",
    name: "Rahul Kumar",
    email: "rahul.kumar@privacyshield.com",
    role: "Developer",
    avatar: "RK",
    department: "Engineering",
    joinedDate: new Date(2024, 1, 5),
  },
]

export const recentUploads: Upload[] = [
  {
    id: "1",
    filename: "surveillance_cam_001.jpg",
    size: 2458000,
    uploadedAt: new Date(Date.now() - 1000 * 60 * 5),
    status: "completed",
    detections: 12,
    processingTime: 2.3,
    thumbnail: "/surveillance-camera-footage.jpg",
    complianceMode: "Strict Mode",
  },
  {
    id: "2",
    filename: "medical_record_scan.pdf",
    size: 1245000,
    uploadedAt: new Date(Date.now() - 1000 * 60 * 15),
    status: "completed",
    detections: 8,
    processingTime: 4.1,
    thumbnail: "/medical-document.png",
    complianceMode: "Strict Mode",
  },
  {
    id: "3",
    filename: "parking_lot_footage.mp4",
    size: 8900000,
    uploadedAt: new Date(Date.now() - 1000 * 60 * 3),
    status: "processing",
    detections: 0,
    thumbnail: "/busy-city-parking-lot.png",
    complianceMode: "Moderate Mode",
  },
  {
    id: "4",
    filename: "conference_photo.jpg",
    size: 3200000,
    uploadedAt: new Date(Date.now() - 1000 * 60 * 45),
    status: "completed",
    detections: 24,
    processingTime: 3.8,
    thumbnail: "/modern-conference-room.png",
    complianceMode: "Custom Mode",
  },
]

export const recentUploadsUpdated: Upload[] = [
  {
    id: "1",
    filename: "mumbai_surveillance_001.jpg",
    size: 2458000,
    uploadedAt: new Date(Date.now() - 1000 * 60 * 5),
    status: "completed",
    detections: 12,
    processingTime: 2.3,
    thumbnail: "/surveillance-camera-footage.jpg",
    complianceMode: "Strict Mode",
  },
  {
    id: "2",
    filename: "patient_record_sharma.pdf",
    size: 1245000,
    uploadedAt: new Date(Date.now() - 1000 * 60 * 15),
    status: "completed",
    detections: 8,
    processingTime: 4.1,
    thumbnail: "/medical-document.png",
    complianceMode: "Strict Mode",
  },
  {
    id: "3",
    filename: "delhi_parking_lot.mp4",
    size: 8900000,
    uploadedAt: new Date(Date.now() - 1000 * 60 * 3),
    status: "processing",
    detections: 0,
    thumbnail: "/busy-city-parking-lot.png",
    complianceMode: "Moderate Mode",
  },
  {
    id: "4",
    filename: "bangalore_conference_2024.jpg",
    size: 3200000,
    uploadedAt: new Date(Date.now() - 1000 * 60 * 45),
    status: "completed",
    detections: 24,
    processingTime: 3.8,
    thumbnail: "/modern-conference-room.png",
    complianceMode: "Custom Mode",
  },
]

export const dashboardMetrics: DashboardMetrics = {
  totalUploads: 1247,
  totalDetections: 18392,
  activeProcessing: 3,
  complianceScore: 98.7,
  avgProcessingTime: 3.2,
  detectionBreakdown: {
    faces: 8234,
    people: 5621,
    text: 2845,
    medical: 1203,
    licensePlates: 489,
  },
}

// Demo credentials
export const DEMO_CREDENTIALS = {
  email: "demo@privacyshield.com",
  password: "demo123",
  user: {
    id: "demo",
    name: "Arjun Patel",
    email: "demo@privacyshield.com",
    role: "Admin",
    avatar: "AP",
    department: "Security Operations",
    joinedDate: new Date(2023, 5, 15),
  },
}

// Processing queue data for real-time processing page
export const processingQueue: ProcessingQueueItem[] = [
  {
    id: "proc-1",
    filename: "mumbai_surveillance_001.jpg",
    size: 2458000,
    thumbnail: "/surveillance-camera-footage.jpg",
    progress: 75,
    status: "processing",
    estimatedTime: "30s",
    models: [
      { name: "Face Detection", status: "completed", confidence: 98, detections: 12 },
      { name: "Person Detection", status: "completed", confidence: 96, detections: 15 },
      { name: "Text Recognition", status: "processing", confidence: 94, detections: 5 },
      { name: "Medical Info", status: "pending", confidence: 0, detections: 1 },
      { name: "License Plate", status: "pending", confidence: 0, detections: 3 },
    ],
  },
  {
    id: "proc-2",
    filename: "patient_record_sharma.pdf",
    size: 1245000,
    thumbnail: "/medical-document.png",
    progress: 45,
    status: "processing",
    estimatedTime: "1m 15s",
    models: [
      { name: "Face Detection", status: "completed", confidence: 97, detections: 3 },
      { name: "Person Detection", status: "processing", confidence: 95, detections: 4 },
      { name: "Text Recognition", status: "pending", confidence: 0, detections: 2 },
      { name: "Medical Info", status: "pending", confidence: 0, detections: 4 },
      { name: "License Plate", status: "pending", confidence: 0, detections: 7 },
    ],
  },
  {
    id: "proc-3",
    filename: "delhi_parking_lot.mp4",
    size: 8900000,
    thumbnail: "/busy-city-parking-lot.png",
    progress: 100,
    status: "completed",
    estimatedTime: "0s",
    models: [
      { name: "Face Detection", status: "completed", confidence: 99, detections: 8 },
      { name: "Person Detection", status: "completed", confidence: 97, detections: 10 },
      { name: "Text Recognition", status: "completed", confidence: 95, detections: 12 },
      { name: "Medical Info", status: "completed", confidence: 98, detections: 0 },
      { name: "License Plate", status: "completed", confidence: 99, detections: 6 },
    ],
  },
]

// Review results data for result review page
export const reviewResults: ReviewResult[] = [
  {
    id: "review-1",
    filename: "mumbai_surveillance_001.jpg",
    originalUrl: "/surveillance-camera-footage.jpg",
    processedUrl: "/surveillance-camera-footage.jpg",
    processingTime: 2.3,
    complianceMode: "Strict Mode",
    detections: [
      {
        id: "det-1",
        type: "face",
        confidence: 98,
        boundingBox: { x: 15, y: 20, width: 12, height: 15 },
      },
      {
        id: "det-2",
        type: "face",
        confidence: 96,
        boundingBox: { x: 45, y: 25, width: 10, height: 13 },
      },
      {
        id: "det-3",
        type: "person",
        confidence: 95,
        boundingBox: { x: 60, y: 30, width: 20, height: 35 },
      },
      {
        id: "det-4",
        type: "license-plate",
        confidence: 99,
        boundingBox: { x: 25, y: 65, width: 8, height: 4 },
      },
      {
        id: "det-5",
        type: "text",
        confidence: 94,
        boundingBox: { x: 70, y: 10, width: 15, height: 5 },
      },
    ],
  },
]

export const mockUploads = recentUploadsUpdated
