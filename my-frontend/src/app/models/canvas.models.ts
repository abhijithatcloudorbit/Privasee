// src/app/models/canvas.models.ts
export interface DetectionBox {
  id: string;
  type: 'face' | 'license_plate' | 'text';
  x: number; // percentage of width
  y: number; // percentage of height
  width: number; // percentage
  height: number; // percentage
  confidence: number; // 0-1
  privacyFilter?: 'blur' | 'pixelate' | 'redact' | 'remove';
  isManualOverride?: boolean;
}

export interface CanvasTool {
  id: 'brush' | 'blur' | 'pixelate' | 'redact' | 'eraser' | 'zoom' | 'pan';
  name: string;
  icon: string;
  size: number; // brush size or effect intensity
  color?: string;
}

export interface CanvasState {
  originalImage: string | null;
  processedImage: string | null;
  detections: DetectionBox[];
  activeTool: CanvasTool;
  zoomLevel: number;
  panOffset: { x: number; y: number };
  showComparison: boolean; // side-by-side or overlay
  showGrid: boolean;
  showDetections: boolean;
}