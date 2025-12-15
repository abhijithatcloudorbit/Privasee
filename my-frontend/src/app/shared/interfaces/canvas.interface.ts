export interface Detection {
  id: string;
  type: 'face' | 'text' | 'license_plate' | 'person' | 'vehicle' | 'signature';
  confidence: number;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  metadata?: {
    text?: string;
    [key: string]: any;
  };
  appliedFilter?: 'blur' | 'pixelate' | 'redact' | 'none';
}

export type ToolType = 'select' | 'brush' | 'eraser' | 'pan' | 'zoom';
export type PrivacyFilter = 'blur' | 'pixelate' | 'redact' | 'none';

export interface CanvasTransform {
  x: number;
  y: number;
  scale: number;
}