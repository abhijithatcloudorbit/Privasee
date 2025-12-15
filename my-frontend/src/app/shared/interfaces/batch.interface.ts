import { Detection } from './canvas.interface';

export interface BatchFile {
  id: string;
  name: string;
  size: number;
  type: string;
  status: 'pending' | 'uploading' | 'uploaded' | 'processing' | 'completed' | 'failed';
  progress: number;
  url?: string;
  detections?: Detection[];
  metadata?: {
    width?: number;
    height?: number;
    uploadedAt?: Date;
    processedAt?: Date;
  };
  // Add these to fix template errors:
  previewUrl?: string; // Optional for backward compatibility
  file?: File; // Optional original file
}

export interface BatchProcessingItem {
  id: string;
  name: string;
  createdAt: Date;
  updatedAt: Date;
  files: BatchFile[];
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'paused';
  progress: number;
  selectedFileIndex: number;
  selectedFile?: BatchFile;
  totalFiles: number;
  processedFiles: number;
  failedFiles: number;
  complianceChecked: boolean;
  complianceStatus: 'pending' | 'pass' | 'fail' | 'review';
  processingTime?: number;
  accuracyScore?: number;
}