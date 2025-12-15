import { Injectable, signal } from '@angular/core';
import { BatchProcessingItem, BatchFile } from '../../shared/interfaces/batch.interface';

export interface UploadProgress {
  id: string;
  name: string;
  size: number;
  type: string;
  status: 'pending' | 'uploading' | 'completed' | 'failed';
  progress: number;
  url?: string;
  imageData?: string;
  detections?: any[];
  uploadedAt?: Date;
  width?: number;
  height?: number;
  file?: File;
}

@Injectable({
  providedIn: 'root'
})
export class FileUploadService {
  // Signal-based state
  batches = signal<BatchProcessingItem[]>([]);
  processingBatches = signal<BatchProcessingItem[]>([]);
  availableBatches = signal<BatchProcessingItem[]>([]);
  completedBatches = signal<BatchProcessingItem[]>([]);
  overallProgress = signal<number>(0);
  processingBatchIds = signal<string[]>([]);
  completedBatchIds = signal<string[]>([]);

  constructor() {
    // Initialize with sample data for testing
    this.initializeMockData();
  }

  private initializeMockData(): void {
    const mockBatch: BatchProcessingItem = {
      id: 'batch-1',
      name: 'Sample Batch',
      createdAt: new Date(),
      updatedAt: new Date(),
      files: [
        {
          id: 'file-1',
          name: 'sample-image.jpg',
          size: 2048576,
          type: 'image/jpeg',
          status: 'completed',
          progress: 100,
          url: 'https://via.placeholder.com/800x600',
          metadata: {
            width: 800,
            height: 600,
            uploadedAt: new Date()
          }
        }
      ],
      status: 'completed',
      progress: 100,
      selectedFileIndex: 0,
      totalFiles: 1,
      processedFiles: 1,
      failedFiles: 0,
      complianceChecked: true,
      complianceStatus: 'pass',
      processingTime: 2500,
      accuracyScore: 0.92
    };

    this.batches.set([mockBatch]);
    this.completedBatches.set([mockBatch]);
    this.completedBatchIds.set(['batch-1']);
  }

  // Batch operations
  startBatchProcessing(batchId: string): void {
    console.log(`Starting batch processing for: ${batchId}`);
    this.updateBatchStatus(batchId, 'processing');
  }

  pauseBatchProcessing(batchId: string): void {
    console.log(`Pausing batch processing for: ${batchId}`);
    this.updateBatchStatus(batchId, 'paused');
  }

  resumeBatchProcessing(batchId: string): void {
    console.log(`Resuming batch processing for: ${batchId}`);
    this.updateBatchStatus(batchId, 'processing');
  }

  removeBatch(batchId: string): void {
    this.batches.update(batches => batches.filter(b => b.id !== batchId));
  }

  removeFile(batchId: string, fileId: string): void {
    this.batches.update(batches =>
      batches.map(batch =>
        batch.id === batchId
          ? {
              ...batch,
              files: batch.files.filter(f => f.id !== fileId),
              totalFiles: batch.totalFiles - 1
            }
          : batch
      )
    );
  }

  clearCompletedBatches(): void {
    this.batches.update(batches => 
      batches.filter(b => b.status !== 'completed')
    );
    this.completedBatches.set([]);
    this.completedBatchIds.set([]);
  }

  // Helper methods
  private updateBatchStatus(batchId: string, status: BatchProcessingItem['status']): void {
    this.batches.update(batches =>
      batches.map(batch =>
        batch.id === batchId
          ? { ...batch, status, updatedAt: new Date() }
          : batch
      )
    );
  }

  toBatchFile(upload: UploadProgress): BatchFile {
    return {
      id: upload.id,
      name: upload.name,
      size: upload.size,
      type: upload.type,
      status: this.mapStatus(upload.status),
      progress: upload.progress,
      url: upload.url,
      detections: upload.detections,
      metadata: {
        width: upload.width,
        height: upload.height,
        uploadedAt: upload.uploadedAt
      }
    };
  }

  private mapStatus(status: UploadProgress['status']): BatchFile['status'] {
    const map: Record<UploadProgress['status'], BatchFile['status']> = {
      'pending': 'pending',
      'uploading': 'uploading',
      'completed': 'uploaded',
      'failed': 'failed'
    };
    return map[status];
  }
}