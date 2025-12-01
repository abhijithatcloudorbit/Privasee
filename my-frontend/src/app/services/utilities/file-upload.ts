import { Injectable, signal } from '@angular/core';

export interface UploadProgress {
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  error?: string;
}

@Injectable({
  providedIn: 'root'
})
export class FileUploadService {
  private uploadQueue = signal<UploadProgress[]>([]);
  private isUploading = signal(false);
  
  // Get signals for reactivity
  readonly queue = this.uploadQueue.asReadonly();
  readonly uploading = this.isUploading.asReadonly();
  
  constructor() {}
  
  // Add files to queue
  addFiles(files: File[]) {
    const newItems: UploadProgress[] = files.map(file => ({
      file,
      progress: 0,
      status: 'pending'
    }));
    
    this.uploadQueue.update(queue => [...queue, ...newItems]);
  }
  
  // Upload files with simulated progress
  async uploadFiles(options: any = {}): Promise<void> {
    this.isUploading.set(true);
    const queue = this.uploadQueue();
    
    // Simulate upload for each file
    for (let i = 0; i < queue.length; i++) {
      if (queue[i].status === 'pending') {
        this.updateFileStatus(i, 'uploading');
        
        // Simulate upload progress
        for (let progress = 0; progress <= 100; progress += 10) {
          await this.delay(100); // Simulate network delay
          this.updateFileProgress(i, progress);
        }
        
        this.updateFileStatus(i, 'processing');
        await this.delay(500); // Simulate processing
        this.updateFileStatus(i, 'completed');
      }
    }
    
    this.isUploading.set(false);
  }
  
  // Clear queue
  clearQueue() {
    this.uploadQueue.set([]);
  }
  
  // Remove single file
  removeFile(index: number) {
    this.uploadQueue.update(queue => queue.filter((_, i) => i !== index));
  }
  
  private updateFileProgress(index: number, progress: number) {
    this.uploadQueue.update(queue => {
      const updated = [...queue];
      updated[index] = { ...updated[index], progress };
      return updated;
    });
  }
  
  private updateFileStatus(index: number, status: UploadProgress['status']) {
    this.uploadQueue.update(queue => {
      const updated = [...queue];
      updated[index] = { ...updated[index], status };
      return updated;
    });
  }
  
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}