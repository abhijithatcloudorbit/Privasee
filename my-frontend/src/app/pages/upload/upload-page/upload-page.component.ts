import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';

interface FileInfo {
  name: string;
  size: string;
  dimensions: string;
}

interface BatchStatus {
  totalFiles: number;
  processedFiles: number;
  isProcessing: boolean;
}

@Component({
  selector: 'app-upload-page',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './upload-page.component.html',
  styleUrls: ['./upload-page.component.scss']
})
export class UploadPageComponent implements OnInit {
  // State
  batchStatus: BatchStatus = {
    totalFiles: 2,
    processedFiles: 0,
    isProcessing: false
  };

  currentFile: FileInfo = {
    name: 'test.jpg',
    size: '2.4 MB',
    dimensions: '1920 × 1080'
  };

  previewImageUrl: string | null = null;
  progressPercentage: number = 0;
  progressStatus: string = 'Ready to start';
  
  private processingInterval: any;

  ngOnInit(): void {
    this.initializeUI();
  }

  ngOnDestroy(): void {
    this.clearProcessingInterval();
  }

  initializeUI(): void {
    this.updateBatchDisplay();
  }

  updateBatchDisplay(): void {
    if (this.batchStatus.isProcessing) {
      this.progressPercentage = Math.round((this.batchStatus.processedFiles / this.batchStatus.totalFiles) * 100);
      this.progressStatus = this.progressPercentage === 100 ? 'Completed' : 'Processing';
    } else {
      this.progressPercentage = 0;
      this.progressStatus = 'Ready to start';
    }
  }

  // Event Handlers
  handleUploadClick(): void {
    console.log('File browser would open here');
    // In a real app, trigger file input click
    // const fileInput = document.createElement('input');
    // fileInput.type = 'file';
    // fileInput.multiple = true;
    // fileInput.accept = 'image/*,video/*';
    // fileInput.click();
  }

  handleDragOver(event: DragEvent): void {
    event.preventDefault();
    const uploadArea = event.target as HTMLElement;
    uploadArea.classList.add('dragover');
  }

  handleDragLeave(): void {
    // Remove class from all elements with upload-area class
    const uploadAreas = document.querySelectorAll('.upload-area');
    uploadAreas.forEach(area => area.classList.remove('dragover'));
  }

  handleDrop(event: DragEvent): void {
    event.preventDefault();
    const uploadArea = event.target as HTMLElement;
    uploadArea.classList.remove('dragover');
    
    if (event.dataTransfer?.files) {
      const files = Array.from(event.dataTransfer.files);
      console.log('Files dropped:', files);
      
      // Update batch count
      this.batchStatus.totalFiles = files.length;
      this.updateBatchDisplay();
      
      // Update preview with first file
      if (files.length > 0) {
        const firstFile = files[0];
        this.currentFile = {
          name: firstFile.name,
          size: this.formatFileSize(firstFile.size),
          dimensions: 'Loading...'
        };
        
        // If it's an image, get dimensions and preview
        if (firstFile.type.startsWith('image/')) {
          this.getImagePreview(firstFile);
        } else {
          this.currentFile.dimensions = 'N/A';
          this.previewImageUrl = null;
        }
      }
    }
  }

  handleUploadButton(): void {
    console.log('Primary upload button clicked');
    // Implement actual upload logic here
  }

  handleSecondaryButton(): void {
    console.log('Secondary button clicked - btn-secondary works!');
    alert('Secondary button works!');
  }

  handleGhostButton(): void {
    console.log('Ghost button clicked - btn-ghost works!');
    alert('Ghost button works!');
  }

  // Start batch processing
  startProcessing(): void {
    if (this.batchStatus.isProcessing || this.batchStatus.totalFiles === 0) return;
    
    this.batchStatus.isProcessing = true;
    this.batchStatus.processedFiles = 0;
    this.updateBatchDisplay();
    
    this.clearProcessingInterval();
    
    this.processingInterval = setInterval(() => {
      this.batchStatus.processedFiles++;
      
      if (this.batchStatus.processedFiles > this.batchStatus.totalFiles) {
        this.batchStatus.processedFiles = this.batchStatus.totalFiles;
      }
      
      this.updateBatchDisplay();
      
      // When all files are processed
      if (this.batchStatus.processedFiles >= this.batchStatus.totalFiles) {
        this.clearProcessingInterval();
        
        // Reset after a delay
        setTimeout(() => {
          this.batchStatus.processedFiles = 0;
          this.batchStatus.isProcessing = false;
          this.updateBatchDisplay();
        }, 2000);
      }
    }, 500);
  }

  // Helper functions
  private formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  private getImagePreview(file: File): void {
    const reader = new FileReader();
    
    reader.onload = (e: ProgressEvent<FileReader>) => {
      const img = new Image();
      
      img.onload = () => {
        this.currentFile.dimensions = `${img.width} × ${img.height}`;
        this.previewImageUrl = e.target?.result as string;
      };
      
      if (e.target?.result) {
        img.src = e.target.result as string;
      }
    };
    
    reader.readAsDataURL(file);
  }

  private clearProcessingInterval(): void {
    if (this.processingInterval) {
      clearInterval(this.processingInterval);
      this.processingInterval = null;
    }
  }
}