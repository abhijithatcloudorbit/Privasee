import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms'; // For ngModel
import { BtnPrimaryComponent } from '../../../atoms/buttons/btn-primary.component';
import { BtnSecondaryComponent } from '../../../atoms/buttons/btn-secondary/btn-secondary';
import { BtnGhostComponent } from '../../../atoms/buttons/btn-ghost/btn-ghost';

@Component({
  selector: 'app-file-upload-form',
  standalone: true,
  imports: [
    CommonModule,  // ← ADD THIS for *ngIf, *ngFor
    FormsModule,   // ← ADD THIS for [(ngModel)]
    BtnPrimaryComponent,
    BtnSecondaryComponent,
    BtnGhostComponent
  ],
  templateUrl: './file-upload-form.component.html',
  styleUrls: ['./file-upload-form.component.scss']
})
export class FileUploadFormComponent {
  // Add all missing properties
  selectedFiles: File[] = [];
  isDragging = false;
  imageDimensions: any[] = [];
  options = {
    blurFaces: true,
    blurLicensePlates: true,
    redactText: true,
    maintainResolution: true
  };
  errorMessage = '';
  isUploading = false;

  // Drag & Drop handlers
  onDragOver(event: DragEvent) {
    event.preventDefault();
    this.isDragging = true;
  }

  onDragLeave(event: DragEvent) {
    event.preventDefault();
    this.isDragging = false;
  }

  onFileDrop(event: DragEvent) {
    event.preventDefault();
    this.isDragging = false;
    if (event.dataTransfer?.files) {
      this.handleFiles(event.dataTransfer.files);
    }
  }

  onFileSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files) {
      this.handleFiles(input.files);
    }
  }

  openCamera() {
    // Placeholder
    console.log('Open camera');
  }

  // Helper methods
  isImage(file: File): boolean {
    return file.type.startsWith('image/');
  }

  getPreviewUrl(file: File): string {
    return URL.createObjectURL(file);
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  removeFile(index: number) {
    this.selectedFiles.splice(index, 1);
    this.imageDimensions.splice(index, 1);
  }

  clearAll() {
    this.selectedFiles = [];
    this.imageDimensions = [];
    this.errorMessage = '';
  }

  uploadFiles() {
    this.isUploading = true;
    // Simulate upload
    setTimeout(() => {
      this.isUploading = false;
      this.clearAll();
    }, 2000);
  }

  private handleFiles(files: FileList) {
    const fileArray = Array.from(files);
    this.selectedFiles = [...this.selectedFiles, ...fileArray];
    
    // Get image dimensions (simulated)
    fileArray.forEach(() => {
      this.imageDimensions.push({
        width: Math.floor(Math.random() * 1000) + 800,
        height: Math.floor(Math.random() * 800) + 600
      });
    });
  }
}