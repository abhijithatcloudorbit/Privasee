import { Component, EventEmitter, Input, Output, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BtnPrimaryComponent } from '../../atoms/buttons/btn-primary.component';
import { BtnSecondaryComponent } from '../../atoms/buttons/btn-secondary/btn-secondary';
import { BadgeComponent } from '../../atoms/misc/badge/badge.component';
import { ProgressBarComponent } from '../../atoms/feedback/progress-bar/progress-bar.component';
import { InputFileComponent } from '../../atoms/inputs/input-file/input-file.component';
import { FileUploadCardComponent } from '../../molecules/file-upload-card/file-upload-card.component';

export interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  progress: number;
  status: 'pending' | 'uploading' | 'completed' | 'error';
  errorMessage?: string;
}

@Component({
  selector: 'app-upload-zone-organism',
  standalone: true,
  imports: [
    CommonModule, 
    InputFileComponent, 
    FileUploadCardComponent,
    BtnPrimaryComponent, 
    BtnSecondaryComponent, 
    BadgeComponent, 
    ProgressBarComponent
  ],
  templateUrl: './upload-zone-organism.component.html',
  styleUrls: ['./upload-zone-organism.component.scss']
})
export class UploadZoneOrganismComponent {
  @Input() acceptTypes: string = 'image/*,.jpg,.jpeg,.png,.gif,.webp';
  @Input() maxFileSize: number = 10 * 1024 * 1024; // 10MB default
  @Input() maxFileCount: number = 20;
  
  @Output() filesSelected = new EventEmitter<File[]>();

  // Signal-based state
  files = signal<UploadedFile[]>([]);
  isUploading = signal<boolean>(false);
  isDragging = signal<boolean>(false);
  totalUploadProgress = signal<number>(0);

  // Computed signals (moved complex logic from template)
  uploadingFilesCount = computed(() => {
    return this.files().filter(f => f.status === 'uploading').length;
  });

  uploadedFilesCount = computed(() => {
    return this.files().length;
  });

  totalFileSize = computed(() => {
    return this.files().reduce((total, file) => total + file.size, 0);
  });

  // Helper methods
  getFileTypeDisplay(type: string): string {
    if (type.includes('/')) {
      return type.split('/')[1].toUpperCase();
    }
    return type.toUpperCase();
  }

  getStatusDisplay(status: UploadedFile['status']): string {
    switch (status) {
      case 'pending': return 'Pending';
      case 'uploading': return 'Uploading';
      case 'completed': return 'Completed';
      case 'error': return 'Error';
      default: return 'Unknown';
    }
  }

  // Drag and Drop Handlers
  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging.set(true);
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging.set(false);
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging.set(false);
    
    if (event.dataTransfer?.files) {
      const files = Array.from(event.dataTransfer.files);
      this.handleFiles(files);
    }
  }

  // Called by <app-input-file>
  onInputFilesSelected(fileList: FileList): void {
    const files = Array.from(fileList);
    this.handleFiles(files);
  }

  onUpload(): void {
    const currentFiles = this.files();
    if (currentFiles.length === 0 || this.isUploading()) return;

    this.isUploading.set(true);
    this.simulateUpload(currentFiles.map(f => f.id));

    // Emit the actual File objects to parent
    const fileObjects = currentFiles.map(file => new File([], file.name, { type: file.type }));
    this.filesSelected.emit(fileObjects);
  }

  private handleFiles(files: File[]): void {
    if (files.length === 0) return;

    // Validate file count
    const currentCount = this.files().length;
    if (currentCount + files.length > this.maxFileCount) {
      alert(`Maximum ${this.maxFileCount} files allowed. You have ${currentCount} files and tried to add ${files.length} more.`);
      return;
    }

    const validFiles: File[] = [];
    const newUploadedFiles: UploadedFile[] = [];

    files.forEach(file => {
      // Validate file type
      if (!this.isFileTypeValid(file)) {
        alert(`File "${file.name}" has an invalid type. Accepted types: ${this.acceptTypes}`);
        return;
      }

      // Validate file size
      if (file.size > this.maxFileSize) {
        alert(`File "${file.name}" exceeds maximum size of ${this.formatBytes(this.maxFileSize)}`);
        return;
      }

      validFiles.push(file);
      
      newUploadedFiles.push({
        id: this.generateId(),
        name: file.name,
        size: file.size,
        type: file.type,
        progress: 0,
        status: 'pending'
      });
    });

    if (validFiles.length > 0) {
      this.files.update(current => [...current, ...newUploadedFiles]);
    }
  }

  private isFileTypeValid(file: File): boolean {
    if (this.acceptTypes === '*/*') return true;
    
    const acceptedTypes = this.acceptTypes.split(',').map(type => type.trim());
    
    // Check file type category
    if (acceptedTypes.some(type => type.includes('/*'))) {
      const category = file.type.split('/')[0] + '/*';
      if (acceptedTypes.includes(category)) return true;
    }
    
    // Check specific extensions
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (fileExtension && acceptedTypes.includes(fileExtension)) return true;
    
    // Check MIME type
    return acceptedTypes.includes(file.type);
  }

  private simulateUpload(fileIds: string[]): void {
    let completed = 0;
    const total = fileIds.length;
    
    fileIds.forEach(fileId => {
      // Simulate upload progress
      let progress = 0;
      const interval = setInterval(() => {
        progress += Math.random() * 10 + 5;
        
        if (progress >= 100) {
          progress = 100;
          clearInterval(interval);
          completed++;
          
          this.updateFileStatus(fileId, 'completed', 100);
          
          if (completed === total) {
            this.isUploading.set(false);
            this.totalUploadProgress.set(100);
          }
        } else {
          this.updateFileStatus(fileId, 'uploading', progress);
          
          this.totalUploadProgress.set(
            Math.round((completed / total) * 100 + (progress / total))
          );
        }
      }, 200);
    });
  }

  private updateFileStatus(fileId: string, status: UploadedFile['status'], progress: number): void {
    this.files.update(files =>
      files.map(file =>
        file.id === fileId ? { ...file, status, progress } : file
      )
    );
  }

  removeFile(fileId: string): void {
    this.files.update(files => files.filter(file => file.id !== fileId));
  }

  clearAllFiles(): void {
    this.files.set([]);
    this.totalUploadProgress.set(0);
    this.isUploading.set(false);
  }

  formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  private generateId(): string {
    return Math.random().toString(36).substring(2) + Date.now().toString(36);
  }

  getFileStatusVariant(status: UploadedFile['status']): 'primary' | 'success' | 'error' | 'warning' | 'neutral' {
    switch (status) {
      case 'completed': return 'success';
      case 'uploading': return 'primary';
      case 'pending': return 'warning';
      case 'error': return 'error';
      default: return 'neutral';
    }
  }

  getFileIcon(type: string): string {
    if (type.startsWith('image/')) return '🖼️';
    if (type.startsWith('video/')) return '🎥';
    if (type.includes('pdf')) return '📄';
    if (type.includes('document') || type.includes('text')) return '📝';
    return '📎';
  }
}