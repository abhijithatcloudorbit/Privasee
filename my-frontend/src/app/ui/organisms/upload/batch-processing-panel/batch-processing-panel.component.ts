import { Component, signal, computed, output, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FileUploadService } from '../../../../services/utilities/file-upload';
import { BatchProcessingItem } from '../../../../shared/interfaces/batch.interface';
//import { BatchFile } from '../../../../shared/interfaces/batch.interface';


@Component({
  selector: 'app-batch-processing-panel',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './batch-processing-panel.component.html',
  styleUrls: ['./batch-processing-panel.component.scss']
})
export class BatchProcessingPanelComponent {
  batchSelected = output<{ batch: BatchProcessingItem; fileIndex: number }>();

  // Inject service
  private fileUploadService = inject(FileUploadService);
  
  // Expose service signals
  batches = this.fileUploadService.batches;
  completedBatches = this.fileUploadService.completedBatches;
  overallProgress = this.fileUploadService.overallProgress;

  // Local state
  selectedBatchId = signal<string | null>(null);
  expandedBatches = signal<Record<string, boolean>>({});

  // Computed values
  processingBatchesCount = computed(() => 
    this.batches().filter(b => b.status === 'processing').length
  );
  
  completedBatchesCount = computed(() => this.completedBatches().length);
  
  pendingBatches = computed(() => 
    this.batches().filter(b => b.status === 'pending').length
  );
  
  totalBatches = computed(() => this.batches().length);

  // Batch operations
  startBatchProcessing(batchId: string): void {
    console.log('Starting batch:', batchId);
  }

  pauseBatchProcessing(batchId: string): void {
    console.log('Pausing batch:', batchId);
  }

  resumeBatchProcessing(batchId: string): void {
    console.log('Resuming batch:', batchId);
  }

  removeBatch(batchId: string): void {
    console.log('Removing batch:', batchId);
  }

  clearCompletedBatches(): void {
    console.log('Clearing completed batches');
  }

  // Selection methods
  onBatchSelected(batch: BatchProcessingItem, fileIndex: number = 0): void {
    this.selectedBatchId.set(batch.id);
    this.batchSelected.emit({ batch, fileIndex });
  }

  onFileSelected(batch: BatchProcessingItem, fileIndex: number): void {
    this.batchSelected.emit({ batch, fileIndex });
  }

  toggleBatchExpansion(batchId: string): void {
    this.expandedBatches.update(current => ({
      ...current,
      [batchId]: !current[batchId]
    }));
  }

  isBatchExpanded(batchId: string): boolean {
    return this.expandedBatches()[batchId] || false;
  }

  isBatchSelected(batchId: string): boolean {
    return this.selectedBatchId() === batchId;
  }

  isFileSelected(batchId: string, fileIndex: number): boolean {
    const batch = this.batches().find(b => b.id === batchId);
    return batch ? batch.selectedFileIndex === fileIndex : false;
  }

  // Helper methods
  getBatchStatusIcon(status: string): string {
    const icons: Record<string, string> = {
      'pending': '⏳',
      'processing': '🔄',
      'completed': '✅',
      'failed': '❌',
      'paused': '⏸️'
    };
    return icons[status] || '❓';
  }

  getBatchStatusColor(status: string): string {
    const colors: Record<string, string> = {
      'pending': 'gray',
      'processing': 'blue',
      'completed': 'green',
      'failed': 'red',
      'paused': 'orange'
    };
    return colors[status] || 'gray';
  }

  formatTime(date: Date): string {
    return new Date(date).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  getFileIcon(type: string): string {
    if (type.includes('image')) return '🖼️';
    if (type.includes('pdf')) return '📄';
    return '📎';
  }
}