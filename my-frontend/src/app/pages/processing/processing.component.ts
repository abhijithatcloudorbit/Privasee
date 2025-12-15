import { Component, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BatchProcessingPanelComponent } from '../../ui/organisms/upload/batch-processing-panel/batch-processing-panel.component';
import { ProcessingCanvasOrganismComponent } from '../../ui/organisms/processing-canvas-organism/processing-canvas-organism.component';
import { DetectionResultsSidebarComponent } from '../../ui/organisms/detection-results-sidebar/detection-results-sidebar.component';
import { BatchProcessingItem } from '../../shared/interfaces/batch.interface';


@Component({
  selector: 'app-processing',
  standalone: true,
  imports: [
    CommonModule,
    BatchProcessingPanelComponent,
    ProcessingCanvasOrganismComponent,
    DetectionResultsSidebarComponent // ADD THIS TO IMPORTS
  ],
  templateUrl: './processing.component.html',
  styleUrls: ['./processing.component.scss']
})
export class ProcessingComponent {
  // Signal for the selected batch and file index
  selectedBatchInfo = signal<{ batch: BatchProcessingItem; fileIndex: number } | null>(null);

  // Computed signal that returns ONLY the BatchProcessingItem for the canvas
  selectedBatch = computed(() => {
    const info = this.selectedBatchInfo();
    if (!info) return null;

    // Return the batch with selectedFile and selectedFileIndex set
    return {
      ...info.batch,
      selectedFileIndex: info.fileIndex,
      selectedFile: info.batch.files[info.fileIndex]
    };
  });

  onBatchSelected(event: { batch: BatchProcessingItem; fileIndex: number }): void {
    this.selectedBatchInfo.set(event);
  }
}