import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-batch-processing-panel',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './batch-processing-panel.component.html',
  styleUrls: ['./batch-processing-panel.component.scss']
})
export class BatchProcessingPanelComponent {
  @Input() files: any[] = [];  // ← ADD THIS
}