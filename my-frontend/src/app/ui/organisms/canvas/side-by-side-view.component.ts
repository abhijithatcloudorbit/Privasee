import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CanvasStateService } from '../../../services/canvas/canvas-state.service';

@Component({
  selector: 'app-side-by-side-view',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './side-by-side-view.component.html',
  styleUrls: ['./side-by-side-view.component.scss']
})
export class SideBySideViewComponent {
  private canvasState = inject(CanvasStateService);
  
  // Signals for split view
  splitPosition = signal(50); // Default 50% split
  showOriginal = signal(true);
  isDragging = signal(false);
  
  // Get transform from canvas state
  get transform() {
    return this.canvasState.transform;
  }
  
  // View manipulation methods
  zoomIn(): void {
    this.canvasState.scale(1.2);
  }

  zoomOut(): void {
    this.canvasState.scale(0.8);
  }

  panLeft(): void {
    this.canvasState.pan(-50, 0);
  }

  panRight(): void {
    this.canvasState.pan(50, 0);
  }

  panUp(): void {
    this.canvasState.pan(0, -50);
  }

  panDown(): void {
    this.canvasState.pan(0, 50);
  }

  resetView(): void {
    this.canvasState.resetTransform();
  }

  resetSplit(): void {
    this.splitPosition.set(50);
  }

  toggleView(): void {
    this.showOriginal.update(value => !value);
  }

  startDrag(event: MouseEvent): void {
    this.isDragging.set(true);
    event.preventDefault();
  }

  updateSplitPosition(event: MouseEvent, container: HTMLElement): void {
    if (this.isDragging()) {
      const rect = container.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const percentage = (x / rect.width) * 100;
      this.splitPosition.set(Math.max(10, Math.min(90, percentage)));
    }
  }

  stopDrag(): void {
    this.isDragging.set(false);
  }
}