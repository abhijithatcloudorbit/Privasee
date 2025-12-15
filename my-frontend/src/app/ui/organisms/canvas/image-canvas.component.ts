import { Component, ElementRef, ViewChild, AfterViewInit, signal, inject, HostListener, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CanvasStateService } from '../../../services/canvas/canvas-state.service';
import { Detection } from '../../../shared/interfaces/canvas.interface';

@Component({
  selector: 'app-image-canvas',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './image-canvas.component.html',
  styleUrls: ['./image-canvas.component.scss']
})
export class ImageCanvasComponent implements AfterViewInit {
  @ViewChild('canvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  
  // Make canvasState public for template access
  canvasState = inject(CanvasStateService);
  
  // Local state
  isDragging = signal(false);
  lastMousePos = signal({ x: 0, y: 0 });
  
  // Template getters
  get hasImage() {
    return () => !!this.canvasState.image();
  }
  
  get scale() {
    return () => this.canvasState.transform().scale;
  }
  
  get offset() {
    return () => ({ 
      x: this.canvasState.transform().x, 
      y: this.canvasState.transform().y 
    });
  }
  
  // Filtered detections (all for now) - use computed signal instead of subscribe
  filteredDetections = computed(() => {
    return this.canvasState.detections();
  });

  ngAfterViewInit(): void {
    this.initializeCanvas();
  }

  private initializeCanvas(): void {
    // Canvas initialization logic
    const canvas = this.canvasRef.nativeElement;
    const ctx = canvas.getContext('2d');
    
    if (ctx && this.canvasState.image()) {
      this.drawImage(ctx);
    }
  }

  private drawImage(ctx: CanvasRenderingContext2D): void {
    const img = new Image();
    img.onload = () => {
      const transform = this.canvasState.transform();
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.save();
      ctx.translate(transform.x, transform.y);
      ctx.scale(transform.scale, transform.scale);
      ctx.drawImage(img, 0, 0);
      ctx.restore();
      
      // Draw detections
      this.drawDetections(ctx);
    };
    img.src = this.canvasState.image()!;
  }

  private drawDetections(ctx: CanvasRenderingContext2D): void {
    const detections = this.canvasState.detections();
    const selectedIds = this.canvasState.selectedDetectionIds();
    const transform = this.canvasState.transform();

    ctx.save();
    ctx.translate(transform.x, transform.y);
    ctx.scale(transform.scale, transform.scale);

    detections.forEach(detection => {
      const isSelected = selectedIds.includes(detection.id);
      const bbox = detection.bbox;

      // Draw detection rectangle
      ctx.strokeStyle = isSelected ? '#ff4757' : this.getColorForType(detection.type);
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);

      // Draw label background
      ctx.fillStyle = this.getColorForType(detection.type);
      ctx.fillRect(bbox.x, bbox.y - 20, 80, 20);

      // Draw label text
      ctx.fillStyle = 'white';
      ctx.font = '12px Arial';
      ctx.fillText(
        `${detection.type} ${Math.round(detection.confidence * 100)}%`,
        bbox.x + 5,
        bbox.y - 5
      );
    });

    ctx.restore();
  }

  // Event handlers
  onZoom(event: WheelEvent): void {
    event.preventDefault();
    const delta = event.deltaY > 0 ? 0.9 : 1.1;
    const rect = (event.target as HTMLElement).getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    this.canvasState.scale(delta, x, y);
    this.initializeCanvas();
  }

  onMouseDown(event: MouseEvent): void {
    this.isDragging.set(true);
    this.lastMousePos.set({ x: event.clientX, y: event.clientY });
  }

  onMouseMove(event: MouseEvent): void {
    if (this.isDragging()) {
      const dx = event.clientX - this.lastMousePos().x;
      const dy = event.clientY - this.lastMousePos().y;
      this.canvasState.pan(dx, dy);
      this.lastMousePos.set({ x: event.clientX, y: event.clientY });
      this.initializeCanvas();
    }
  }

  onMouseUp(event: MouseEvent): void {
    this.isDragging.set(false);
  }

  // Helper methods for template
  getDetectionClass(detection: Detection): string {
    const base = 'detection-rect';
    const selected = this.canvasState.selectedDetectionIds().includes(detection.id) ? 'selected' : '';
    const type = detection.type.replace('_', '-');
    return `${base} ${selected} ${type}`.trim();
  }

  getDetectionLabel(detection: Detection): string {
    return `${detection.type} (${Math.round(detection.confidence * 100)}%)`;
  }

  private getColorForType(type: string): string {
    const colors: Record<string, string> = {
      'face': '#3498db',
      'text': '#2ecc71',
      'license_plate': '#e74c3c',
      'person': '#9b59b6',
      'vehicle': '#f39c12',
      'signature': '#1abc9c'
    };
    return colors[type] || '#95a5a6';
  }

  // Canvas actions
  centerImage(): void {
    this.canvasState.resetTransform();
    this.initializeCanvas();
  }

  zoomIn(): void {
    this.canvasState.scale(1.2);
    this.initializeCanvas();
  }

  zoomOut(): void {
    this.canvasState.scale(0.8);
    this.initializeCanvas();
  }

  // Handle document mouse up to stop dragging when mouse leaves canvas
  @HostListener('document:mouseup')
  onDocumentMouseUp(): void {
    this.isDragging.set(false);
  }

  @HostListener('document:mousemove', ['$event'])
  onDocumentMouseMove(event: MouseEvent): void {
    if (this.isDragging()) {
      this.onMouseMove(event);
    }
  }
}