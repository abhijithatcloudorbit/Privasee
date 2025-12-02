import { Component, Input, ElementRef, ViewChild, AfterViewInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { TextCaptionComponent } from '../../../atoms/typography/text-caption/text-caption.component';

@Component({
  selector: 'app-image-comparison-slider',
  standalone: true,
  imports: [CommonModule, TextCaptionComponent],
  templateUrl: './image-comparison-slider.component.html',
  styleUrls: ['./image-comparison-slider.component.scss']
})
export class ImageComparisonSliderComponent implements AfterViewInit, OnDestroy {
  @Input() beforeImage = '';
  @Input() afterImage = '';
  @Input() beforeLabel = 'Original';
  @Input() afterLabel = 'Processed';
  @Input() initialPosition = 50;
  
  @ViewChild('slider') slider!: ElementRef<HTMLDivElement>;
  @ViewChild('container') container!: ElementRef<HTMLDivElement>;
  
  position = this.initialPosition;
  isDragging = false;
  
  ngAfterViewInit(): void {
    this.updateSliderPosition();
  }
  
  ngOnDestroy(): void {
    this.removeEventListeners();
  }
  
  onMouseDown(event: MouseEvent): void {
    this.isDragging = true;
    this.updatePosition(event);
    this.addEventListeners();
  }
  
  onTouchStart(event: TouchEvent): void {
    this.isDragging = true;
    this.updatePosition(event.touches[0]);
    this.addEventListeners();
  }
  
  onMouseMove(event: MouseEvent): void {
    if (this.isDragging) {
      this.updatePosition(event);
    }
  }
  
  onTouchMove(event: TouchEvent): void {
    if (this.isDragging && event.touches.length > 0) {
      this.updatePosition(event.touches[0]);
    }
  }
  
  onMouseUp(): void {
    this.isDragging = false;
    this.removeEventListeners();
  }
  
  private updatePosition(event: MouseEvent | Touch): void {
    const container = this.container.nativeElement;
    const rect = container.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const percent = Math.max(0, Math.min(100, (x / rect.width) * 100));
    
    this.position = percent;
    this.updateSliderPosition();
  }
  
  private updateSliderPosition(): void {
    if (this.slider) {
      this.slider.nativeElement.style.left = `${this.position}%`;
    }
  }
  
  private addEventListeners(): void {
    document.addEventListener('mousemove', this.onMouseMove.bind(this));
    document.addEventListener('mouseup', this.onMouseUp.bind(this));
    document.addEventListener('touchmove', this.onTouchMove.bind(this));
    document.addEventListener('touchend', this.onMouseUp.bind(this));
  }
  
  private removeEventListeners(): void {
    document.removeEventListener('mousemove', this.onMouseMove.bind(this));
    document.removeEventListener('mouseup', this.onMouseUp.bind(this));
    document.removeEventListener('touchmove', this.onTouchMove.bind(this));
    document.removeEventListener('touchend', this.onMouseUp.bind(this));
  }
}