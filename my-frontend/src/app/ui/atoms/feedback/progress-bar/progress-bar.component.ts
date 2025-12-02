import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-progress-bar',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './progress-bar.component.html',
  styleUrls: ['./progress-bar.component.scss']
})
export class ProgressBarComponent {
  @Input() value = 0;
  @Input() max = 100;
  @Input() showLabel = false;
  @Input() variant: 'primary' | 'success' | 'warning' | 'error' = 'primary';
  @Input() size: 'sm' | 'md' | 'lg' = 'md';
  
  get percentage(): number {
    return Math.min(100, Math.max(0, (this.value / this.max) * 100));
  }
}