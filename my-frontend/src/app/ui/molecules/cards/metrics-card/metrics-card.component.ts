import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-metrics-preview-card',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './metrics-card.component.html',
  styleUrls: ['./metrics-card.component.scss'],
})
export class MetricsPreviewCardComponent {
  @Input() label!: string;
  @Input() value!: string | number;
  @Input() trend!: number;
  @Input() trendPositive = true;
  @Input() description!: string;
}
