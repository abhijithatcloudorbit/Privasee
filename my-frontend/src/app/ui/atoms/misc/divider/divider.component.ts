import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-divider',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './divider.component.html',
  styleUrls: ['./divider.component.scss']
})
export class DividerComponent {
  @Input() direction: 'horizontal' | 'vertical' = 'horizontal';
  @Input() margin: 'none' | 'sm' | 'md' | 'lg' = 'md';
}