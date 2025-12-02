import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-text-body',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './text-body.component.html',
  styleUrls: ['./text-body.component.scss']
})
export class TextBodyComponent {
  @Input() size: 'sm' | 'md' | 'lg' = 'md';
  @Input() color: 'default' | 'muted' | 'primary' = 'default';
  @Input() weight: 'normal' | 'medium' | 'semibold' = 'normal';
  @Input() class = '';
}