import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-text',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './text.component.html',
  styleUrls: ['./text.component.scss']
})
export class TextComponent {
  @Input() size: 'xs' | 'sm' | 'base' | 'lg' | 'xl' = 'base';
  @Input() weight: 'normal' | 'medium' | 'semibold' | 'bold' = 'normal';
  @Input() color: 'default' | 'muted' | 'primary' | 'error' = 'default';
  @Input() class = '';
}