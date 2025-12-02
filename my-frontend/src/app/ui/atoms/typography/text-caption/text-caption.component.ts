import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-text-caption',
  standalone: true,
  imports: [CommonModule],
  template: `<span class="text-caption" [class]="color"><ng-content></ng-content></span>`,
  styles: [`
    .text-caption {
      font-size: 0.875rem;
      line-height: 1.25rem;
    }
    .muted { color: var(--color-gray-500); }
  `]
})
export class TextCaptionComponent {
  @Input() color: '' | 'muted' = '';
}