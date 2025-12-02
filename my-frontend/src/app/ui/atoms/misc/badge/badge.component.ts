import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-badge',
  standalone: true,
  imports: [CommonModule],
  template: `
    <span class="badge" [class]="variant + ' ' + size">
      <ng-content></ng-content>
    </span>
  `,
  styles: [`
    .badge {
      display: inline-flex;
      align-items: center;
      padding: 0.25rem 0.5rem;
      border-radius: 0.375rem;
      font-size: 0.75rem;
      font-weight: 500;
      line-height: 1;
    }
    .primary { background-color: var(--color-primary); color: white; }
    .success { background-color: var(--color-success); color: white; }
    .error { background-color: var(--color-error); color: white; }
    .warning { background-color: var(--color-warning); color: white; }
    .gray { background-color: var(--color-gray-500); color: white; }
    .sm { font-size: 0.625rem; padding: 0.125rem 0.375rem; }
  `]
})
export class BadgeComponent {
  @Input() variant: 'primary' | 'success' | 'error' | 'warning' | 'gray' = 'primary';
  @Input() size: 'sm' | 'md' = 'md';
}