// atoms/buttons/btn-secondary.component.ts
import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-btn-secondary',
  standalone: true,
  imports: [CommonModule],
  template: `
    <button
      class="btn-secondary"
      [class.btn-secondary--small]="size === 'sm'"
      [class.btn-secondary--medium]="size === 'md'"
      [class.btn-secondary--large]="size === 'lg'"
      [class.btn-secondary--disabled]="disabled"
      [disabled]="disabled"
      (click)="onClick.emit($event)"
    >
      <ng-content></ng-content>
    </button>
  `,
  styles: [`
    .btn-secondary {
      background: var(--color-surface-2);
      border: 1px solid var(--color-border);
      color: var(--color-text);
      border-radius: var(--radius-md);
      font-weight: 500;
      cursor: pointer;
      transition: all var(--transition-normal, 200ms);
      
      &:hover:not(:disabled) {
        background: var(--color-surface-3);
        border-color: var(--color-primary);
        transform: translateY(-1px);
      }
      
      &:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
      
      &--small {
        padding: var(--spacing-1) var(--spacing-3);
        font-size: var(--text-sm);
      }
      
      &--medium {
        padding: var(--spacing-2) var(--spacing-4);
        font-size: var(--text-base);
      }
      
      &--large {
        padding: var(--spacing-3) var(--spacing-6);
        font-size: var(--text-lg);
      }
    }
  `]
})
export class BtnSecondaryComponent {
  @Input() size: 'sm' | 'md' | 'lg' = 'md';
  @Input() disabled = false; // Add this line
  @Output() click = new EventEmitter<MouseEvent>();
  
  // Alias for click event to match your pattern
  @Output() onClick = this.click;
}