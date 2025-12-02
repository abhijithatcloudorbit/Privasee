import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-toast',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './toast.component.html',
  styleUrls: ['./toast.component.scss']
})
export class ToastComponent {
  @Input() type: 'info' | 'success' | 'warning' | 'error' = 'info';
  @Input() message = '';
  @Input() title = '';
  @Input() dismissible = true;
  @Input() visible = true;
  
  close(): void {
    this.visible = false;
  }
  
  get icon(): string {
    switch (this.type) {
      case 'success': return '✓';
      case 'warning': return '⚠';
      case 'error': return '✗';
      default: return 'ℹ';
    }
  }
}