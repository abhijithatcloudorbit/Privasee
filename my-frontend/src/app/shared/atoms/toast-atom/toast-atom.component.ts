import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface ToastData {
  id: number;
  message: string;
  type: 'success' | 'error' | 'warning' | 'info';
  duration: number;
  actionText?: string;
  action?: () => void;
}

@Component({
  selector: 'app-toast-atom',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './toast-atom.component.html',
  styleUrls: ['./toast-atom.component.scss'],
})
export class ToastAtomComponent {
  toasts: ToastData[] = [];
  nextId = 1;

  /** Call this from parent (Playground) */
  showToast(
    message: string,
    type: ToastData['type'] = 'info',
    duration = 3000,
    actionText?: string,
    action?: () => void
  ) {
    const id = this.nextId++;

    this.toasts.push({
      id,
      message,
      type,
      duration,
      actionText,
      action,
    });

    setTimeout(() => this.removeToast(id), duration);
  }

  removeToast(id: number) {
    this.toasts = this.toasts.filter(t => t.id !== id);
  }

  onAction(toast: ToastData) {
    toast.action?.();
    this.removeToast(toast.id);
  }
}
