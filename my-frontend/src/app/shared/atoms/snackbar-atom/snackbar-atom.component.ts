import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

export type SnackbarType = 'success' | 'error' | 'warning' | 'info';
export type SnackbarPosition = 
  'top-left' | 'top-center' | 'top-right' |
  'bottom-left' | 'bottom-center' | 'bottom-right';

@Component({
  selector: 'app-snackbar-atom',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './snackbar-atom.component.html',
  styleUrls: ['./snackbar-atom.component.scss']
})
export class SnackbarAtomComponent {
  @Input() message: string = '';
  @Input() type: SnackbarType = 'info';
  @Input() duration: number = 3000;
  @Input() position: SnackbarPosition = 'bottom-center';
  
  @Input() open: boolean = false;
  @Output() openChange = new EventEmitter<boolean>();

  timeout: any;

  ngOnChanges() {
    if (this.open) this.autoClose();
  }

  autoClose() {
    clearTimeout(this.timeout);
    this.timeout = setTimeout(() => {
      this.close();
    }, this.duration);
  }

  close() {
    this.open = false;
    this.openChange.emit(false);
  }
}
