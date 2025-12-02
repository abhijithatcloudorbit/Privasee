import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-input-toggle',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './input-toggle.component.html',
  styleUrls: ['./input-toggle.component.scss']
})
export class InputToggleComponent {
  @Input() label = '';
  @Input() checked = false;
  @Input() disabled = false;
  @Input() size: 'sm' | 'md' | 'lg' = 'md';
  
  @Output() toggle = new EventEmitter<boolean>();
  
  onToggle(): void {
    if (this.disabled) return;
    
    this.checked = !this.checked;
    this.toggle.emit(this.checked);
  }
}