import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-chip',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './chip.component.html',
  styleUrls: ['./chip.component.scss']
})
export class ChipComponent {
  @Input() removable = false;
  @Input() selected = false;
  @Input() size: 'sm' | 'md' = 'md';
  
  @Output() removed = new EventEmitter<void>();
  @Output() clicked = new EventEmitter<void>();
  
  onRemove(event: Event): void {
    event.stopPropagation();
    this.removed.emit();
  }
  
  onClick(): void {
    this.clicked.emit();
  }
}