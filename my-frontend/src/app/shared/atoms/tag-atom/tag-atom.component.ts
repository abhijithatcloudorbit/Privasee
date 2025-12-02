import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-tag-atom',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './tag-atom.component.html',
  styleUrls: ['./tag-atom.component.scss'],
})
export class TagAtomComponent {
  @Input() label: string = '';
  @Input() color: string = '#2563eb';               // default blue
  @Input() variant: 'solid' | 'outline' = 'solid';
  @Input() size: 'sm' | 'md' | 'lg' = 'md';
  @Input() removable: boolean = false;

  @Output() removed = new EventEmitter<void>();

  removeTag() {
    this.removed.emit();
  }
}
