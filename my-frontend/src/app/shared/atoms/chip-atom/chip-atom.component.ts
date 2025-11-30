import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { IconAtomComponent } from '../icon/icon-atom.component'; // adjust path if needed

@Component({
  selector: 'app-chip-atom',
  standalone: true,
  imports: [CommonModule, IconAtomComponent],
  templateUrl: './chip-atom.component.html',
  styleUrls: ['./chip-atom.component.scss']
})
export class ChipAtomComponent {

  @Input() label: string = '';
  @Input() color: string = '#2563eb';
  @Input() textColor: string = '#ffffff';
  @Input() variant: 'solid' | 'outline' = 'solid';
  @Input() size: 'sm' | 'md' | 'lg' = 'md';
  @Input() removable: boolean = false;    // must use [removable]="true"
  @Input() icon: string | null = null;    // Material icon name if needed

  @Output() clicked = new EventEmitter<void>();
  @Output() removed = new EventEmitter<void>();

  onClick() {
    this.clicked.emit();
  }

  onRemove(e: Event) {
    e.stopPropagation();
    this.removed.emit();
  }

  get classes() {
    return {
      ['chip-' + this.size]: true,
      ['chip-outline']: this.variant === 'outline',
    };
  }
}
