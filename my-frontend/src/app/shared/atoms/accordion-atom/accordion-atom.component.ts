import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  standalone: true,
  selector: 'app-accordion-atom',
  imports: [CommonModule],
  templateUrl: './accordion-atom.component.html',
  styleUrls: ['./accordion-atom.component.scss'],
})
export class AccordionAtomComponent {
  @Input() label: string = '';
  @Input() open: boolean = false;
  @Input() disabled: boolean = false;

  @Output() openChange = new EventEmitter<boolean>();

  toggle() {
    if (this.disabled) return;
    this.open = !this.open;
    this.openChange.emit(this.open);
  }
}
