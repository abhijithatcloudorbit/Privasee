import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormControl, ReactiveFormsModule } from '@angular/forms';

@Component({
  selector: 'app-checkbox-atom',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './checkbox-atom.component.html',
  styleUrls: ['./checkbox-atom.component.scss'],
})
export class CheckboxAtomComponent {
  @Input() label: string = '';
  @Input() color: string = '#2563eb'; // default blue
  @Input() size: number = 18;         // must be used like [size]="20"

  @Input() control: FormControl<boolean> = new FormControl(false, { nonNullable: true });

  @Output() changed = new EventEmitter<boolean>();

  onChange(e: Event) {
    const target = e.target as HTMLInputElement;
    this.control.setValue(target.checked, { emitEvent: true });
    this.changed.emit(target.checked);
  }
}
