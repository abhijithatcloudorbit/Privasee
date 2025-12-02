import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormControl, ReactiveFormsModule } from '@angular/forms';

@Component({
  selector: 'app-radio-atom',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './radio-atom.component.html',
  styleUrls: ['./radio-atom.component.scss']
})
export class RadioAtomComponent {

  @Input() label: string = '';
  @Input() value: any;
  @Input() color: string = '#2563eb';    // default blue
  @Input() size: number = 18;            // must use [size]="number"

  @Input() control: FormControl<any> = new FormControl(null);

  @Output() changed = new EventEmitter<any>();

  onSelect() {
    this.control.setValue(this.value, { emitEvent: true });
    this.changed.emit(this.value);
  }
}
