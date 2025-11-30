import { Component, Input, Output, EventEmitter } from '@angular/core';
import { NgClass } from '@angular/common';
import { FormsModule } from '@angular/forms';

export type SliderAppearance = 'default' | 'minimal' | 'outlined' | 'filled';

@Component({
  standalone: true,
  selector: 'app-slider-atom',
  templateUrl: './slider-atom.component.html',
  styleUrls: ['./slider-atom.component.scss'],
  imports: [NgClass, FormsModule],
})
export class SliderAtomComponent {
  @Input() min: number = 0;
  @Input() max: number = 100;
  @Input() step: number = 1;
  @Input() value: number = 50;

  // NEW — standardized cross-UI property
  @Input() appearance: SliderAppearance = 'default';

  @Input() disabled: boolean = false;

  @Output() valueChange = new EventEmitter<number>();

  onChange(event: Event) {
    const input = event.target as HTMLInputElement;
    this.value = Number(input.value);
    this.valueChange.emit(this.value);
  }
}
