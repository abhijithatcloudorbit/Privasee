import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-heading-atom',
  standalone: true,
  imports: [CommonModule],   
  templateUrl: './heading-atom.component.html',
  styleUrls: ['./heading-atom.component.scss'],
})
export class HeadingAtomComponent {
  @Input() level: 1 | 2 | 3 | 4 | 5 | 6 = 1;
  @Input() color: string = '#111';
  @Input() align: 'left' | 'center' | 'right' = 'left';
  @Input() weight: number | string = 700;

  get tag() {
    return `h${this.level}` as keyof HTMLElementTagNameMap;
  }
}
