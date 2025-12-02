import { Component, Input } from '@angular/core';
import { NgStyle } from '@angular/common';

@Component({
  selector: 'app-text-atom',
  standalone: true,
  imports: [NgStyle],
  templateUrl: './text-atom.component.html',
  styleUrls: ['./text-atom.component.scss']
})
export class TextAtomComponent {

  @Input() size: string = '16px';     // text size
  @Input() color: string = '#111';    // default text color
  @Input() weight: string = '400';    // font weight
  @Input() align: 'left' | 'center' | 'right' = 'left'; // alignment
  @Input() lineHeight: string = '1.5';

}
