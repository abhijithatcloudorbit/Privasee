import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-icon-atom',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './icon-atom.component.html',
  styleUrls: ['./icon-atom.component.scss'],
})
export class IconAtomComponent {
  @Input() name: string = 'favorite';
  @Input() size: number = 24;
  @Input() color: string = '#000000';
}
