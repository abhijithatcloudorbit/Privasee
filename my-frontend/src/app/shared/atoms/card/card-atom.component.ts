import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-card-atom',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './card-atom.component.html',
  styleUrls: ['./card-atom.component.scss'],
})
export class CardAtomComponent {
  @Input() padding: string = '16px';
  @Input() shadow: boolean = true;
  @Input() border: boolean = false;
}
