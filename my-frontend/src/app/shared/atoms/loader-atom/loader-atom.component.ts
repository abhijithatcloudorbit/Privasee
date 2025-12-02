import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-loader-atom',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './loader-atom.component.html',
  styleUrls: ['./loader-atom.component.scss']
})
export class LoaderAtomComponent {
  @Input() size: number = 32;      // must be bound like [size]="40"
  @Input() thickness: number = 4;  // spinner border thickness
  @Input() color: string = '#2563eb'; // default blue
}
