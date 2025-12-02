import { Component, Input } from '@angular/core';
import { NgStyle, NgIf } from '@angular/common';

@Component({
  selector: 'app-avatar-atom',
  standalone: true,
  imports: [NgStyle, NgIf],
  templateUrl: './avatar-atom.component.html',
  styleUrls: ['./avatar-atom.component.scss']
})
export class AvatarAtomComponent {

  @Input() src: string | null = null;        // Image source
  @Input() alt: string = 'avatar';           // Alt text
  @Input() size: number = 48;                // Avatar size in px
  @Input() shape: 'circle' | 'rounded' | 'square' = 'circle';
  @Input() initials: string | null = null;   // Fallback initials
  @Input() bg: string = '#e2e8f0';           // Fallback background color
  @Input() color: string = '#1a202c';        // Fallback text color

  imageError: boolean = false;

  onError() {
    this.imageError = true;
  }

  get borderRadius() {
    switch (this.shape) {
      case 'circle': return '50%';
      case 'rounded': return '12px';
      case 'square': 
      default: return '0px';
    }
  }
}
