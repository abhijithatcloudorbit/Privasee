import {
  Component,
  Input,
  Output,
  EventEmitter,
  HostListener,
} from '@angular/core';
import { NgIf } from '@angular/common';

@Component({
  standalone: true,
  selector: 'app-modal-atom',
  templateUrl: './modal-atom.component.html',
  styleUrls: ['./modal-atom.component.scss'],
  imports: [NgIf],
})
export class ModalAtomComponent {
  @Input() open: boolean = false;                 // controls modal visibility
  @Input() closeOnBackdrop: boolean = true;       // click outside to close
  @Input() width: string = '480px';               // modal width
  @Input() disableClose: boolean = false;         // disable close actions

  @Output() openChange = new EventEmitter<boolean>();

  closeModal() {
    if (this.disableClose) return;
    this.open = false;
    this.openChange.emit(false);
  }

  // close on ESC
  @HostListener('document:keydown.escape')
  onEsc() {
    if (this.open && !this.disableClose) {
      this.closeModal();
    }
  }
}
