import {
  Component,
  Input,
  HostListener,
  ElementRef,
  Renderer2,
  OnDestroy,
} from '@angular/core';

@Component({
  standalone: true,
  selector: 'app-tooltip-atom',
  templateUrl: './tooltip-atom.component.html',
  styleUrls: ['./tooltip-atom.component.scss'],
})
export class TooltipAtomComponent implements OnDestroy {
  @Input() text = '';
  @Input() position: 'top' | 'bottom' | 'left' | 'right' = 'top';
  @Input() delay = 150; // ms
  @Input() disabled = false;

  visible = false;
  timeoutId!: any;

  constructor(private el: ElementRef, private renderer: Renderer2) {}

  @HostListener('mouseenter')
  onMouseEnter() {
    if (this.disabled) return;

    this.timeoutId = setTimeout(() => {
      this.visible = true;
    }, this.delay);
  }

  @HostListener('mouseleave')
  onMouseLeave() {
    clearTimeout(this.timeoutId);
    this.visible = false;
  }

  @HostListener('focus')
  onFocus() {
    if (!this.disabled) this.visible = true;
  }

  @HostListener('blur')
  onBlur() {
    this.visible = false;
  }

  ngOnDestroy() {
    clearTimeout(this.timeoutId);
  }
}
