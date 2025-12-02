import { Component, Input, ContentChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { InputTextComponent } from '../../../atoms/inputs/input-text/input-text.component';
import { TextBodyComponent } from '../../../atoms/typography/text-body/text-body.component';
import { TextCaptionComponent } from '../../../atoms/typography/text-caption/text-caption.component';

@Component({
  selector: 'app-form-field',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './form-field.component.html',
  styleUrls: ['./form-field.component.scss']
})
export class FormFieldComponent {
  @Input() label = '';
  @Input() required = false;
  @Input() helpText = '';
  @Input() error = '';
  
  @ContentChild('inputControl') inputControl: any;
}