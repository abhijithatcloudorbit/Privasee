import { Component } from '@angular/core';
import { FormControl, FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ViewChild } from '@angular/core';

import { ButtonAtomComponent } from '../shared/atoms/button/button-atom.component';
import { TextInputAtomComponent } from '../shared/atoms/text-input/text-input-atom.component';
import { CardAtomComponent } from '../shared/atoms/card/card-atom.component';
import { IconAtomComponent } from '../shared/atoms/icon/icon-atom.component';
import { HeadingAtomComponent } from '../shared/atoms/heading/heading-atom.component';
import { TextAtomComponent } from '../shared/atoms/text-atom/text-atom.component';
import { AvatarAtomComponent } from '../shared/atoms/avatar-atom/avatar-atom.component';
import { TagAtomComponent } from '../shared/atoms/tag-atom/tag-atom.component';
import { DividerAtomComponent } from '../shared/atoms/divider-atom/divider-atom.component';
import { SpacerAtomComponent } from '../shared/atoms/spacer-atom/spacer-atom.component';
import { LoaderAtomComponent } from '../shared/atoms/loader-atom/loader-atom.component';
import { BadgeAtomComponent } from '../shared/atoms/badge-atom/badge-atom.component';
import { ChipAtomComponent } from '../shared/atoms/chip-atom/chip-atom.component';
import { CheckboxAtomComponent } from '../shared/atoms/checkbox-atom/checkbox-atom.component';
import { RadioAtomComponent } from '../shared/atoms/radio-atom/radio-atom.component';
import { SwitchAtomComponent  } from '../shared/atoms/switch-atom/switch-atom.component';
import { ToggleButtonAtomComponent } from '../shared/atoms/toggle-button-atom/toggle-button-atom.component';
import { TextareaAtomComponent } from '../shared/atoms/textarea-atom/textarea-atom.component';
import { SliderAtomComponent } from '../shared/atoms/slider-atom/slider-atom.component';
import { TooltipAtomComponent } from '../shared/atoms/tooltip-atom/tooltip-atom.component';
import { ModalAtomComponent } from '../shared/atoms/modal-atom/modal-atom.component';
import { SnackbarAtomComponent } from '../shared/atoms/snackbar-atom/snackbar-atom.component';
import { ToastAtomComponent } from '../shared/atoms/toast-atom/toast-atom.component';
import { DropdownAtomComponent } from '../shared/atoms/dropdown-atom/dropdown-atom.component';
import { AccordionAtomComponent } from '../shared/atoms/accordion-atom/accordion-atom.component';
import { TabsAtomComponent } from '../shared/atoms/tabs-atom/tabs-atom.component';



@Component({
  standalone: true,
  selector: 'app-playground',
  templateUrl: './playground.page.html',
  styleUrls: ['./playground.page.scss'],
  imports: [
    FormsModule,
    CommonModule,
    ButtonAtomComponent,
    TextInputAtomComponent,
    CardAtomComponent,
    IconAtomComponent,
    HeadingAtomComponent,
    TextAtomComponent,
    AvatarAtomComponent,
    TagAtomComponent,
    DividerAtomComponent,
    SpacerAtomComponent,
    LoaderAtomComponent,
    BadgeAtomComponent,
    ChipAtomComponent,
    CheckboxAtomComponent,
    RadioAtomComponent,
    SwitchAtomComponent,
    ToggleButtonAtomComponent,
    TextareaAtomComponent,
    SliderAtomComponent,
    TooltipAtomComponent,
    ModalAtomComponent,
    SnackbarAtomComponent,
    ToastAtomComponent,
    DropdownAtomComponent,
    AccordionAtomComponent,
    TabsAtomComponent,
  ],
})
export class PlaygroundPage {

  bioText = '';

  // ⭐ FIXED: add radioControl
  radioControl = new FormControl('option1', { nonNullable: true });

  // Your existing control
  termsControl = new FormControl(false, { nonNullable: true });

  // Switch Value
  switchValue: boolean = false;

  onChipRemoved() {
    console.log('chip removed!');
  }

    selectedMode = 'daily';

  toggleOptions = [
    { label: 'Daily', value: 'daily' },
    { label: 'Weekly', value: 'weekly' },
    { label: 'Monthly', value: 'monthly' } ]

  sliderValue = 30;
  log(event: any) {
  console.log(event);
  }
  console = console;

  // Modal control
    modalOpen = false;

  openModal() {
    this.modalOpen = true;
  }

  closeModal() {
    this.modalOpen = false;
}

  // Snackbar control
  snackbarOpen = false;

  showSnackbar() {
    this.snackbarOpen = true;
  }

  // Toast reference
    toastRef!: ToastAtomComponent;
    @ViewChild(ToastAtomComponent)
toast!: ToastAtomComponent;

showSuccessToast() {
  this.toast.showToast('Profile saved!', 'success');
}

showUndoToast() {
  this.toast.showToast(
    'Item deleted',
    'warning',
    5000,
    'Undo',
    () => console.log('Undo clicked!')
  );
}

  // Dropdown control
  selectedCountry: string | number | null = null;

  onCountryChange(newValue: string | number) {
    this.selectedCountry = newValue;
  }

  //tabs control
  currentTab = 0;
}
