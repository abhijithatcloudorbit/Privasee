import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CheckboxAtomComponent } from './checkbox-atom.component';

describe('CheckboxAtomComponent', () => {
  let component: CheckboxAtomComponent;
  let fixture: ComponentFixture<CheckboxAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CheckboxAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CheckboxAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
