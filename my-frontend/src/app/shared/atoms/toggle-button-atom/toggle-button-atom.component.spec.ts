import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ToggleButtonAtomComponent } from './toggle-button-atom.component';

describe('ToggleButtonAtomComponent', () => {
  let component: ToggleButtonAtomComponent;
  let fixture: ComponentFixture<ToggleButtonAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ToggleButtonAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ToggleButtonAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
