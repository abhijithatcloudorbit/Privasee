import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RadioAtomComponent } from './radio-atom.component';

describe('RadioAtomComponent', () => {
  let component: RadioAtomComponent;
  let fixture: ComponentFixture<RadioAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RadioAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(RadioAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
