import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SliderAtomComponent } from './slider-atom.component';

describe('SliderAtomComponent', () => {
  let component: SliderAtomComponent;
  let fixture: ComponentFixture<SliderAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SliderAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SliderAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
