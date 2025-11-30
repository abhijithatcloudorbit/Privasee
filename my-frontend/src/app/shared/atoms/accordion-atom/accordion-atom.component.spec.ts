import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AccordionAtomComponent } from './accordion-atom/accordion-atom.component';

describe('AccordionAtomComponent', () => {
  let component: AccordionAtomComponent;
  let fixture: ComponentFixture<AccordionAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AccordionAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AccordionAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
