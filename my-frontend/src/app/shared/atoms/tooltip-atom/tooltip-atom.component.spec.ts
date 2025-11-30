import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TooltipAtomComponent } from './tooltip-atom.component';

describe('TooltipAtomComponent', () => {
  let component: TooltipAtomComponent;
  let fixture: ComponentFixture<TooltipAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TooltipAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TooltipAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
