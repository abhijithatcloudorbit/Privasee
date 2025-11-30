import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TextInputAtomComponent } from './text-input-atom.component';

describe('TextInputAtomComponent', () => {
  let component: TextInputAtomComponent;
  let fixture: ComponentFixture<TextInputAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TextInputAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TextInputAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
