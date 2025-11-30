import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TextareaAtomComponent } from './textarea-atom.component';

describe('TextareaAtomComponent', () => {
  let component: TextareaAtomComponent;
  let fixture: ComponentFixture<TextareaAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TextareaAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TextareaAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
