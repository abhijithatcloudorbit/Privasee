import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TextAtomComponent } from './text-atom.component';

describe('TextAtomComponent', () => {
  let component: TextAtomComponent;
  let fixture: ComponentFixture<TextAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TextAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TextAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
