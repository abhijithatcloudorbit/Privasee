import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CardAtomComponent } from './card-atom.component';

describe('CardAtomComponent', () => {
  let component: CardAtomComponent;
  let fixture: ComponentFixture<CardAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CardAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CardAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
