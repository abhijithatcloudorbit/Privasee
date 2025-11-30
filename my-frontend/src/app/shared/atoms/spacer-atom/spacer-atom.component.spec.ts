import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SpacerAtomComponent } from './spacer-atom.component';

describe('SpacerAtomComponent', () => {
  let component: SpacerAtomComponent;
  let fixture: ComponentFixture<SpacerAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SpacerAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SpacerAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
