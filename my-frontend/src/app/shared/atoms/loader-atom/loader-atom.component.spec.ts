import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LoaderAtomComponent } from './loader-atom.component';

describe('LoaderAtomComponent', () => {
  let component: LoaderAtomComponent;
  let fixture: ComponentFixture<LoaderAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LoaderAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LoaderAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
