import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TagAtomComponent } from './tag-atom.component';

describe('TagAtomComponent', () => {
  let component: TagAtomComponent;
  let fixture: ComponentFixture<TagAtomComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TagAtomComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TagAtomComponent);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
