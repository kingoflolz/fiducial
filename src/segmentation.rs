use image::{ImageBuffer, Luma};
use std::collections::VecDeque;
use std::mem;

#[cfg(test)]
use imageproc::gray_image;

type Label = u32;

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureVector {
    pub area: u32,
    pub bounding_box: [(u32, u32); 2],
    pub max_x: u32,
    pub valid: bool,
    pub merge: AugmentedLabel,
    pub child: Option<u32>,
    pub last_child: u32,
    pub sibling: Option<u32>,
    pub last_sibling: u32,
    pub color: bool,
    pub visited: bool,
    pub fom: [f32; 2], // TODO: use fixed point
    pub zom: f32,
}

impl FeatureVector {
    fn new(x: u32, y: u32, value: u32, color: bool, merge: AugmentedLabel) -> FeatureVector {
        let actual_value = if color { value } else { 255 - value };

        FeatureVector {
            max_x: x,
            area: 1,
            bounding_box: [(x, y), (x, y)],
            valid: true,
            merge,
            child: None,
            last_child: 0,
            sibling: None,
            last_sibling: 0,
            color,
            visited: false,
            fom: [(x * actual_value) as f32, (y * actual_value) as f32],
            zom: actual_value as f32,
        }
    }

    pub fn get_com(&self) -> (i32, i32) {
        (
            (self.fom[0] / self.zom) as i32,
            (self.fom[1] / self.zom) as i32,
        )
    }

    pub fn get_com_f32(&self) -> (f32, f32) {
        (self.fom[0] / self.zom, self.fom[1] / self.zom)
    }

    pub(crate) fn add_pixel_no_area(&mut self, x: u32, y: u32, value: u32, color: bool) {
        let actual_value = if color { value } else { 255 - value };
        self.fom[0] += (x * actual_value) as f32;
        self.fom[1] += (y * actual_value) as f32;
        self.zom += actual_value as f32;
    }

    fn add_pixel(&mut self, x: u32, y: u32, value: u32, color: bool) {
        let actual_value = if color { value } else { 255 - value };

        //assert!(self.valid);
        self.area += 1;
        self.fom[0] += (x * actual_value) as f32;
        self.fom[1] += (y * actual_value) as f32;
        self.zom += actual_value as f32;

        // extending bb
        if y > self.bounding_box[1].1 {
            self.max_x = x;
        } else {
            self.max_x = u32::max(x, self.max_x);
        }

        self.bounding_box[0].0 = u32::min(x, self.bounding_box[0].0);
        self.bounding_box[0].1 = u32::min(y, self.bounding_box[0].1);
        self.bounding_box[1].0 = u32::max(x, self.bounding_box[1].0);
        self.bounding_box[1].1 = u32::max(y, self.bounding_box[1].1);
    }

    fn merge(&mut self, other: &FeatureVector) {
        self.fom[1] += other.fom[1];
        self.fom[0] += other.fom[0];
        self.zom += other.zom;

        if self.bounding_box[1].1 == other.bounding_box[1].1 + 1 {
            // if self is lower by one
            self.max_x = self.max_x;
        } else if self.bounding_box[1].1 + 1 == other.bounding_box[1].1 {
            // if other is lower by one
            self.max_x = other.max_x;
        } else if self.bounding_box[1].1 == other.bounding_box[1].1 {
            self.max_x = u32::max(self.max_x, other.max_x);
        } else if self.bounding_box[1].1 < other.bounding_box[1].1 {
            println!("rip");
        }

        self.area += other.area;
        self.fom[0] += other.fom[0];
        self.fom[1] += other.fom[1];
        self.zom += other.zom;
        self.bounding_box[0].0 = u32::min(other.bounding_box[0].0, self.bounding_box[0].0);
        self.bounding_box[0].1 = u32::min(other.bounding_box[0].1, self.bounding_box[0].1);
        self.bounding_box[1].0 = u32::max(other.bounding_box[1].0, self.bounding_box[1].0);
        self.bounding_box[1].1 = u32::max(other.bounding_box[1].1, self.bounding_box[1].1);
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Copy)]
pub struct AugmentedLabel {
    label: Label,
    row: u16,
}

impl AugmentedLabel {
    fn bg() -> AugmentedLabel {
        AugmentedLabel { label: 0, row: 0 }
    }
}

/// returns "oldest" id first
fn sort_aug_label(a: AugmentedLabel, b: AugmentedLabel) -> (AugmentedLabel, AugmentedLabel) {
    assert_ne!(a, b);
    if a.row < b.row || (a.row == b.row && a.label < b.label) {
        (a, b)
    } else {
        (b, a)
    }
}

fn mut_two<T>(first_index: usize, second_index: usize, items: &mut [T]) -> (&mut T, &mut T) {
    assert!(first_index != second_index);
    let split_at_index = if first_index < second_index {
        second_index
    } else {
        first_index
    };
    let (first_slice, second_slice) = items.split_at_mut(split_at_index);
    if first_index < second_index {
        (&mut first_slice[first_index], &mut second_slice[0])
    } else {
        (&mut second_slice[0], &mut first_slice[second_index])
    }
}

#[cfg(test)]
fn print_img(input: &ImageBuffer<Luma<u8>, Vec<u8>>) {
    println!("image:");
    let (x, y) = input.dimensions();
    for j in 0..y {
        for i in 0..x {
            if input.get_pixel(i, j).data[0] > 0 {
                print!("X")
            } else {
                print!(" ")
            }
        }
        println!("");
    }
}

pub fn to_topo(
    input: &ImageBuffer<Luma<u8>, Vec<u8>>,
    input_grey: &ImageBuffer<Luma<u8>, Vec<u8>>,
) -> Vec<FeatureVector> {
    //print_img(input);
    let mut final_output: Vec<FeatureVector> = Vec::new();

    let mut active_components: Vec<FeatureVector> = Vec::new();

    let mut label_fifo = VecDeque::new();

    for i in 1..input.width() + 10 {
        label_fifo.push_back(AugmentedLabel { label: i, row: 0 });
    }

    for _ in 1..input.width() + 11 {
        active_components.push(FeatureVector {
            area: 0,
            bounding_box: [(0, 0), input.dimensions()],
            max_x: input.width(),
            valid: true,
            merge: AugmentedLabel { label: 0, row: 0 },
            child: None,
            last_child: 0,
            sibling: None,
            last_sibling: 0,
            color: false,
            visited: false,
            fom: [0.0f32, 0.0f32],
            zom: 0.0,
        });
    }

    let mut merger_stack = Vec::new();
    //let mut label_stack = Vec::new();

    let mut prev_row_label = vec![AugmentedLabel::bg(); input.dimensions().0 as usize];
    let mut current_row_label: Vec<AugmentedLabel> = vec![];
    for i in input.enumerate_pixels() {
        let mut current_label;

        // update Lc with M[L[C]]
        if i.1 > 0 && i.0 < input.width() - 1 {
            prev_row_label[(i.0 + 1) as usize] =
                active_components[prev_row_label[(i.0 + 1) as usize].label as usize].merge;
        }

        // update neighborhood
        let current = i.2.data[0] > 0;
        let value = input_grey.get_pixel(i.0, i.1).data[0];
        let top = if i.1 > 0 {
            input.get_pixel(i.0, i.1 - 1).data[0] > 0
        } else {
            false
        };
        let top_label = active_components[prev_row_label[i.0 as usize].label as usize].merge;
        //let top_label_raw = prev_row_label[i.0 as usize];

        let left = if i.0 > 0 {
            input.get_pixel(i.0 - 1, i.1).data[0] > 0
        } else {
            false
        };

        //assert!(active_components.len() > 0);
        //assert!(current_row_label.len() > 0 || i.0 == 0);
        //let left_label_m = active_components[if i.0 > 0 {current_row_label[(i.0-1) as usize].label} else { 0 } as usize].merge;
        let left_label = if i.0 > 0 {
            current_row_label[(i.0 - 1) as usize]
        } else {
            AugmentedLabel { label: 0, row: 0 }
        };

        if current {
            //println!("black")
        }

        // update data structures
        if current == left && current == top {
            // either add new pix or merge
            // adding pixel
            let top_component = &active_components[top_label.label as usize];

            if top_component.valid {
                if top_label == left_label {
                    current_label = left_label;
                    active_components[current_label.label as usize].add_pixel(
                        i.0,
                        i.1,
                        value as u32,
                        current,
                    );
                } else {
                    // to merge, keep oldest ID, free newest ID
                    let (merged_idx, free_idx) = sort_aug_label(top_label, left_label);

                    let (merged_fv, free_fv) = mut_two(
                        merged_idx.label as usize,
                        free_idx.label as usize,
                        &mut active_components,
                    );

                    merged_fv.merge(free_fv);
                    merged_fv.add_pixel(i.0, i.1, value as u32, current);

                    //let merged_child = merged_fv.child;
                    //let merged_sib = merged_fv.sibling;

                    match (free_fv.child, merged_fv.child) {
                        (Some(_free), Some(merged)) => {
                            merged_fv.child = free_fv.child;
                            if final_output[free_fv.last_child as usize].sibling.is_some() {
                                assert!(final_output[final_output[free_fv.last_child as usize]
                                    .last_sibling
                                    as usize]
                                    .sibling
                                    .is_none());
                                let last_last_sibling =
                                    final_output[free_fv.last_child as usize].last_sibling as usize;
                                final_output[last_last_sibling].sibling = Some(merged);
                                final_output[last_last_sibling].last_sibling = merged;
                            } else {
                                final_output[free_fv.last_child as usize].sibling = Some(merged);
                                final_output[free_fv.last_child as usize].last_sibling =
                                    merged_fv.last_child;
                            }
                        }
                        (Some(_free), None) => {
                            merged_fv.child = free_fv.child;
                            merged_fv.last_child = free_fv.last_child;
                        }
                        _ => {}
                    }

                    match (free_fv.sibling, merged_fv.sibling) {
                        (Some(_free), Some(_merged)) => {
                            unreachable!();
                            // merged_fv.sibling = free_fv.sibling;
                            // if final_output[free_fv.last_sibling as usize].sibling.is_none() {
                            //     final_output[free_fv.last_sibling as usize].last_sibling = merged;
                            // }
                            // final_output[free_fv.last_sibling as usize].sibling = Some(merged);
                        }
                        (Some(free), None) => {
                            merged_fv.sibling = free_fv.sibling;
                            merged_fv.last_sibling = free;
                        }
                        _ => {}
                    }

                    if free_fv.valid {
                        free_fv.valid = false;
                        label_fifo.push_back(free_idx);

                        if merged_idx == top_label {
                            merger_stack.push((free_idx, merged_idx));
                        }
                    }

                    //*free_fv = merged_fv.clone();

                    free_fv.bounding_box = [(u32::MAX, u32::MAX), (0, 0)];
                    free_fv.area = 0;
                    free_fv.zom = 0f32;
                    free_fv.fom = [0f32, 0f32];
                    free_fv.merge = merged_fv.merge;
                    free_fv.valid = false;

                    current_label = merged_fv.merge;
                }
            } else {
                // we should be able to recover the root from the stack
                current_label = left_label;
                active_components[current_label.label as usize].add_pixel(
                    i.0,
                    i.1,
                    value as u32,
                    current,
                );
            }
        } else if current == left {
            // add to region
            current_label = left_label;
            active_components[left_label.label as usize].add_pixel(i.0, i.1, value as u32, current);
        } else if current == top {
            // add to region
            current_label = top_label;
            active_components[top_label.label as usize].add_pixel(i.0, i.1, value as u32, current);
        } else {
            // create region
            current_label = label_fifo.pop_front().unwrap();
            current_label.row = i.1 as u16;

            active_components[current_label.label as usize] =
                FeatureVector::new(i.0, i.1, value as u32, current, current_label);
        }

        // enclosing logic
        if top != current {
            // possible closing
            let top_component = &active_components[top_label.label as usize];

            if top_component.bounding_box[1].1 < i.1 {
                // ensure not on the same level

                if top_component.max_x <= i.0 {
                    // defs enclosing
                    final_output.push((*top_component).clone());

                    let current_component = &mut active_components[current_label.label as usize];
                    current_component.zom += final_output.last().unwrap().zom;
                    current_component.fom[0] += final_output.last().unwrap().fom[0];
                    current_component.fom[1] += final_output.last().unwrap().fom[1];

                    assert!(current_component.valid);

                    if let Some(cc_child) = current_component.child {
                        assert!(final_output.last_mut().unwrap().sibling.is_none());
                        final_output.last_mut().unwrap().sibling = Some(cc_child);
                    }

                    if current_component.child == None {
                        current_component.last_child = (final_output.len() - 1) as u32
                    }

                    current_component.child = Some((final_output.len() - 1) as u32);

                    label_fifo.push_back(top_label);
                    //println!("enclosing size {} at {}, {}", top_component.area, i.0, i.1);
                }
            }
        }

        current_row_label.push(current_label);

        if i.0 == input.dimensions().0 - 1 {
            // end of row operations
            while let Some((freed, merged)) = merger_stack.pop() {
                active_components[freed.label as usize].merge =
                    active_components[merged.label as usize].merge
            }

            mem::swap(&mut prev_row_label, &mut current_row_label);
            // for i in 0..input.width() {
            //     print!("{}", current_row_label[i as usize].label)
            // }
            current_row_label.clear();
        }
    }

    final_output.push(active_components[0].clone());

    final_output
}

pub fn check_invariants(topo: &Vec<FeatureVector>, total_area: u32) {
    assert_eq!(topo.iter().map(|x| x.area).sum::<u32>(), total_area);
}

pub fn area(topo: &mut Vec<FeatureVector>, start: u32) -> u32 {
    assert!(!topo[start as usize].visited);
    topo[start as usize].visited = true;
    let mut child = topo[start as usize].child;
    let mut sum = topo[start as usize].area;

    while let Some(c) = child {
        sum += area(topo, c);
        child = topo[c as usize].sibling;
    }
    return sum;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::add_border;
    use image::ImageBuffer;

    #[quickcheck]
    fn qc_check_identities(input: Vec<bool>) -> bool {
        let length = input.len();

        if length < 9 {
            return true;
        }

        let img_size = (length as f32).sqrt().ceil() as u32 + 2;

        let mut image: ImageBuffer<Luma<u8>, Vec<u8>> =
            ImageBuffer::from_raw(img_size, img_size, vec![0; (img_size * img_size) as usize])
                .unwrap();

        let mut idx = 0;
        for i in 0..img_size - 2 {
            for j in 0..img_size - 2 {
                if idx < length {
                    image.put_pixel(
                        i + 1,
                        j + 1,
                        Luma {
                            data: [if input[idx] { 255 } else { 0 }],
                        },
                    );
                    idx += 1;
                }
            }
        }

        add_border(&mut image);
        let mut fv = to_topo(&image, &image);
        println!(
            "area: {}, size: {}",
            fv.iter().map(|v| v.area).sum::<u32>(),
            img_size * img_size
        );

        let len = (fv.len() - 1) as u32;
        let area = area(&mut fv, len as u32);

        println!("area: {}", area);
        return area == img_size * img_size;
    }

    use rand::prelude::*;
    use test::Bencher;

    #[bench]
    fn bench_topo(b: &mut Bencher) {
        let width = 1280;
        let height = 960;

        let mut rng = thread_rng();

        let mut flat_array = Vec::new();
        for _ in 0..width * height {
            if rng.gen::<f32>() > 0.9f32 {
                flat_array.push(0);
            } else {
                flat_array.push(255);
            }
        }
        let mut image =
            ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, flat_array).unwrap();
        add_border(&mut image);

        b.iter(|| to_topo(&image, &image));
    }
}
