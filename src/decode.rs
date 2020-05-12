use crate::segmentation::FeatureVector;
use euclid::default::Vector2D;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum TagComponents {
    Root {
        nodes: Vec<TagComponents>,
        class: TagClass,
    },
    None,
    Baseline {
        nodes: [FeatureVector; 2],
    },
    Normal {
        data: bool,
        node: FeatureVector,
    },
}

pub struct DecodedTag {
    pub data: usize,
    pub nodes: Vec<TagComponents>,
    pub top_right: usize,
    pub top_left: usize,
    pub bottom_left: usize,
    pub node_pos: Vec<(f32, f32)>,
    pub class: TagClass
}

impl DecodedTag {
    pub fn decode_tag(root: &TagComponents) -> Option<DecodedTag> {
        if let TagComponents::Root { nodes, class } = root {
            if let TagComponents::Baseline {
                nodes: baseline_nodes,
            } = &nodes[0]
            {
                let mut node_pos = Vec::new();
                let mut top_right = &baseline_nodes[0];
                let mut top_right_id = 0;
                let mut min_angle = 999.9f32;
                let mut top_left = &baseline_nodes[0];
                let mut top_left_id = 0;

                for (idx, i) in nodes.iter().enumerate() {
                    if let TagComponents::Normal { node, .. } = i {
                        let baseline0: Vector2D<_> = baseline_nodes[0].get_com_f32().into();
                        let baseline1: Vector2D<_> = baseline_nodes[1].get_com_f32().into();
                        let node_vec: Vector2D<_> = node.get_com_f32().into();
                        let base_vec0 = baseline0 - node_vec;
                        let base_vec1 = baseline1 - node_vec;

                        let average_length = (base_vec0 + base_vec1) / 2.0f32;

                        let angle = (base_vec0.angle_to(base_vec1).radians.abs() + 0.02)
                            / average_length.length();
                        if angle < min_angle {
                            top_right_id = idx;
                            min_angle = angle;
                            top_right = node;

                            if base_vec1.length() > base_vec0.length() {
                                node_pos.clear();
                                node_pos.push(baseline_nodes[1].get_com_f32());
                                node_pos.push(baseline_nodes[0].get_com_f32());
                                top_left = &baseline_nodes[1];
                                top_left_id = 1;
                            } else {
                                node_pos.clear();
                                node_pos.push(baseline_nodes[0].get_com_f32());
                                node_pos.push(baseline_nodes[1].get_com_f32());
                                top_left = &baseline_nodes[0];
                                top_left_id = 0;
                            }
                        }
                    }
                }

                let mut max_angle = 0.0f32;

                let top_left_vec: Vector2D<_> = top_left.get_com_f32().into();
                let top_vec = top_left_vec - top_right.get_com_f32().into();

                for (_, i) in nodes.iter().enumerate() {
                    if let TagComponents::Normal { node, .. } = i {
                        let node_vec: Vector2D<_> = node.get_com_f32().into();
                        let angle = top_vec.angle_to(top_left_vec - node_vec).radians.abs();
                        if angle > max_angle {
                            max_angle = angle;
                        }
                    }
                }

                let mut distance = Vec::new();

                for (idx, i) in nodes.iter().enumerate() {
                    if let TagComponents::Normal { node, .. } = i {
                        let node_vec: Vector2D<_> = node.get_com_f32().into();
                        let angle = top_vec.angle_to(top_left_vec - node_vec).radians.abs();

                        let angle_diff = max_angle - angle;

                        if angle_diff < 0.1 {
                            distance.push(((top_left_vec - node_vec).length(), idx));
                        }
                    }
                }

                distance.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                if distance.len() != class.get_height() - 1 {
                    println!("wrong number of nodes in column");
                    return None;
                }

                let mut column = Vec::new();
                column.push(top_left);
                for (_, idx) in &distance {
                    if let TagComponents::Normal { node, .. } = &nodes[*idx] {
                        column.push(node);
                    }
                }

                if column.len() != class.get_height() {
                    println!("wrong height");
                    return None
                }

                let mut data = Vec::new();

                for row_start in column {
                    let mut row = Vec::new();
                    for (idx, i) in nodes.iter().enumerate() {
                        if let TagComponents::Normal { node, .. } = i {
                            let node_vec: Vector2D<_> = node.get_com_f32().into();
                            let row_start_vec: Vector2D<_> = row_start.get_com_f32().into();
                            let row_vec: Vector2D<_> = row_start_vec - node_vec;

                            if row_vec.length() < 0.1
                                || top_vec.angle_to(row_vec).radians.abs() < 0.1
                            {
                                row.push((row_vec.length(), idx));
                            }
                        }
                    }

                    row.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                    //println!("row: {:?}", row);

                    if row.len() != class.get_width() && row.len() + 2 != class.get_width() {
                        println!("wrong row_width: {}", row.len());
                        return None;
                    }

                    for (_, i) in row {
                        if let TagComponents::Normal { node, data: point } = &nodes[i] {
                            node_pos.push(node.get_com_f32());
                            // println!("adding {}th node", i);
                            data.push(*point);
                        } else {
                            return None;
                        }
                    }
                }

                // println!("node_pos.len(): {}", node_pos.len());

                let mut data_int = 0;
                for i in data {
                    data_int <<= 1;
                    if i {
                        data_int |= 1;
                    }
                }

                if node_pos.len() != class.total_node_count() + 1 {
                    return None;
                }

                Some(DecodedTag {
                    data: data_int,
                    top_right: top_right_id,
                    nodes: nodes.clone(),
                    top_left: top_left_id,
                    bottom_left: distance.last().unwrap().1,
                    node_pos,
                    class: class.clone()
                })
            } else {
                unreachable!();
            }
        } else {
            unreachable!();
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TagClass {
    Topotag3x3,
    Topotag4x4,
    Topotag5x5,
}

impl TagClass {
    fn normal_node_count(&self) -> usize {
        match self {
            TagClass::Topotag3x3 => 7,
            TagClass::Topotag4x4 => 14,
            TagClass::Topotag5x5 => 23,
        }
    }

    fn total_node_count(&self) -> usize {
        self.normal_node_count() + 1
    }

    fn get_height(&self) -> usize {
        match self {
            TagClass::Topotag3x3 => 3,
            TagClass::Topotag4x4 => 4,
            TagClass::Topotag5x5 => 5,
        }
    }

    fn get_width(&self) -> usize {
        match self {
            TagClass::Topotag3x3 => 3,
            TagClass::Topotag4x4 => 4,
            TagClass::Topotag5x5 => 5,
        }
    }

    pub fn get_expected_node_pos(&self) -> Vec<(f32, f32)> {
        let mut out = Vec::new();
        for i in 0..self.get_height() {
            for j in 0..self.get_width() {
                if i == 0 && j == 1 {
                    out.push((1.5, 1.0))
                } else {
                    out.push((j as f32 + 1.0, i as f32 + 1.0))
                }
            }
        }
        out
    }
}

pub fn detect_tag(
    topo: &Vec<FeatureVector>,
    start: u32,
    output: &mut HashMap<[(u32, u32); 2], TagComponents>,
    class: &Vec<TagClass>,
) -> TagComponents {
    let mut child = topo[start as usize].child;
    let current = &topo[start as usize];
    let mut children = Vec::new();
    let mut all_children = Vec::new();

    while let Some(c) = child {
        child = topo[c as usize].sibling;
        if topo[c as usize].area * 30 > current.area {
            children.push(c)
        }
        all_children.push(c)
    }

    if children.len() == 2 && !current.color {
        // possible baseline
        let mut has_grandchildren = false;
        for &c in &children {
            if topo[c as usize].child.is_some() {
                has_grandchildren = true;
            }
        }

        let nodes = [
            topo[children[0] as usize].clone(),
            topo[children[1] as usize].clone(),
        ];
        if !has_grandchildren {
            return TagComponents::Baseline { nodes };
        }
    }

    if children.len() < 2 && !current.color {
        // possible normal node
        if children.len() == 0 {
            return TagComponents::Normal {
                data: false,
                node: current.clone(),
            };
        }

        if children.len() == 1 {
            let mut has_grandchildren = false;
            for &c in &children {
                if topo[c as usize].child.is_some() {
                    has_grandchildren = true;
                }
            }
            if !has_grandchildren {
                return TagComponents::Normal {
                    data: true,
                    node: current.clone(),
                };
            }
        }
    }

    for tag_class in class {
        if children.len() == tag_class.total_node_count() && current.color {
            let mut normal_nodes = Vec::new();
            let mut baseline_nodes = Vec::new();
            for &c in &children {
                match detect_tag(topo, c, output, &class) {
                    normal @ TagComponents::Normal { .. } => {
                        normal_nodes.push(normal.clone());
                    }
                    baseline @ TagComponents::Baseline { .. } => {
                        baseline_nodes.push(baseline.clone());
                    }
                    _ => (),
                }
            }

            if normal_nodes.len() == tag_class.normal_node_count() && baseline_nodes.len() == 1 {
                let mut nodes = Vec::new();
                nodes.push(baseline_nodes[0].clone());
                nodes.append(&mut normal_nodes);

                let root = TagComponents::Root {
                    nodes,
                    class: (*tag_class).clone(),
                };
                output.insert(current.bounding_box, root.clone());
                return root;
            }
        }
    }

    for &c in &all_children {
        detect_tag(topo, c, output, &class);
    }

    return TagComponents::None;
}
