use crate::fft_solver::*;
use crate::util::*;
use std::io::prelude::*;
use std::ops::Range;

#[derive(Debug)]
pub struct PeriodicSolveNode<const GRID_DIMENSION: usize> {
    pub input_aabb: AABB<GRID_DIMENSION>,
    pub output_aabb: AABB<GRID_DIMENSION>,
    pub convolution_id: OpId,
    pub steps: usize,

    /// calculate remaining output dections based on either
    ///  - AABB::decomposition for central solve
    ///  - APFrustrum::decomposition for frustrum solves
    pub remainder: Range<NodeId>,

    /// Should we swap input / output and run this?
    pub time_cut: Option<NodeId>,
}

#[derive(Debug)]
pub struct DirectSolveNode<const GRID_DIMENSION: usize> {
    pub input_aabb: AABB<GRID_DIMENSION>,
    pub output_aabb: AABB<GRID_DIMENSION>,
    pub steps: usize,
}

/// Used for central periodic solve, can't appear in frustrums
#[derive(Debug)]
pub struct RepeatNode {
    pub n: usize,
    pub node: NodeId,
    pub next: Option<NodeId>,
}

#[derive(Debug)]
pub enum PlanNode<const GRID_DIMENSION: usize> {
    PeriodicSolve(PeriodicSolveNode<GRID_DIMENSION>),
    DirectSolve(DirectSolveNode<GRID_DIMENSION>),
    Repeat(RepeatNode),
}

pub struct APPlan<const GRID_DIMENSION: usize> {
    pub nodes: Vec<PlanNode<GRID_DIMENSION>>,
    pub root: NodeId,
}

impl<const GRID_DIMENSION: usize> APPlan<GRID_DIMENSION> {
    pub fn get_node(&self, node: NodeId) -> &PlanNode<GRID_DIMENSION> {
        &self.nodes[node]
    }

    pub fn to_dot_file<P: AsRef<std::path::Path>>(&self, path: &P) {
        let mut writer =
            std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        writeln!(writer, "digraph plan {{").unwrap();
        //writeln!(writer, "  ratio=1.3;").unwrap();

        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                PlanNode::PeriodicSolve(_) => {
                    writeln!(writer, " n_{} [label=\"periodic\"];", i).unwrap();
                }
                PlanNode::DirectSolve(_) => {
                    writeln!(writer, " n_{} [label=\"direct\"];", i).unwrap();
                }
                PlanNode::Repeat(_) => {
                    writeln!(writer, " n_{} [label=\"repeat\"];", i).unwrap();
                }
            }
        }

        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                PlanNode::PeriodicSolve(p) => {
                    for r in p.remainder.clone() {
                        writeln!(writer, " n_{} -> n_{} [color=blue];", i, r)
                            .unwrap();
                    }
                    if let Some(r) = p.time_cut {
                        writeln!(writer, " n_{} -> n_{} [color=red];", i, r)
                            .unwrap();
                    }
                }
                PlanNode::DirectSolve(_) => {}
                PlanNode::Repeat(r) => {
                    writeln!(writer, " n_{} -> n_{} [color=green];", i, r.node)
                        .unwrap();
                    if let Some(r2) = r.next {
                        writeln!(writer, " n_{} -> n_{} [color=black];", i, r2)
                            .unwrap();
                    }
                }
            }
        }

        writeln!(writer, "}}").unwrap();
    }
}
