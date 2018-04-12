extern crate ndarray;
extern crate bytes;

use bytes::Bytes;
use ndarray::prelude::Array2;
use std::collections::hash_map::Keys;
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// constant functions not yet supported: const fn wordvec_length() -> usize { 300 }

pub fn cur() -> Duration {
    SystemTime::now().duration_since(UNIX_EPOCH).expect("SystemTime::duration_since failed")
}

pub struct SemanticShape {
    // goal: determine a polygon in 300-dimensional semantic space.
    points: Vec<[f64; 300]>
}

pub struct SemanticEmbedding {
    // glove
    glove_vec: [f64; 300],
    twelve_closest: [(String, f64); 12], // word, distance

    // wordnet
    supers: Vec<String>,
    subs: Vec<String>,
    synonyms: Vec<String>,
    antonyms: Vec<String>,
    meaning: String,
}

pub struct InstMetaData {
    name: String
}

pub enum InstanceData {
    String(InstMetaData),
    Path(InstMetaData),
    Bytes(InstMetaData),
}

pub struct ProcessedData {}

pub struct Instance {
    name: String,
    // observed: duration since epoch at time observed (first int: seconds, second int: nanoseconds after seconds).
    observed: (u64, u32),
    // occurred: duration since epoch at time occurred (first int: seconds, second int: nanoseconds after seconds).
    occurred: (u64, u32),
    // occurred_accuracy: 0 when occurred is unknown. 1 is seconds, 2 is tenths of seconds, etc.
    occured_accuracy: u8,
    data: InstanceData,
    semantics: SemanticEmbedding,
    processed_data: HashMap<String, Vec<ProcessedData>>, // String is format name
}

impl Instance {}

pub struct Format {
    name: String,
    processed_from: Vec<InstanceData>, // types that can be processed into this format
}

pub struct Proof {
    format: String,
    // matches name field of Format this works on
    times_used: u32,
    // should increment until it maxes out, then stay constant
    avg_runtime: Duration,
    tier: u8, // smaller value -> more consistent
    // todo add executable or function as datafield for running proof. call it "apply"
    // apply: Fn
}


fn strongest_proof<'a>(map: &'a HashMap<String, Vec<Proof>>, keys: Keys<String, Vec<ProcessedData>>) -> Option<&'a Proof> {
    let mut choice: Option<&'a Proof> = None;
    for k in keys {
        if map.contains_key(k) && map[k].len() > 0 {
            match choice {
                None => choice = Some(&map[k][0]),
                _ => (),
            }
            let c = choice.unwrap();
            for p in &map[k] {
                if p.tier < c.tier || (p.tier == c.tier && p.avg_runtime < c.avg_runtime) {
                    choice = Some(p);
                }
            }
        }
    }
    choice
}

fn select_proof<'a>(proofs: &'a HashMap<String, Vec<Proof>>, instance: &Instance) -> Option<&'a Proof> {
    /**
    Selects based on tier and then average performance time.
    **/
    let available_formats = instance.processed_data.keys();
    strongest_proof(proofs, available_formats)
}


pub struct Assertion {
    proofs: HashMap<String, Vec<Proof>>,
    //format.name is the string key
    assertion_id: u64,
    container_name: String, // should equal ID of AssertionContainer
}

impl Assertion {
    fn prove(self, instance: Instance) {
        let poption = select_proof(&self.proofs, &instance);
        match poption {
            Some(_proof) => println!("Neat!"), // todo based on four potential assertion outcomes
            None => println!("ok!"),
        }
        // todo
    }
}

pub struct AssertionDiagnostic {}

impl AssertionDiagnostic {
    // todo: cluster analysis on correlation matrix
    // todo: outlier analysis on proof outputs
}

pub struct AssertionContainer {
    name: String,
    semantic_shape: SemanticShape,
    // space of contained assertions
    assertions: Vec<Assertion>,
    diagnostics: Vec<AssertionDiagnostic>,
}
