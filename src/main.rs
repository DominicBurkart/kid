// mod argument_types

extern crate ndarray;
extern crate bytes;
extern crate geo;

use bytes::Bytes;
use geo::{Bbox, Coordinate, Point, Polygon};
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
    points: Vec<[f64; 300]>
}

pub fn vecdist(v1: &mut [f64], v2: &mut [f64]) {

}

pub fn cosdist(v1: &mut [f64], v2: &mut [f64]) {

    if v1.len() != v2.len() { panic!("Arrays of two lengths passed to cosdist")}
    for i in (1..v1.len()){

    }
    1 - similarity // cosine distance from similarity
}

pub struct Location {
    // a collection of places within one linguistic data source (e.g., "in the parlor" or "out back")
    domain: Vec<String>,

    // n-dimensional point
    spatial: Option<Vec<f64>>,

    // geocoordinate
    geo: Option<Point<f64>>,

    // linguistic embedding
    ling: Option<SemanticEmbedding>,
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
    name: String,
    kind: String,
}

pub enum InstanceData {
    String(InstMetaData),
    Path(InstMetaData),
    Bytes(InstMetaData),
    Array2(InstMetaData),
}

pub struct ProcessedData {
    name: String,
    format_name: String
}

pub struct Action {
    // todo
}

pub struct State {
    // todo
}

pub struct Entity {
    // todo
}

pub struct PhysicalEntity {
    shape: GeometricShape,
    scale: u64, // log scale where 0 == subatomic
}

pub struct EventConjugate {
    actions: Vec<Action>,
    states: Vec<State>,
    entities: Vec<Entity>,
}

pub fn contains_conjugate(inst: &Instance, conj: &EventConjugate) -> bool {
    true // todo. attempts to match event conjugate to the instance.
}

pub fn compare_conjugate_similarity(inst: &Instance, conj1: &EventConjugate, conj2: &EventConjugate) -> f64 {
    1. // todo. attempts to fuzzy match event conjugate to the instance. output is relative similarity to conj1 vers conj2 [0,1], bigger values => more similar
    // needs to pull apart similarities in the two conjugates
}

/// Causal rule. Since we can never know if we are not detecting some entity, we must treat even
/// our causal rules probabilistically – we can never prove sufficiency, just necessity.
///
/// Thus, our causal rules retain information on when these rules are violated (for example,
/// paper will light on fire, unless the paper is wet. The additional state of wetness precludes the
/// general causal rule that paper will burn when exposed to flame). These exceptions are generalized
/// by finding how similar a given scene is to the general rule versus the exception.
pub struct CausalRule {
    name: String,
    prior: EventConjugate,
    outcome: (f64, EventConjugate),
    // confidence and outcome
    known_exceptions: Vec<EventConjugate>,
}

pub trait Effect {
    fn effect(&self) -> Vec<&(f64, EventConjugate)>;
    fn effect_given(&self, inst: &Instance) -> Vec<(f64, &EventConjugate)>;
}

pub trait Prob {
    fn freq(&self) -> f64;
    fn prob(&self, inst: &Instance) -> f64;
}

pub struct GeometricShape {
    dimensions: u32,
    points: Vec<Vec<f64>>,
    orientation: Vec<f64>,
}

/// Probability that the causal rule applies in a given instance.
impl Prob for CausalRule {
    fn freq(&self) -> f64 {
        self.outcome.0
    }
    fn prob(&self, inst: &Instance) -> f64 {
        let mut min: f64 = self.outcome.0;
        for ev in &self.known_exceptions {
            if contains_conjugate(inst, &ev) {
                return 0.; // causal outcome may still occur, but etiology is unknown. rely on inference.
            }
            let cur = compare_conjugate_similarity(inst, &self.prior, ev);
            if cur < min {
                min = cur;
            }
        }
        min
    }
}

/// Effect yields Vectors of possible outcomes to deal with inferences; for causal rules, there is
/// only one potential outcome (as well as the probability that it
///
impl Effect for CausalRule {
    fn effect(&self) -> Vec<&(f64, EventConjugate)> {
        let mut v = Vec::new();
        v.push(&self.outcome);
        v
    }
    fn effect_given(&self, inst: &Instance) -> Vec<(f64, &EventConjugate)> {
        let mut v = Vec::new();
        v.push((self.prob(inst), &self.outcome.1));
        v
    }
}

pub struct Event {
    prior: EventConjugate,
    posterior: EventConjugate,
}

pub struct Instance {
    name: String,
    // observed: duration since epoch at time observed (first int: seconds, second int: nanoseconds after seconds).
    observed: Duration,
    // occurred: duration since epoch at time occurred (first int: seconds, second int: nanoseconds after seconds).
    occurred: Option<Duration>,
    occurred_accuracy: u8,
    data: InstanceData,
    events: Vec<Event>,
    semantics: SemanticEmbedding,
    processed_data: HashMap<String, Vec<ProcessedData>>, // String is format name
}

impl Instance {}

pub struct Format {
    name: String,
    processed_from: Vec<InstanceData>, // types that can be processed into this format
}

pub struct Proof {
    format: Format,
    // matches name field of Format this works on
    times_used: u32,
    // should increment until it maxes out, then stay constant
    avg_runtime: Duration,
    //averaging uses times_used
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

/// Selects based on tier and then average performance time.
fn select_proof<'a>(proofs: &'a HashMap<String, Vec<Proof>>, instance: &Instance) -> Option<&'a Proof> {
    strongest_proof(proofs, instance.processed_data.keys())
}

pub struct Assertion {
    proofs: HashMap<String, Vec<Proof>>,
    //format.name is the string key
    id: u64,
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

pub struct AssertionMaster {
    containers: HashMap<String, AssertionContainer>,
}

fn main() {
    for number in (1..4).rev() {
        println!("{}!", number);
    }
    println!("LIFTOFF!!!");
    let x = [1.,2.,3.];
    let y = [1.5, 2.5, 3.5];
    let z = x - y;
    for n in z.iter() {
        println!("{}", n);
    }
}