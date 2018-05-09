/// notable data structures:
/// Assertions are arguments with boolean or numeric output (e.g., x is a subset of y, or count the number of sheep in this image).
/// Causal rules suggest transformations in a system's entities, their actions, or their states.
/// Events contain information on how, in a specific instance, a system's entities, their actions, or their states changed.
/// Instances are specific points of data (e.g., a video of a person dancing).
#[macro_use(array)]
extern crate ndarray;
extern crate bytes;
extern crate geo;
extern crate regex;
#[macro_use]
extern crate lazy_static;

use bytes::Bytes;
use geo::{Bbox, Coordinate, Point, Polygon};
use ndarray::prelude::{Array1, Array2};
use regex::Regex;
use std::collections::hash_map::Keys;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};


pub fn cur() -> Duration {
    SystemTime::now().duration_since(UNIX_EPOCH).expect("SystemTime::duration_since failed")
}

pub struct SemanticShape {
    points: Vec<[f64; 300]>
}

fn shape_from_instances(instances: &Vec<Instance>) -> SemanticShape {
    panic!("Not implemented") // todo
}

/// euclidean distance between two points of arbitrary dimensions.
pub fn euc_dist(v1: &Array1<f64>, v2: &Array1<f64>) -> f64 {
    if v1.len() != v2.len() { panic!("Arrays of two lengths passed to vecdist") }
    ((v1 - v2) * (v1 - v2)).scalar_sum().sqrt()
}

/// Cosine distance operation on two arrays. Returns NAN if one or more input arrays is all-zero.
pub fn cos_dist(v1: &Array1<f64>, v2: &Array1<f64>) -> f64 {
    if v1.len() != v2.len() { panic!("Arrays of two lengths passed to cosdist") }
    1. - (v1.dot(v2) / ((v1 * v1).scalar_sum().sqrt() * (v2 * v2).scalar_sum().sqrt()))
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
    // silly way of saying I have no idea how to store instancedata. maybe just as files?
    Str(String, InstMetaData),
    Pat(PathBuf, InstMetaData),
    Byt(Bytes, InstMetaData),
    Ar2(Array2<f64>, InstMetaData),
}

pub struct ProcessedData {
    name: String,
    format_name: String,
}

/// Actions must match on name and id to be considered identical (IDs themselves aren't unique).
pub struct Action {
    id: u64,
    name: String,
}

/// States must match on name and id to be considered identical (IDs themselves aren't unique).
pub struct State {
    id: u64,
    name: String,
}

/// Entities must match on name and id to be considered identical (IDs themselves aren't unique).
pub struct Entity {
    id: u64,
    name: String,
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
    known_exceptions: Vec<Event>,
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

/// Probability that the causal rule applies in a given instance. Does not factor in higher-order
/// similarity modeling (just a series of binary similarity comparisons).
///
/// Ideally, clusters of similar exceptions should be formed and this should compare new events to
/// these clusters instead of to individual exceptions (i.e., exceptions should be abstracted away
/// from specific instances as more of them are observed for computational ease).
impl Prob for CausalRule {
    fn freq(&self) -> f64 {
        self.outcome.0
    }
    fn prob(&self, inst: &Instance) -> f64 {
        let mut min: f64 = self.outcome.0;
        for ev in &self.known_exceptions {
            if contains_conjugate(inst, &ev.prior) {
                return 0.; // causal outcome may still occur, but etiology is unknown. rely on inference.
            }
            let cur = compare_conjugate_similarity(inst, &self.prior, &ev.prior);
            if cur < min {
                min = cur;
            }
        }
        min
    }
}

/// Effect yields Vectors of possible outcomes to deal with inferences; for causal rules, there is
/// only one potential outcome (as well as the probability that the causal rule applies)
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
    posterior: Option<EventConjugate>,
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
    causal_rules: Option<Vec<CausalRule>>,
    // some input gives us explicit causal rules. this is second+ order knowledge.
    assertions: Option<Vec<Assertion>>,
    // some input gives us explicit assertions. this is second+ order knowledge.
    semantics: Option<SemanticEmbedding>,
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

enum Response {
    B(bool),
    U(u64),
    I(i64),
    F(f64),
    S(String),
}

trait Provable {
    fn prove(inst: Instance) -> (Response, f64); // proof output
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
    container_name: String,
    // should equal ID of AssertionContainer
    last_diagnostic: Duration,
    // since epoch
    updated_since: bool, // updated since last diagnostic
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
    semantic_shape: Option<SemanticShape>,
    // space of contained assertions
    assertions: Vec<Assertion>,
    diagnostics: Vec<AssertionDiagnostic>,
}

pub struct AssertionMaster {
    containers: HashMap<String, AssertionContainer>,
}

fn generate_assertions(inst: &Instance) -> Vec<Assertion> {
    panic!("Not implemented") // todo
}

fn generate_diagnostics(inst: &Vec<Assertion>) -> Vec<AssertionDiagnostic> {
    panic!("Not implemented") // todo
}

/// These are what each line could represent.
enum MinParseItem {
    A(Assertion),
    C(CausalRule),
    E(Event),
}

fn string_min_parse(s: String) -> Option<MinParseItem> {

    // todo current problem: how do we get from relations to A / C / E ?

    // trivial assertion: relation exists if MRT file suggests it does, with probability based on
    // trust of the file.

    fn parse_relation(relstr: &str, parstr: &str) -> Relation {
        panic!("Not implemented")
        // todo: split params into unique values. How do we deal with multiple analogous relations?
    }

    fn parse_relations(s: &str) -> Vec<Relation> { // todo this should be returning a vec of relation objects, right?
        let mut vec = Vec::new();
        for m in RELATION.find_iter(s) {
            // for each m we know that we have chars, a start parent, chars, and an end paren.
            let fs = m.start();
            let sp = fs + STARTPAREN.find(s[fs..]).unwrap().start(); // start parenthesis
            let relation = s[fs..sp]; //relation
            let params = s[sp + 1..ENDPAREN.find(s[sp + 1..]).unwrap().start()]; //relation values
            vec.push(parse_relation(relation, params));
        }
        vec
    }

    fn parse_assertion(s: String) -> Assertion {
        let mut prior = EventConjugate {
            actions: Vec::new(),
            states: Vec::new(),
            entities: Vec::new(),
        };

        let mut posterior = EventConjugate {
            actions: Vec::new(),
            states: Vec::new(),
            entities: Vec::new(),
        };
    }

    lazy_static! {
            static ref RELATION : Regex = Regex::new("[[:alpha:]]*?[(][[:alpha:]]*?[)]").unwrap();
            static ref STARTPAREN : Regex = Regex::new("[(]").unwrap();
            static ref ENDPAREN : Regex = Regex::new("[)]").unwrap();

            static ref OPERATORS: [String; 4] = ["-".to_string(),
            "+".to_string(), "->".to_string(), ":".to_string()];

            static ref CORE_PHRASES: [String; 3] = ["action".to_string(),
            "state".to_string(), "entity".to_string()];
        }

    // todo split processing here based on if it's an event, rule, or assertion.


    panic!("Not implemented"); // todo
}

fn parse_minimal(fname: &Path, name: String) -> Instance {
    fn read_lines(fname: &Path) -> Vec<String> {
        fn remove_comments(s: &str) -> Option<&str> {
            match s.find("//") {
                Some(index) => return remove_comments(s[..index]),
                None => {
                    if !s.contains("->") && !s.contains(":") {
                        None // cleans out empty lines. Also removes misformatted lines.
                    }
                    Some(s)
                }
            }
        }

        let file = File::open(fname).unwrap(); //todo deal with potential file errors.
        let buf_reader = BufReader::new(file);
        let mut out = Vec::new();
        for l in buf_reader.lines() {
            let s = l.unwrap(); // todo deal with potential string errors.
            match remove_comments(&s) {
                Some(cleaned) => out.push(cleaned.to_string()),
                None => (),
            }
        }
        out
    }


    fn process_lines(stringvec: Vec<String>) -> (Vec<Assertion>, Vec<CausalRule>, Vec<Event>) {
        let mut va = Vec::new();
        let mut vc = Vec::new();
        let mut ve = Vec::new();
        for astr in stringvec {
            let m = string_min_parse(astr);
            match m {
                Some(MinParseItem::A(assertion)) => {
                    va.push(assertion);
                }
                Some(MinParseItem::C(rule)) => {
                    vc.push(rule);
                }
                Some(MinParseItem::E(event)) => {
                    ve.push(event);
                }
                None => (),
            }
        }
        (va, vc, ve)
    }

    fn get_semantics(events: &Vec<Event>) -> SemanticEmbedding {
        panic!("Not implemented") // todo
    }

    let (mut va, mut vc, mut ve) = process_lines(read_lines(fname));
    let mut sems = get_semantics(&ve);

    Instance {
        name,
        observed: cur(),
        occurred: None, // later versions should estimate time + provide accuracy estimate.
        occurred_accuracy: 0,
        data: InstanceData::Pat(fname.to_path_buf(),
                                InstMetaData {
                                    name: fname.to_str().unwrap().to_string(), // todo error handling
                                    kind: "File".to_string(),
                                }),
        events: ve,
        assertions: Some(va),
        causal_rules: Some(vc),
        processed_data: HashMap::new(),
        semantics: Some(sems),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    /// tests distance metrics (euclidean and cosine)
    fn test_dist() {
        use super::*;
        let a1: Array1<f64> = array![1., 0., 0.];
        let a2: Array1<f64> = array![1., 0., 1.];
        assert_eq!(1., euc_dist(&a1, &a2)); // minimal test

        let a3: Array1<f64> = array![-1., 0., 2.];
        let a4: Array1<f64> = array![-2., 0., 2.];
        assert_eq!(1., euc_dist(&a3, &a4)); // negative numbers

        let point_arrays = [
            a1, a2, a3, a4,
            array![-2435345., 123412423., -1999.] as Array1<f64>,
            array![0.01, 0.02, 0.03] as Array1<f64>, // decimals
            array![9999999999999., 9999999999999., -9999999999999.] as Array1<f64>, // larger nums
            array![0., 0., 0.] as Array1<f64>,
            array![5.,5.,6.] as Array1<f64>
        ];

        for v1 in point_arrays.iter() {
            for v2 in point_arrays.iter() {
                println!("v1: {:?}", v1);
                println!("v2: {:?}\n\n", v2);

                // check symmetric
                assert_eq!(euc_dist(v2, v1), euc_dist(v1, v2));
                assert!(cos_dist(v2, v1).is_nan() || (cos_dist(v2, v1) == cos_dist(v1, v2)));

                // distance from self should be zero
                assert_eq!(0., euc_dist(v1, v1));
                assert!(cos_dist(v1, v1).is_nan() || 0 == (10000000. * cos_dist(v1, v1)).round());
                // (account for rounding errors with cos_dist)

                // distance between any non-identical vectors should not be zero
                if v1 != v2 {
                    assert_ne!(0., euc_dist(v1, v2));
                    assert_ne!(0., cos_dist(v1, v2));
                }
            }
        }
    }
}

fn main() {
    println!("Minimal use case.");

    let min_txt_path = Path::new("src/minimal.txt");

    let min_inst = parse_minimal(min_txt_path, "minimal".to_string());

    let mut assertions = generate_assertions(&min_inst);
    let mut diagnostics = generate_diagnostics(&assertions);

    let mut inst_vec = vec![min_inst];
    // how we store instances is going to matter a lot.
    // important considerations: accessibility based on entities, actions, semantic content, and
    // state. I haven't decided on the best data structure for this yet.
    // Maybe each assertion_container can have pointers to the relevant instances which are
    // stored in a giant vector somewhere in the heap? It's okay if recalling specific instances
    // (aka episodic memory) is slower than the assertion stuff; that's also true in human minds.

    // generally we would want to check + rebalance all of our assertions (and how this is done
    // given new data will be central to the functioning of this algorithm), but for now let's
    // only look at the case of the first assertions from the first instance.

    let mut am = AssertionMaster {
        containers: HashMap::new()
    };
    let mut core_ac = AssertionContainer {
        name: "core".to_string(),
        assertions,
        diagnostics,
        semantic_shape: Some(shape_from_instances(&inst_vec)),
    };
    // we now have all of our assertions in a single container.

    //next up we want to predict something.
}