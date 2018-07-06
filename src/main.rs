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
use std::collections::HashSet;
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

#[derive(Debug)]
pub struct Action {
    instance_names: Vec<String>,
    name: String,
}

#[derive(Debug)]
pub struct State {
    instance_names: Vec<String>,
    name: String,
}

// i'm super interested in seeing what the data structure for holding Actions, States, and Entities will look like!
#[derive(Debug)]
pub struct Entity {
    instance_names: Vec<String>,
    name: String,
}

pub struct PhysicalEntity {
    instance_name: String,
    name: String,
    shape: GeometricShape,
    scale: u64, // log scale where 0 == subatomic
}

pub struct EventConjugate {
    actions: Vec<Action>,
    states: Vec<State>,
    entities: Vec<Entity>,
}


pub fn match_events(ev1: &Event, ev2: &Event) -> bool {
    true // todo. attempts to match two event conjugates.
}

//pub fn fuzzy_match_events(c1: &EventConjugate, c2: &EventConjugate) -> f64 {
//    true // todo. yields similarity of events.
//}
//
//pub fn compare_event_similarity(c1: &EventConjugate, c2: &EventConjugate, c3: &EventConjugate) -> f64 {
//    true // todo. compares similarity of c1 to c2 and c3 and yields the ratio of greater similarity of c1 to c2 than c3, [0,1] (0 -> c1 and c3 are very similar, c1 and c2 are totally dissimilar).
//}
//
pub fn contains_conjugate(inst: &Instance, conj: &EventConjugate) -> bool {
    true // todo. attempts to match event conjugate to the instance's before event conjugate.
}

//
//pub fn conjugate_similarity(inst: &Instance, conj: &EventConjugate) -> bool {
//    true // todo. attempts to match event conjugate to the instance's before event conjugate.
//}
//
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
    before: EventConjugate,
    outcome: (f64, EventConjugate),
    // confidence and outcome
    known_exceptions: Vec<Event>,
}

pub trait Effect {
    fn effect(&self) -> Vec<&(f64, EventConjugate)>;
}

pub trait Prob {
    fn freq(&self) -> f64;
    fn prob(self, inst: &Instance) -> f64;
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

    fn prob(self, inst: &Instance) -> f64 {
        let mut min: f64 = self.outcome.0;

        for ev in self.known_exceptions {
            let ev_before = &ev.before.unwrap();
            if contains_conjugate(inst, &ev_before) {
                return 0.; // causal outcome may still occur, but etiology is unknown. rely on inference.
            }
            let cur = compare_conjugate_similarity(inst, &self.before, ev_before);
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
}

pub struct Event {
    before: Option<EventConjugate>,
    after: Option<EventConjugate>,
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

pub struct Format {
    name: String,
    processed_from: Vec<InstanceData>, // types that can be processed into this format
}


//enum ProofResponse {
//    B(bool),
//    U(u64),
//    I(i64),
//    F(f64),
//    S(String),
//}

pub struct Proof {
    format: Format,
    // matches name field of Format this works on
    times_used: u32,
    // should increment until it maxes out, then stay constant
    avg_runtime: Duration,
    //averaging uses times_used
    tier: u8,
    // smaller value -> more consistent
    conditions: Vec<Event>, // all the actions, states, entities, and relations necessary for the proof.
}

impl Proof {
    fn prove(self, inst: Instance) -> bool {
        let mut matched = false;
        for sev in self.conditions.iter() {
            matched = false;
            for iev in inst.events.iter() {
                if match_events(iev, sev) {
                    matched = true;
                }
            }
            if !matched {
                return false;
            }
        }
        true
    }
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
    //format.name is the string key
    proofs: HashMap<String, Vec<Proof>>,
    id: u64,
    // should equal ID of AssertionContainer
    container_name: String,
    // duration since epoch of last_diagnostic
    last_diagnostic: Duration,
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
    containers: HashMap<String, AssertionContainer>, // this might become a b tree based on semantic overlap instead.
}

fn generate_assertions(inst: &Instance) -> Vec<Assertion> {
    panic!("Not implemented") // todo
}

fn generate_diagnostics(inst: &Vec<Assertion>) -> Vec<AssertionDiagnostic> {
    panic!("Not implemented") // todo
}

#[derive(Debug)]
enum RelationParams {
    Entity(Entity),
    Action(Action),
    State(State),
}

#[derive(Debug)]
struct Relation {
    params: Vec<RelationParams>,
}

/// These are what each line could represent.
enum MinParseItem {
    A(Assertion),
    C(CausalRule),
    E(Event),
}

fn string_min_parse(s: &str, e: &mut HashMap<String, Vec<String>>, r: &mut HashMap<String, (String, String)>) -> Option<MinParseItem> {
    let debug = true;
    if debug {
        println!("Parsing string: {}", s);
    }
    // todo current problem: how do we get from relations to A / C / E ?

    // trivial implementation: relation exists if MRT file suggests it does, with probability based on
    // trust of the file.

    lazy_static! {
        pub static ref RELATION : Regex = Regex::new("^[[a-zA-Z0-9_]]*[(]").unwrap();
        pub static ref STARTPAREN : Regex = Regex::new("[(]").unwrap();
        pub static ref ENDPAREN : Regex = Regex::new("[)]").unwrap();

        pub static ref OPERATORS : [String; 4] = ["-".to_string(),
        "+".to_string(), "->".to_string(), ":".to_string()];

        pub static ref CORE_PHRASES: [String; 3] = ["action".to_string(),
        "state".to_string(), "entity".to_string()];
    }

    let primitive = |s: &str| -> bool {
        match s {
            "action" => true,
            "state" => true,
            "entity" => true,
            _ => false
        }
    };

    let prim_rel = |relstr: &str, parstr: &str| -> Relation {
        match relstr {
            "action" => {
                panic!("Not implemented")
            }
            "state" => {
                panic!("Not implemented")
            }
            "entity" => {
                panic!("Not implemented")
            }
            _ => panic!("Non-primary assertion passed to prim_rel. Unable to parse.")
        }
    };

    fn fill_in_variables(params: &str, unpacked: &str) -> Vec<Relation> {
        let mut relations = Vec::new();
        // uh..
        relations
    };

    let mut parse_relations = |s: &str| -> Vec<Relation> { panic!("This should be overwritten.") };

    let mut parse_relations = |s: &str| -> Vec<Relation> {
        let parse_relation = |relstr: &str, parstr: &str| -> Vec<Relation> {
            fn spl<'a>(s: &'a str) -> Vec<&'a str> {
                let mut vec: Vec<&str> = Vec::new();
                for val in s.split(",") {
                    vec.push(val.trim());
                }
                vec
            }

            let &(ref original, ref unpacked) = &r[relstr];

            let mut parmap = HashMap::new();
            let keys = spl(&original);
            let vals = spl(parstr);

            assert_eq!(keys.len(), vals.len());

            for i in 0..keys.len() {
                parmap.insert(keys[i], vals[i]);
            }

            //println!("{:?}", &e[relstr]);

            parse_relations(&unpacked) // recurse until the relations have all been simplified.
        };

        if debug {
            println!("in parse_relations");
        }
        let mut vec = Vec::new();
        for m in RELATION.find_iter(s) {
            if debug {
                println!("iterating");
            }
            // for each m we know that we have chars, a start paren, chars, and an end paren.
            let (relation, params) = {
                let fs = m.start();
                let sp = fs + STARTPAREN.find(&s[fs..]).unwrap().start(); // start parenthesis
                if debug {
                    println!("relation start index: {}", fs);
                    println!("relation end index: {}", sp);
                }
                let relation = &s[fs..sp]; //relation
                if debug {
                    println!("relation: {}", relation);
                }
                let params = &s[sp + 1..ENDPAREN.find(&s[sp + 1..]).unwrap().start() + sp + 1]; //relation values
                (relation, params)
            };
            if primitive(relation) {
                vec.push(prim_rel(relation, params));
            } else {
                vec.extend(parse_relation(relation, params));
            }
        }
        vec
    };

    let parse_assertion = |s: &str, e: &mut HashMap<String, Vec<String>>| -> Assertion {
        println!("in parse_assertion");
        let mut before = EventConjugate {
            actions: Vec::new(),
            states: Vec::new(),
            entities: Vec::new(),
        };

        let mut after = EventConjugate {
            actions: Vec::new(),
            states: Vec::new(),
            entities: Vec::new(),
        };

        panic!("Not implemented")
    };

    fn parse_event(s: &str, e: &mut HashMap<String, Vec<String>>) -> Event {
        panic!("Not implemented")
    }

    enum Ce {
        A,
        // assertion
        E,
        // entity
        Bad,
    }

    let c = |str: &str| -> Ce {
        let n = |v: std::option::Option<usize>| -> bool {
            match v {
                Some(_) => true,
                None => false
            }
        };

        if n(OPERATORS[2].find(str)) && !n(OPERATORS[3].find(str)) {
            Ce::E
        } else if !n(OPERATORS[2].find(str)) && n(OPERATORS[3].find(str)) {
            Ce::A
        } else {
            Ce::Bad
        }
    };

    println!("{:?}", parse_relations(s));

    match c(&s) {
        Ce::A => Some(MinParseItem::A(parse_assertion(s, e))),
        Ce::E => Some(MinParseItem::E(parse_event(s, e))),
        Ce::Bad => None
    }
}

/// Usually kid will be constantly predicting a whole bunch of things and processing those
/// predictions in a lot of ways (e.g. to self-optimize and to decide how to act). But, for the
/// minimal case, we're just looking at a simple prediction
fn minimal_predict_string(before: String, am: AssertionMaster) -> String {
    panic!("Not implemented"); // todo
}

fn parse_minimal(fname: &Path, name: String) -> Instance {
    fn read_lines(fname: &Path) -> Vec<String> {
        fn remove_comments(s: &str) -> Option<&str> {
            match s.find("//") {
                Some(index) => return remove_comments(&s[..index]),
                None => {
                    if !s.contains("->") && !s.contains(":") {
                        return None; // cleans out empty lines. Also removes misformatted lines.
                    }
                    return Some(s);
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

    let mut recursive_relation_defs: HashMap<String, (String, String)> = HashMap::new();

    let mut process_lines = |stringvec: Vec<String>| -> (Vec<Assertion>, Vec<CausalRule>, Vec<Event>) {
        let mut va = Vec::new();
        let mut vc = Vec::new();
        let mut ve = Vec::new();
        let mut ents: HashMap<String, Vec<String>> = HashMap::new();
        for astr in stringvec {
            let m = string_min_parse(&astr, &mut ents, &mut recursive_relation_defs);
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
    };

    fn get_semantics(events: &Vec<Event>) -> Option<SemanticEmbedding> {
        println!("Function get_semantics is not implemented. Yielding None for now."); // todo
        None
    }

    let (mut va, mut vc, mut ve) = process_lines(read_lines(fname));

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
        semantics: get_semantics(&ve),
        events: ve,
        assertions: match va.len() {
            0 => None,
            _ => Some(va)
        },
        causal_rules: match vc.len() {
            0 => None,
            _ => Some(vc)
        },
        processed_data: HashMap::new(),
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
    // diagnostics will be essential for optimizing assertion calculation.

    let mut inst_vec = vec![min_inst];
    // I'm still deciding how to deal with semantics and instance organization for when we have
    // A Lot of instances.
    // important considerations: accessibility based on entities, actions, semantic content, and
    // state. I haven't decided on the best data structure for this yet.

    // Maybe a series of trees for each search method with the instance index in a giant vector out
    // in the heap? It's okay if recalling specific instances (aka episodic memory) is slower than
    // the assertion stuff; that's fine and normal in humans.

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
    // we now have all of our assertions in a single container. put it in the master and we're good!

    am.containers.insert(core_ac.name.clone(), core_ac);

    //next up we want to predict something.
    let ptext = "state(match, burning) + state(newspaper, wet) + symmetric_action(newspaper, match, touching)".to_string();
    println!("{}", minimal_predict_string(ptext, am));
}
