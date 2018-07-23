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
#[macro_use]
extern crate quick_error;
extern crate kodama;

use bytes::Bytes;
use geo::{Bbox, Coordinate, Point, Polygon};
use kodama::{linkage, Method, Dendrogram};
use ndarray::{ArrayBase, Array, Axis};
use ndarray::prelude::{Array1, Array2};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Keys;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};


pub fn cur() -> Duration {
    SystemTime::now().duration_since(UNIX_EPOCH).expect("SystemTime::duration_since failed")
}

#[derive(Clone)]
pub struct SemanticShape {
    points: Vec<[f64; 300]>
}

fn shape_from_instances(instances: &Vec<Instance>) -> SemanticShape {
    unimplemented!() // todo
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

#[derive(Debug, Clone)]
pub struct InstMetaData {
    name: String,
    kind: String,
}

#[derive(Debug, Clone)]
pub enum InstanceData {
    // silly way of saying I have no idea how to store instancedata. maybe just as files?
    Str(String, InstMetaData),
    Pat(PathBuf, InstMetaData),
    Byt(Bytes, InstMetaData),
    Ar2(Array2<f64>, InstMetaData),
}

#[derive(Debug)]
pub struct ProcessedData {
    name: String,
    format_name: String,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Action {
    source: String,
    //entity id
    target: String,
    //entity id
    name: String,
    is_variable: bool,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct State {
    ent: String,
    // entity id
    name: String,
    is_variable: bool,
}

// i'm super interested in seeing what the data structure for holding Actions, States, and Entities will look like!
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Entity {
    name: String,
    is_variable: bool,
}

#[derive(Debug)]
pub struct PhysicalEntity {
    instance_name: String,
    name: String,
    shape: GeometricShape,
    scale: u64, // log scale where 0 == subatomic
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct EventConjugate {
    actions: Vec<Action>,
    states: Vec<State>,
    entities: Vec<Entity>,
}

impl EventConjugate {
    pub fn new() -> Self {
        EventConjugate {
            actions: Vec::new(),
            states: Vec::new(),
            entities: Vec::new(),
        }
    }
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
pub fn compare_conjugate_similarity(conj1: &EventConjugate, conj2: &EventConjugate) -> f64 {
    unimplemented!() // todo. attempts to fuzzy match event conjugate to the instance. output is relative similarity to conj1 vers conj2 [0,1], bigger values => more similar
    // needs to pull apart similarities in the two conjugates
}

fn conjugate_similarity_matrix(vector: &Vec<&EventConjugate>) -> Array2<f64> {
    let n = vector.len();
    let mut ar = Array2::<f64>::ones((n, n));
    let mut done = HashMap::new();

    for ((i, j), elt) in ar.indexed_iter_mut() {
        if i != j {
            let mirror = done.get(&(j, i));
            match mirror {
                Some(m) => {
                    *elt = *m;
                    continue;
                }
                None => {
                    *elt = compare_conjugate_similarity(vector.get(i).unwrap(),
                                                        vector.get(j).unwrap());
                }
            }
        } else {
            continue;
            // *elt = 1. is unnecessary since we instantiated all values as 1.
        }
        done.insert((j, i), *elt);
    }

    ar
}

pub fn conjugate_union(conj1: &EventConjugate, conj2: &EventConjugate) -> EventConjugate {
    unimplemented!()
}

/// Causal rule. Since we can never know if we are not detecting some entity, we must treat even
/// our causal rules probabilistically – we can never prove sufficiency, just necessity.
///
/// Thus, our causal rules retain information on when these rules are violated (for example,
/// paper will light on fire, unless the paper is wet. The additional state of wetness precludes the
/// general causal rule that paper will burn when exposed to flame). These exceptions are generalized
/// by finding how similar a given scene is to the general rule versus the exception.
#[derive(Debug)]
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

#[derive(Debug)]
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
            let cur = compare_conjugate_similarity(&self.before, ev_before);
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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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
                    break
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


#[derive(Debug, Clone)]
pub struct Assertion {
    //format.name is the string key
    proofs: HashMap<String, Vec<Proof>>,
    id: usize,
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

#[derive(Clone)]
pub struct AssertionDiagnostic {}

impl AssertionDiagnostic {
// todo: cluster analysis on correlation matrix
// todo: outlier analysis on proof outputs
}

#[derive(Clone)]
pub struct AssertionContainer {
    name: String,
    semantic_shape: Option<SemanticShape>,
    // space of contained assertions
    assertions: Vec<Assertion>,
    diagnostics: Vec<AssertionDiagnostic>,
}

impl AssertionContainer {
    fn next_id(&self) -> usize { self.assertions.len() }
    pub fn add(&mut self, mut a: Assertion) -> bool {
        a.id = self.next_id();
        self.assertions.push(a);
        true
    }
}

pub struct AssertionMaster {
    containers: HashMap<String, AssertionContainer>, // this might become a b tree based on semantic overlap instead.
}

/// takes a (symmetrics) similarity matrix and clusters all of the values. Returns a vector of the
/// vectors of the indices for the original values for each cluster, e.g:
/// vec![vec![value_index1, value_index2...] // cluster 1 values, vec![value_indexN, ...] // cluster 2 values, ... // so on for each cluster]
///
/// current algorithm is the kodama crate's implementation of the hierarchical clustering work by Müllner
fn cluster_mat(ar: &Array2<f64>) -> Vec<Vec<usize>> {
    let dend = hierarchical_clustering(&mut condense_similarity_matrix(ar));
    // todo implement decision for K here or in hierarchical_clustering
    unimplemented!();
}

/// yields a triangle (not including diagonal) from the symmetric similarity matrix for kodama
fn condense_similarity_matrix(ar: &Array2<f64>) -> Vec<f64> {
    let debug = true;
    if debug {
        assert_eq!(ar.shape()[0], ar.shape()[1]);
    }
    let n = ar.shape()[0];
    let mut out = Vec::new();
    for i1 in 1..n - 1 {
        for i2 in i1..n {
            out.push(ar[[i1, i2]]);
        }
    }
    out
}

fn hierarchical_clustering(sim: &mut Vec<f64>) -> Dendrogram<f64> {
    let l = sim.len();
    let dend = linkage(sim, l, Method::Average);
    // todo decide on K here?
    unimplemented!();
}

/// first pass of generating assertions from a vector of instances.
fn generate_assertions<'a>(insts: Vec<&'a Instance>) -> Vec<Assertion> {
    let debug = true;

    fn set<'a>(e: &'a Event, a: &mut Vec<Assertion>, c: &mut HashMap<RelRef<'a>, HashSet<&'a Event>>) {
        let mut check_or_insert = |r: RelRef<'a>| {
            if c.contains_key(&r) {
                let mut s = c.get_mut(&r).unwrap();
                if !s.contains(e) {
                    s.insert(e);
                }
            } else {
                let mut s = HashSet::new();
                s.insert(e);
                c.insert(r, s);
            }
        };

        match e.before {
            Some(ref evconj) => {
                for v in evconj.actions.iter() {
                    check_or_insert(RelRef::Action(v));
                }
                for v in evconj.states.iter() {
                    check_or_insert(RelRef::State(v));
                }
                for v in evconj.entities.iter() {
                    check_or_insert(RelRef::Entity(v));
                }
            }
            None => (),
        };

        match e.after {
            Some(ref evconj) => {
                for v in evconj.actions.iter() {
                    check_or_insert(RelRef::Action(v));
                }
                for v in evconj.states.iter() {
                    check_or_insert(RelRef::State(v));
                }
                for v in evconj.entities.iter() {
                    check_or_insert(RelRef::Entity(v));
                }
            }
            None => (),
        };
    }

    /// the goal of this function is to detect (disprovable, specifically postulated) patterns in a series of events.
    let mut relation_assertion= |s: &HashSet<&'a Event>, out: &mut Vec<Assertion>| {
        /// take in a set of event conjugates and, for the non-None conjugates, form shared
        /// event / action / state system.
        let mut soft_rules = |vector: Vec<&'a Option<EventConjugate>>| -> Vec<EventConjugate> {
            let mut maximal = |v: Vec<&'a Option<EventConjugate>>| -> (Option<EventConjugate>, Vec<&'a EventConjugate>) {
                let mut maximal_union = EventConjugate::new();
                let mut unwrapped = Vec::new();

                let mut first = true;
                for item in v.iter() {
                    match item {
                        &&Some(ref evconj) => {
                            unwrapped.push(evconj);
                            if first {
                                maximal_union = evconj.clone();
                                first = false;
                            }
                            maximal_union = conjugate_union(&maximal_union, evconj);
                        }
                        &&None => ()
                    }
                }
                if first == true {
                    return (None, unwrapped);
                }
                (Some(maximal_union), unwrapped)
            };

            if debug {
                assert!(vector.len() > 0);
            }

            let mut conjs = Vec::new();

            let all_evs = match maximal(vector) {
                (Some(m), all_evs) => {
                    conjs.push(m);
                    all_evs
                },
                (None, all_evs) => all_evs
            };


            // next partition the events by clustering on conjugate similarity.
            let mut mat = conjugate_similarity_matrix(&all_evs);
            let i_cluster = cluster_mat(&mut mat);
            for clust in i_cluster {}

            conjs
        };

        let mut into_assertion = |before: EventConjugate, after: EventConjugate| -> Assertion {
            unimplemented!()
//            Assertion {
//
////            //format.name is the string key
////            proofs: HashMap < String,
////            Vec<Proof> >,
////            id: usize,
////            // should equal ID of AssertionContainer
////            container_name: String,
////            // duration since epoch of last_diagnostic
////            last_diagnostic: Duration,
////            updated_since: bool, // updated since last diagnostic
//            }
        };

        let mut befores = Vec::new();
        let mut afters = Vec::new();
        for v in s.iter() {
            befores.push(&v.before);
            afters.push(&v.after);
        }

        let before_patterns = soft_rules(befores);
        let after_patterns = soft_rules(afters);

        for b in before_patterns.iter() {
            for a in after_patterns.iter() {
                out.push(into_assertion(b.clone(), a.clone()));
            }
        }
    };

    let mut out: Vec<Assertion> = Vec::new();
    let mut crosscheck: HashMap<RelRef, HashSet<&'a Event>> = HashMap::new();
    for inst in insts.iter() {
        for e in inst.events.iter() {
            set(e, &mut out, &mut crosscheck);
        }
    }

    for rel_events in crosscheck.values() {
        relation_assertion(rel_events, &mut out);
    }

    out
}

fn generate_diagnostics(inst: &Vec<Assertion>) -> Vec<AssertionDiagnostic> {
    unimplemented!() // todo
}

#[derive(Debug, Eq, PartialEq, Hash)]
enum Relation {
    Entity(Entity),
    Action(Action),
    State(State),
}

#[derive(Debug, Eq, PartialEq, Hash)]
enum RelRef<'a> {
    Entity(&'a Entity),
    Action(&'a Action),
    State(&'a State),
}

/// These are what each line could represent.
enum MinParseItem {
    A(Assertion),
    C(CausalRule),
    E(Event),
}

/// splits a string at its commas and trims whitespace.
fn spl<'a>(s: &'a str) -> Vec<&'a str> {
    let mut vec: Vec<&str> = Vec::new();
    for val in s.split(",") {
        vec.push(val.trim());
    }
    vec
}

/// parses an MRT (minimal relations text) string.
fn string_min_parse<'a>(s: &'a str, e: &mut HashMap<String, Vec<String>>, r: &'a mut HashMap<String, (String, String)>, filepath: &str) -> Option<MinParseItem> {
    quick_error! {
        #[derive(Debug)]
        pub enum KidError {
            UndefinedComplexRelation {
                description("Undefined complex relation")
            }
        }
    }

    lazy_static! {
        pub static ref RELATION: Regex = Regex::new("^[[a-zA-Z0-9_]]*[(]").unwrap();
        pub static ref STARTPAREN: Regex = Regex::new("[(]").unwrap();
        pub static ref ENDPAREN : Regex = Regex::new("[)]").unwrap();
        pub static ref ARROW : Regex = Regex::new("->").unwrap();
        pub static ref COLON : Regex = Regex::new(":").unwrap();

        pub static ref CORE_PHRASES: [String; 3] = ["action".to_string(),
        "state".to_string(), "entity".to_string()];
    }


    // trivial implementation: relation exists if MRT file suggests it does, with probability based on
// trust of the file.
    let debug = true;

    if debug {
        println!("Parsing string: {}", s);
    }

    /// checks if the relation &str matches "action" "state" or "entity"
    fn primitive(s: &str) -> bool {
        match s {
            "action" => true,
            "state" => true,
            "entity" => true,
            _ => false
        }
    };

    /// returns the relation string and parameter string for a given relation.
    /// For example, "action(kelly, lauren, hugs)" -> ("action", "kelly, lauren, hugs")
    fn rel_and_params<'r>(m: regex::Match, s: &'r str, debug: bool) -> (&'r str, &'r str) {
        let fs = m.start();
        let sp = fs + STARTPAREN.find(&s[fs..]).unwrap().start(); // start parenthesis
        if debug {
            println!("relation start index: {}", fs);
            println!("relation end index: {}", sp);
        }
        let relation = &s[fs..sp]; //relation
        if debug {
            println!("relation: {}", relation);
            println!("string: {}", s);
        }
        let params = &s[sp + 1..ENDPAREN.find(&s[sp + 1..]).unwrap().start() + sp + 1]; //relation values
        (relation, params)
    };

    /// parses relations of both sides of a colon or arrow in an MRT line
    fn parsplit(es: &str, split: usize, r: &mut HashMap<String, (String, String)>, debug: bool) -> (Result<Vec<Relation>, KidError>, Result<Vec<Relation>, KidError>) {
        let (s1, s2) = es.split_at(split);
        if s2.starts_with(":") {
            (parse_relations(s1, r, debug),
             parse_relations(&s2[1..].trim(), r, debug))
        } else if s2.starts_with("->") {
            (parse_relations(s1, r, debug),
             parse_relations(&s2[2..].trim(), r, debug))
        } else {
            panic!("Bad separator")
        }
    }

    /// takes in a series of relations (usually one side of a color or arrow in an MRT line) and
    /// recursively parses those relations until they are reduced into simple relations (entity,
    /// state, and action declarations).
    fn parse_relations<'b>(s: &'b str, r: &mut HashMap<String, (String, String)>, debug: bool) -> Result<Vec<Relation>, KidError> {
        /// Add implicitly declared entities (entities only referenced by actions and states).
        fn add_impl_entities<'params>(rel: &str, params: &'params str, ents: &
        mut HashSet<&'params str>) -> Vec<Entity> {
            fn is_entity(rel: &str, i: i32, ent: &str) -> bool {
                match rel {
                    "action" => {
                        match i {
                            0 => true,
                            1 => true,
                            2 => false,
                            _ => panic!("Unexpected i")
                        }
                    }
                    "state" => {
                        match i {
                            0 => true,
                            1 => false,
                            _ => panic!("Unexpected i")
                        }
                    }
                    "entity" => {
                        assert! {i == 0}
                        true
                    }
                    _ => false, // wait until parse_relations recurses into just simple relations.
                }
            }

            let mut out = Vec::new();
            let mut i = 0;
            for ent in spl(params) {
                if !ents.contains(ent) && is_entity(rel, i, ent) {
                    out.push(
                        Entity {
                            name: ent.to_string(),
                            is_variable: ent.starts_with("_"),
                        }
                    );
                    ents.insert(ent);
                } // todo deal with entity variables
                i = i + 1;
            }
            out
        }

        let prim_rel = |relstr: &str, parstr: &str| -> Relation {
            if debug {
                println!("Parsing primitive relation");
            }
            match relstr {
                "action" => {
                    let splitted = spl(parstr);
                    Relation::Action(
                        Action {
                            source: splitted[0].to_string(),
                            target: splitted[1].to_string(),
                            name: splitted[2].to_string(),
                            is_variable: splitted[2].starts_with("_"),
                        }
                    )
                }
                "state" => {
                    let splitted = spl(parstr);
                    Relation::State(
                        State {
                            ent: splitted[0].to_string(),
                            name: splitted[1].to_string(),
                            is_variable: splitted[1].starts_with("_"),
                        }
                    )
                }
                "entity" => {
                    Relation::Entity(
                        Entity {
                            name: parstr.to_string(),
                            is_variable: parstr.starts_with("_"),
                        }
                    )
                }
                _ => panic!("Non-primary assertion passed to prim_rel. Unable to parse.")
            }
        };

        /// parse a single complex relation by recursively calling parse_relations on
        /// the definition of the complex relation, with the values in the relation imputed into the
        /// definition.
        let mut parse_relation = |relstr: &str, parstr: &str, r: &mut HashMap<String, (String, String)>| -> Result<Vec<Relation>, KidError> {
            if !r.contains_key(relstr) {
                if debug { println!("Undefined complex relation: {}", relstr) }
                return Err(KidError::UndefinedComplexRelation);
            }

            let mapped = {
                // "mapped" is the definition of the complex relation with the variables filled in.

                // since we've seen this key before, let's unpack it.
                let &(ref or, ref un) = r.get(relstr).unwrap();
                if debug {
                    println!("original parameters of complex relation: {}", or);
                    println!("un-imputed definition of complex relation: {}", un);
                }
                let original = or.to_string();
                let unpacked = un.to_string();
                let mut parmap = HashMap::new();
                let keys = spl(&original);
                let vals = spl(parstr);
                if debug { assert_eq!(keys.len(), vals.len()) };
                for i in 0..keys.len() {
                    parmap.insert(keys[i], vals[i]);
                }

                let n = |s: &str, i: usize| -> usize {
                    // remainder of string or next alphanumeric / underscore sequence
                    match RELATION.find(&s[i..]) {
                        Some(relmatch) => relmatch.start(),
                        None => s.len()
                    }
                };

                let mut out = "".to_string();

                for pmatch in STARTPAREN.find_iter(un) {
                    let startp = pmatch.start();

                    out = out + &un[..startp] + "("; // start with the relation and paren

                    let endp = startp + ENDPAREN.find(&un[startp..]).unwrap().start();
                    for defparam in spl(&un[startp + 1..endp]).into_iter() {
                        if parmap.contains_key(defparam) {
                            out = out + " " + parmap.get(defparam).unwrap() + ",";
                        } else {
                            out = out + " " + defparam + ",";
                        }
                    }
                    out = out[..out.len() - 1].to_string() + &un[endp..n(un, endp)];
                }
                if debug { println!("Mapped: {}", out) };
                out.to_string()
            };

            parse_relations(&mapped, r, debug) // recurse until the relations have all been simplified.
        };

        if debug {
            println!("in parse_relations");
        }
        let mut vec = Vec::new();
        let mut ents = HashSet::new();
        if debug {
            println!("Find_iter through string {}", s);
        }
        for m in RELATION.find_iter(s) {
            if debug {
                println!("iterating");
            }
// for each m we know that we have chars, a start paren, chars, and an end paren.
            let (relation, params) = rel_and_params(m, s, debug);

            for implicit_entity in add_impl_entities(relation, params, &mut ents) {
                vec.push(Relation::Entity(implicit_entity))
            }

            if primitive(relation) {
                if debug {
                    println!("Parsing primitive relation: {}", relation);
                }
                vec.push(prim_rel(relation, params));
            } else {
                match parse_relation(relation, params, r) {
                    Ok(resp) => vec.extend(resp),
                    Err(KidError::UndefinedComplexRelation) => {
                        if vec.len() == 0 {
                            // todo i think this is where we're messing up in parsing right-side references to complex relations.
                            // define the complex relation then!
                            r.insert(relation.to_string(), (params.to_string(), s[COLON.find(&s).unwrap().start() + 1..].to_string()));
                        } else {
                            panic!("Undeclared complex relation")
                        }
                    }
                };
            }
        }
        println!("{:?}", vec);
        Ok(vec)
    };

    let mut parse_event = |es: &str, r: &mut HashMap<String, (String, String)>| -> Event {
        fn vec_to_conjugate(v: Vec<Relation>) -> EventConjugate {
            let mut o = EventConjugate {
                actions: Vec::new(),
                states: Vec::new(),
                entities: Vec::new(),
            };
            for r in v.into_iter() {
                match r {
                    Relation::Action(a) => o.actions.push(a),
                    Relation::State(s) => o.states.push(s),
                    Relation::Entity(e) => o.entities.push(e)
                }
            };
            o
        }

        let time_split = ARROW.find(es).unwrap().start();
        if debug {
            println!("Finding relations for two substrings: {} and {}", &es[..time_split], &es[time_split..]);
        }
        let (bvec, avec) = parsplit(es, time_split, r, debug);

        if debug { println!("Relation vectors for event found. Converting to EventConjugate and returning Event.") };

        let before = match bvec {
            Ok(bev) => Some(vec_to_conjugate(bev)),
            Err(error) => {
                if debug {
                    println!("Error in parse_event: {}", error);
                }
                None
            }
        };

        let after = match avec {
            Ok(afv) => Some(vec_to_conjugate(afv)),
            Err(error) => {
                if debug {
                    println!("Error in parse_event: {}", error);
                }
                None
            }
        };

        Event { before, after }
    };

    let mut parse_assertion = |s: &str, r: &mut HashMap<String, (String, String)>| -> Assertion {
        let proof = Proof {
            format: Format {
                name: "MRT".to_string(),
                processed_from: vec![InstanceData::Pat(PathBuf::from(filepath), InstMetaData {
                    name: "MRT".to_string(),
                    kind: "txt".to_string(),
                })],
            },
            times_used: 0,
            avg_runtime: Duration::new(0, 0),
            tier: 0,
            conditions: vec![parse_event(&COLON.replace_all(s, "->").into_owned(), r)],
        };

        let mut proofs = HashMap::new();
        proofs.insert(proof.format.name.clone(), vec![proof]);

        Assertion {
            container_name: "core".to_string(),
            id: 0, // overwritten when added to assertion container.
            proofs,
            updated_since: true,
            last_diagnostic: Duration::new(0, 0),
            //duration: SystemTime::now().duration_since(UNIX_EPOCH).unwrap(),
        }
    };

    enum Ce {
        A,
        // assertion
        R,
        // complex relation declaration
//        C, // causal rule
        E,
        // entity
        Bad,
    }

    /// true iff string is a definition for a complex relation. Inputs the complex relation
   /// into the hashmap for storing known complex relations and returns true, or returns false
   /// if the string doesn't represent a complex relation definition.
   ///
   /// rules of a complex relation:
   /// 1.) left- and right-hand sides of the relation must be separated by colon.
   /// 2.) left-hand side must have exactly one relation, which should be non-primitive.
    let mut complex_def = |st: &str, r: &mut HashMap<String, (String, String)>| -> bool {
        match COLON.find(st) {
            Some(colonmatch) => {
                let relstart = RELATION.find(st).unwrap().start(); // is this always 0 or is there sometimes whitespace?
                let colstart = COLON.find(st).unwrap().start();
                let parenstart = STARTPAREN.find(st).unwrap().start();
                if !primitive(&st[relstart..parenstart]) &&
                    RELATION.find_iter(&st[..COLON.find(st).unwrap().start()]).count() == 1 {
                    let (rel, params) = rel_and_params(RELATION.find(st).unwrap(), &st[relstart..ENDPAREN.find(st).unwrap().start() + 1], debug);
                    r.insert(rel.to_string(), (params.to_string(), st[colonmatch.start() + 1..].to_string()));
                    return true;
                }
                false
            }
            None => false
        }
    };

    let mut c = |st: &'a str, r: &mut HashMap<String, (String, String)>| -> Ce {
        let n = |v: Option<regex::Match>| -> bool {
            match v {
                Some(_) => true,
                None => false
            }
        };


        if debug {
            println!("in c() with str of {}", st);
            println!("COLON match: {}", n(COLON.find(st)));
            println!("ARROW match: {}", n(ARROW.find(st)));
        }

        if n(COLON.find(st)) && !n(ARROW.find(st)) {
            if complex_def(st, r) {
                Ce::R // already parsed by complex_def.
            } else {
                Ce::A
            }
        } else if !n(COLON.find(st)) && n(ARROW.find(st)) {
            Ce::E
        } else {
            Ce::Bad
        }
    };

    match c(&s, r) {
        Ce::A => {
            if debug {
                println!("Assertion detected. Parsing.")
            }
            Some(MinParseItem::A(parse_assertion(s, r)))
        }
        Ce::E => {
            if debug {
                println!("Event detected. Parsing.")
            }
            Some(MinParseItem::E(parse_event(s, r)))
        }
        Ce::R => {
            if debug {
                println!("Complex relation parsed.")
            }
            None
        }
        Ce::Bad => {
            if debug {
                println!("Bad string detected.")
            }
            None
        }
    }
}

/// Usually kid will be constantly predicting a whole bunch of things and processing those
/// predictions in a lot of ways (e.g. to self-optimize and to decide how to act). But, for the
/// minimal case, we're just looking at a simple prediction
fn minimal_predict_string(before: String, am: AssertionMaster) -> String {
    unimplemented!(); // todo
}

fn parse_minimal(fname: &Path, name: String) -> Instance {
    let debug = true;

    fn read_lines(fname: &Path) -> Vec<String> {
        fn remove_comments(s: &str) -> Option<&str> {
            match s.find("//") {
                Some(index) => return remove_comments(&s[..index]),
                None => {
                    if !s.contains("->") & &!s.contains(":") {
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
            let m = string_min_parse(&astr, &mut ents, &mut recursive_relation_defs, &fname.to_str().unwrap());
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

    println!("{:?}", va);
    println!("{:?}", vc);
    println!("{:?}", ve);

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

        let a3: Array1<f64> = array![ - 1., 0., 2.];
        let a4: Array1<f64> = array![ - 2., 0., 2.];
        assert_eq!(1., euc_dist(&a3, &a4)); // negative numbers

        let point_arrays = [
            a1, a2, a3, a4,
            array![ - 2435345., 123412423., -1999.] as Array1<f64>,
            array![0.01, 0.02, 0.03] as Array1<f64>, // decimals
            array![9999999999999., 9999999999999., - 9999999999999.] as Array1<f64>, // larger nums
            array![0., 0., 0.] as Array1<f64>,
            array![5., 5., 6.] as Array1<f64>
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
                assert!(cos_dist(v1, v1).is_nan() || 0. == (10000000. * cos_dist(v1, v1)).round());
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

    let mut min_inst = parse_minimal(min_txt_path, "minimal".to_string());

    let mut assertions = generate_assertions(vec![&min_inst]);
    match min_inst.assertions.clone() {
        Some(v) => {
            assertions.extend(v)
        }
        None => (),
    };
    let mut diagnostics = generate_diagnostics(&assertions);
// diagnostics will be essential for optimizing assertion calculation.

    let mut
    inst_vec = vec![min_inst];
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
        assertions: Vec::new(),
        diagnostics,
        semantic_shape: Some(shape_from_instances(&inst_vec)),
    };
    for ass in assertions.into_iter() {
        core_ac.add(ass);
    }

// we now have all of our assertions in a single container. put it in the master and we're good!
    am.containers.insert(core_ac.name.clone(), core_ac);

//next up we want to predict something.
    let ptext = "state(match, burning) + state(newspaper, wet) + symmetric_action(newspaper, match, touching)".to_string();
    println!("{}", minimal_predict_string(ptext, am));
}