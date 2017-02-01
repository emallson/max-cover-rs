#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate docopt;
extern crate rustc_serialize;
extern crate rand;
#[macro_use]
extern crate rplex;

use std::collections::{BTreeSet, BTreeMap};
use std::fs::File;
use docopt::Docopt;
use rand::{thread_rng, sample, Rng};
use rand::distributions::{Range, IndependentSample};
use rplex::*;

const USAGE: &'static str = "
Constructs and (optimally) solves Maximum k-Coverage instances.

Usage:
    cover generate <output> <elements> <sets> [--max-size <size>]
    cover solve <input> <k> [--threads <t>] [--write <name>]
    cover (-h | --help)
    cover --version

Options:
    -h --help           Show this screen.
    --version           Show version.
    --threads <t>       Set number of threads used.
    --max-size <size>   Maximum set size.
    --write <name>      Write solution to <name>.
";

#[derive(Debug, RustcDecodable)]
struct Args {
    cmd_generate: bool,
    cmd_solve: bool,
    arg_elements: Option<usize>,
    arg_sets: Option<usize>,
    // arg_density: Option<f32>,
    arg_output: Option<String>,
    arg_input: Option<String>,
    arg_k: Option<usize>,
    flag_threads: Option<usize>,
    flag_max_size: Option<usize>,
    flag_write: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Instance {
    ground: BTreeSet<usize>,
    sets: Vec<BTreeSet<usize>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Solution {
    objective: f64,
    sol: Vec<usize>,
}

fn do_generate(num_elements: usize, num_sets: usize, max_size: Option<usize>) -> Instance {
    let ground: BTreeSet<usize> = (0..num_elements).collect();
    let mut rng = thread_rng();
    let range = Range::new(1, max_size.unwrap_or(num_elements + 1));

    let mut sets = BTreeSet::new();

    for _ in 0..num_sets {
        loop {
            let size = range.ind_sample(&mut rng);
            let set: BTreeSet<usize> =
                sample(&mut rng, &ground, size).into_iter().map(|&x| x).collect();
            if !sets.contains(&set) {
                sets.insert(set);
                break;
            }
        }
    }

    Instance {
        ground: ground,
        sets: sets.into_iter().collect(),
    }
}

fn write(inst: Instance, fname: &str) {
    let mut f = File::create(fname).unwrap();
    serde_json::to_writer(&mut f, &inst).unwrap();
}

fn read(fname: &str) -> Result<Instance, serde_json::Error> {
    let f = File::open(fname).unwrap();
    serde_json::from_reader(&f)
}

fn do_solve(inst: Instance, k: usize, threads: Option<usize>, write: Option<String>) {
    let mut env = Env::new().unwrap();
    env.set_param(EnvParam::Threads(threads.unwrap_or(1) as u64)).unwrap();
    env.set_param(EnvParam::ScreenOutput(true)).unwrap();
    let mut prob = Problem::new(&env, "maxcover").unwrap();
    let mut containment = BTreeMap::new();

    prob.set_objective_type(ObjectiveType::Maximize).unwrap();

    let element_vars = inst.ground
        .iter()
        .map(|&x| {
            let name = format!("e{}", x);
            prob.add_variable(var!(name -> 1.0 as Binary)).unwrap()
        })
        .collect::<Vec<_>>();

    let set_vars = inst.sets
        .iter()
        .enumerate()
        .map(|(i, set)| {
            let name = format!("s{}", i);
            for element in set.iter() {
                containment.entry(element).or_insert_with(Vec::new).push(i);
            }
            prob.add_variable(var!(name -> 0.0 as Binary)).unwrap()
        })
        .collect::<Vec<_>>();

    for (&var, element) in element_vars.iter().zip(inst.ground.iter()) {
        let name = format!("cover{}", element);
        let mut con = con!(name: 0.0 <= sum containment.entry(element).or_insert_with(Vec::new).iter().map(|&i| &set_vars[i]));
        con.add_wvar(WeightedVariable::new_idx(var, -1.0));
        prob.add_constraint(con).unwrap();
    }
    prob.add_constraint(con!("cardinality": (k as f64) >= sum set_vars.iter())).unwrap();

    prob.write("problem.lp").unwrap();
    let sol = prob.solve().unwrap();
    let sol_sets = set_vars.into_iter()
        .filter(|&var| sol.variables[var] == VariableValue::Binary(true))
        .collect::<Vec<_>>();

    let out_sol = Solution {
        objective: sol.objective,
        sol: sol_sets,
    };

    println!("{:?}", out_sol);
    if let Some(fname) = write {
        serde_json::to_writer_pretty(&mut File::create(fname).unwrap(), &out_sol).unwrap();
    }
}

fn main() {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| {
            d.version(Some(env!("CARGO_PKG_VERSION").to_string()))
                .decode()
        })
        .unwrap_or_else(|e| e.exit());

    if args.cmd_generate {
        let inst = do_generate(args.arg_elements.unwrap(),
                               args.arg_sets.unwrap(),
                               args.flag_max_size);
        write(inst, &args.arg_output.unwrap());
    } else if args.cmd_solve {
        let inst = read(&args.arg_input.unwrap()).unwrap();
        do_solve(inst,
                 args.arg_k.unwrap(),
                 args.flag_threads,
                 args.flag_write);
    } else {
        panic!("no command given");
    }
}
