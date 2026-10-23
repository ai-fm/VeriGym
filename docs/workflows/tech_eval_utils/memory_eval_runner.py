from verigym.frameworks.stormpy.stormpy_utils import load_stormpy_model
from verigym.environments.frameworkexplicitenv import FrameworkExplicitEnv

import argparse
import json
import os
import psutil
import stormpy



parser = argparse.ArgumentParser()
parser.add_argument("task")
parser.add_argument("filename")
parser.add_argument("--constants", default="")
args = parser.parse_args()

if args.task == "stormpy":

    process = psutil.Process(os.getpid())

    before = process.memory_info().rss

    program = stormpy.parse_prism_program(args.filename)
    program = program.define_constants(
        stormpy.parse_constants_string(program.expression_manager, args.constants)
    )
    program = program.label_unlabelled_commands({})
    options = stormpy.BuilderOptions()
    options.set_build_state_valuations()
    options.set_build_choice_labels()
    options.set_build_with_choice_origins()
    options.set_build_all_reward_models()
    options.set_build_all_labels()
    
    mdp = stormpy.build_sparse_model_with_options(program, options)

    after = process.memory_info().rss

    print(json.dumps({
        "before": before,
        "after": after,
        "change": after - before,
    }))

if args.task == "env":

    process = psutil.Process(os.getpid())

    before_mdp = process.memory_info().rss
    
    mdp = load_stormpy_model(
        prismpath=args.filename,
        constants=args.constants
    )
    before = process.memory_info().rss

    env = FrameworkExplicitEnv.from_stormpy(mdp)

    after = process.memory_info().rss

    print(json.dumps({
        "before": before,
        "after": after,
        "change": after - before,
        "change_mdp": after - before_mdp,
    }))