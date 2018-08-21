# kid

[![Build Status](https://travis-ci.org/DominicBurkart/kid.svg?branch=master)](https://travis-ci.org/DominicBurkart/kid)
[![Coverage Status](https://coveralls.io/repos/github/DominicBurkart/kid/badge.svg)](https://coveralls.io/github/DominicBurkart/kid)
[![codecov](https://codecov.io/gh/DominicBurkart/kid/branch/master/graph/badge.svg)](https://codecov.io/gh/DominicBurkart/kid)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FDominicBurkart%2Fkid.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2FDominicBurkart%2Fkid?ref=badge_shield)


An inference engine in rust! Currently in development.

Use case: Given a series of actions, states, and entities, predict what
will happen next. For example, given a lit match touching dry paper,
predict that the paper will catch on fire.

Notable features:
- use of human-readable, high-level input: all observations are
described as a series of actions, states, and entities.
- understandable inferences used throughout: rules generated based on
input are also expressed as a series of actions, states, and entities.

Beyond the minimal use case:
- rule chaining: rules can be defined as a set of other rules.
- use of word embeddings (here fastText) to generalize learned rules
onto similar cases.
- self-optimization: sets of classifiers are trained for popular rules,
and both accuracy and speed are used to select the classifier used.
- rule reproduction: if different classifiers yield different output for
the same rule, the program splits the rule and determines which new rule
works best in each application of the previous, single rule.

Upcoming tasks:
- restructure assertions (particularly how they hold proofs)
- restructure Formats / InstanceData and how they're made to clarify
data sources.

## License
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FDominicBurkart%2Fkid.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2FDominicBurkart%2Fkid?ref=badge_large)