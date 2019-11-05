
import experiments1 as experiments

_, english = experiments.verb_forgetting_conditions(m=.5, r=.5, e=.2, s=.8)
print(english)
_, german = experiments.verb_forgetting_conditions(m=.5, r=.5, e=.2, s=0)
print(german)
