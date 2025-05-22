from algos.algo import Algo
from algos.policy import Policy, BasePolicy

from algos.q import Q
from algos.double import Double
from algos.wdq import WDQ

from algos.categorical_q import CategoricalQ
from algos.categorical_double  import CategoricalDouble
from algos.addq import ADDQ

__all__ = [
    Algo, Policy, BasePolicy,
    Q,
    Double,
    WDQ,
    CategoricalQ,
    CategoricalDouble,
    ADDQ
]