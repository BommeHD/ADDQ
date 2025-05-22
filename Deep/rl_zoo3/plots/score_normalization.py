"""
Min and Max score for each env for normalization when plotting.
Min score corresponds to random agent.
Max score corresponds to acceptable performance, for instance
human level performance in the case of Atari games.
"""

from typing import NamedTuple

import numpy as np


class ReferenceScore(NamedTuple):
    env_id: str
    min: float
    max: float


reference_scores = [
    # Mujoco Envs based on highest/lowest seen
    ReferenceScore("Ant-v4", 0, 8000),
    ReferenceScore("HalfCheetah-v4", 4000, 18000),
    ReferenceScore("Hopper-v4", 0, 400),
    ReferenceScore("Walker2d-v4", 0, 7000),
    ReferenceScore("Humanoid-v4", 500,9000),
    # Atari Envs based on https://arxiv.org/abs/2003.13350
    ReferenceScore("AlienNoFrameskip-v4",227.80,7127.70),
    ReferenceScore("AmidarNoFrameskip-v4",5.80,1719.50),
    ReferenceScore("AssaultNoFrameskip-v4",222.40,742.00),
    ReferenceScore("AsterixNoFrameskip-v4",210.00,8503.30),
    ReferenceScore("AsteroidsNoFrameskip-v4",719.10,47388.70),
    ReferenceScore("AtlantisNoFrameskip-v4",12850.00,29028.10),
    ReferenceScore("BankHeistNoFrameskip-v4",14.20,753.10),
    ReferenceScore("BattleZoneNoFrameskip-v4",2360.00,37187.50),
    ReferenceScore("BeamRiderNoFrameskip-v4",363.90,16926.50),
    ReferenceScore("BerzerkNoFrameskip-v4",123.70,2630.40),
    ReferenceScore("BowlingNoFrameskip-v4",23.10,160.70),
    ReferenceScore("BoxingNoFrameskip-v4",0.10,12.10),
    ReferenceScore("BreakoutNoFrameskip-v4",1.70,30.50),
    ReferenceScore("CentipedeNoFrameskip-v4",2090.90,12017.00),
    ReferenceScore("ChopperCommandNoFrameskip-v4",811.00,7387.80),
    ReferenceScore("CrazyClimberNoFrameskip-v4",10780.50,35829.40),
    ReferenceScore("DefenderNoFrameskip-v4",2874.50,18688.90),
    ReferenceScore("DemonAttackNoFrameskip-v4",152.10,1971.00),
    ReferenceScore("DoubleDunkNoFrameskip-v4",-18.60,-16.40),
    ReferenceScore("EnduroNoFrameskip-v4",0.00,860.50),
    ReferenceScore("FishingDerbyNoFrameskip-v4",-91.70,-38.70),
    ReferenceScore("FreewayNoFrameskip-v4",0.00,29.60),
    ReferenceScore("FrostbiteNoFrameskip-v4",65.20,4334.70),
    ReferenceScore("GopherNoFrameskip-v4",257.60,2412.50),
    ReferenceScore("GravitarNoFrameskip-v4",173.00,3351.40),
    ReferenceScore("HeroNoFrameskip-v4",1027.00,30826.40),
    ReferenceScore("IceHockeyNoFrameskip-v4",-11.20,0.90),
    ReferenceScore("JamesbondNoFrameskip-v4",29.00,302.80),
    ReferenceScore("KangarooNoFrameskip-v4",52.00,3035.00),
    ReferenceScore("KrullNoFrameskip-v4",1598.00,2665.50),
    ReferenceScore("KungFuMasterNoFrameskip-v4",258.50,22736.30),
    ReferenceScore("MontezumaRevengeNoFrameskip-v4",0.00,4753.30),
    ReferenceScore("MsPacmanNoFrameskip-v4",307.30,6951.60),
    ReferenceScore("NameThisGameNoFrameskip-v4",2292.30,8049.00),
    ReferenceScore("PhoenixNoFrameskip-v4",761.40,7242.60),
    ReferenceScore("PitfallNoFrameskip-v4",-229.40,6463.70),
    ReferenceScore("PongNoFrameskip-v4",-20.70,14.60),
    ReferenceScore("PrivateEyeNoFrameskip-v4",24.90,69571.30),
    ReferenceScore("QbertNoFrameskip-v4",163.90,13455.00),
    ReferenceScore("RiverraidZoneNoFrameskip-v4",1338.50,17118.00),
    ReferenceScore("RoadRunnerNoFrameskip-v4",11.50,7845.00),
    ReferenceScore("RobotankNoFrameskip-v4",2.20,11.90),
    ReferenceScore("SeaquestNoFrameskip-v4",68.40,42054.70),
    ReferenceScore("SkiingNoFrameskip-v4",-17098.10,-4336.90),
    ReferenceScore("SolarisNoFrameskip-v4",1236.30,12326.70),
    ReferenceScore("SpaceInvadersNoFrameskip-v4",148.00,1668.70),
    ReferenceScore("StarGunnerNoFrameskip-v4",664.00,10250.00),
    ReferenceScore("SurroundNoFrameskip-v4",-10.00,6.50),
    ReferenceScore("TennisNoFrameskip-v4",-23.80,-8.30),
    ReferenceScore("TimePilotNoFrameskip-v4",3568.00,5229.20),
    ReferenceScore("TutankhamNoFrameskip-v4",11.40,167.60),
    ReferenceScore("UpNDownNoFrameskip-v4",533.40,11693.20),
    ReferenceScore("VentureNoFrameskip-v4",0.00,1187.50),
    ReferenceScore("VideoPinballNoFrameskip-v4",0.00,17667.90),
    ReferenceScore("WizardOfWorNoFrameskip-v4",563.50,4756.50),
    ReferenceScore("YarsRevengeNoFrameskip-v4",3092.90,54576.90),
    ReferenceScore("ZaxxonNoFrameskip-v4",32.50,9173.30)
]
# Normalization values taken from paper "Agent 57: Outperforming the Atari Human Benchmark"

# Alternative scaling
# Min is a poorly optimized algorithm
# reference_scores = [
#     ReferenceScore("HalfCheetahBulletEnv-v0", 1000, 3000),
#     ReferenceScore("AntBulletEnv-v0", 1000, 3500),
#     ReferenceScore("HopperBulletEnv-v0", 1000, 2500),
#     ReferenceScore("Walker2DBulletEnv-v0", 500, 2500),
# ]

min_max_score_per_env = {reference_score.env_id: reference_score for reference_score in reference_scores}


def normalize_score(score: np.ndarray, env_id: str) -> np.ndarray:
    """
    Normalize score to be in [0, 1] where 1 is maximal performance.

    :param score: unnormalized score
    :param env_id: environment id
    :return: normalized score
    """
    if env_id not in min_max_score_per_env:
        raise KeyError(f"No reference score for {env_id}")
    reference_score = min_max_score_per_env[env_id]
    return (score - reference_score.min) / (reference_score.max - reference_score.min)


# From rliable, for atari games:
#
# RANDOM_SCORES = {
#  'Alien': 227.8,
#  'Amidar': 5.8,
#  'Assault': 222.4,
#  'Asterix': 210.0,
#  'BankHeist': 14.2,
#  'BattleZone': 2360.0,
#  'Boxing': 0.1,
#  'Breakout': 1.7,
#  'ChopperCommand': 811.0,
#  'CrazyClimber': 10780.5,
#  'DemonAttack': 152.1,
#  'Freeway': 0.0,
#  'Frostbite': 65.2,
#  'Gopher': 257.6,
#  'Hero': 1027.0,
#  'Jamesbond': 29.0,
#  'Kangaroo': 52.0,
#  'Krull': 1598.0,
#  'KungFuMaster': 258.5,
#  'MsPacman': 307.3,
#  'Pong': -20.7,
#  'PrivateEye': 24.9,
#  'Qbert': 163.9,
#  'RoadRunner': 11.5,
#  'Seaquest': 68.4,
#  'UpNDown': 533.4
# }
#
# HUMAN_SCORES = {
#  'Alien': 7127.7,
#  'Amidar': 1719.5,
#  'Assault': 742.0,
#  'Asterix': 8503.3,
#  'BankHeist': 753.1,
#  'BattleZone': 37187.5,
#  'Boxing': 12.1,
#  'Breakout': 30.5,
#  'ChopperCommand': 7387.8,
#  'CrazyClimber': 35829.4,
#  'DemonAttack': 1971.0,
#  'Freeway': 29.6,
#  'Frostbite': 4334.7,
#  'Gopher': 2412.5,
#  'Hero': 30826.4,
#  'Jamesbond': 302.8,
#  'Kangaroo': 3035.0,
#  'Krull': 2665.5,
#  'KungFuMaster': 22736.3,
#  'MsPacman': 6951.6,
#  'Pong': 14.6,
#  'PrivateEye': 69571.3,
#  'Qbert': 13455.0,
#  'RoadRunner': 7845.0,
#  'Seaquest': 42054.7,
#  'UpNDown': 11693.2
# }
