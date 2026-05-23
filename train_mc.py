import warnings, sys
warnings.filterwarnings('ignore')
import numpy as np
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import MORecordEpisodeStatistics
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS

# Con gamma=1.0: "no hacer nada" 200 pasos = [-200,0,0]  (claramente malo para r_t)
# Con gamma=0.99: retorno descontado = -86.6 (el agente lo consideraba "bueno")
GAMMA = 1.0
REF   = np.array([-201.0, -201.0, -201.0])
TOTAL = 60_000
PER_ITER = 30_000   # 2 iteraciones grandes: cada peso tiene 30k pasos para aprender

print(f"GPI-LS Mountain Car | gamma={GAMMA} | {TOTAL:,} steps | {PER_ITER:,}/iter", flush=True)

env      = mo_gym.make('mo-mountaincar-v0')
env      = MORecordEpisodeStatistics(env, gamma=GAMMA)
eval_env = mo_gym.make('mo-mountaincar-v0')

agent = GPILS(
    env,
    per=True,
    initial_epsilon=1.0,
    final_epsilon=0.05,
    epsilon_decay_steps=25_000,    # explora fuerte la primera mitad
    target_net_update_freq=200,    # actualizaciones frecuentes del target
    gradient_updates=3,            # era 10 → 3x más rápido
    log=False,
)
agent.train(
    total_timesteps=TOTAL,
    eval_env=eval_env,
    ref_point=REF,
    timesteps_per_iter=PER_ITER,
)
env.close(); eval_env.close()
print("Training done.", flush=True)

WEIGHTS = [
    (np.array([0.8,  0.1,  0.1]),  'Prioriza tiempo'),
    (np.array([0.34, 0.33, 0.33]), 'Balance'),
    (np.array([0.2,  0.4,  0.4]),  'Evitar aceleraciones'),
    (np.array([0.5,  0.25, 0.25]), 'Tiempo moderado'),
    (np.array([0.1,  0.45, 0.45]), 'Min. aceleraciones'),
]

print(f"\n{'Config':<28} {'r_t':>7} {'r_L':>7} {'r_R':>7}", flush=True)
results = []
for w, label in WEIGHTS:
    rets = []
    ev = mo_gym.make('mo-mountaincar-v0')
    for _ in range(5):
        obs, _ = ev.reset()
        r = np.zeros(3); done = False; s = 0
        while not done and s < 200:
            try:    a = agent.eval(obs, w)
            except: a = ev.action_space.sample()
            obs, rew, t, tr, _ = ev.step(a)
            r += rew; done = t or tr; s += 1
        rets.append(r)
    ev.close()
    m = np.mean(rets, axis=0)
    results.append(m)
    print(f"{label:<28} {m[0]:>7.1f} {m[1]:>7.1f} {m[2]:>7.1f}", flush=True)

best_rt = max(r[0] for r in results)
print(f"\nBest r_t: {best_rt:.1f}  (converged = r_t > -200)", flush=True)
print("CONVERGED" if best_rt > -199 else "NOT CONVERGED", flush=True)
sys.exit(0 if best_rt > -199 else 1)
